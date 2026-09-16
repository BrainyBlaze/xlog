//! Fixed-roster device-resident semantic action sampling.
//!
//! Inputs are bound before capture. Launch and replay do not observe the device;
//! terminal observation is explicit. RNG successors remain pending, not published
//! world state. Descriptor meanings belong to the retained semantic admission.

use crate::semantic_work::{ExecutionWork, ModelWorkEvent, ModelWorkKind, ModelWorkRecording};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    mem::size_of,
    sync::Arc,
};

use cudarc::driver::sys;
use xlog_core::{Result as XlogResult, XlogError};

use crate::cuda_graph::CapturedCudaGraph;
use crate::dlpack::DlpackManagedTensor;
use crate::launch::{CudaEnqueue, LaunchEnqueueError, LaunchRecorder};
use crate::memory::TrackedCudaSlice;
use crate::memory::{
    CudaColumn, DeviceAllocationProvenance, DeviceMemoryView, DeviceRead, GpuMemoryReservation,
};
use crate::provider::resident_schedule::{validate_execution_domain, ResidentExecutionDomain};
use crate::semantic_hypergraph::{
    material_bytes, material_u32, material_u64, SemanticMaterialReader, SemanticRootMaterial,
};
use crate::semantic_training_view::{
    SemanticSelectedTrainingView, SemanticTrainingObjective, SemanticTrainingViewArena,
    SemanticTrainingViewOrigin, SemanticTrainingViewOriginRecord, SemanticTrainingViewPort,
    SemanticTrainingViewRow, SemanticTrainingViewSelection,
};
use crate::{
    CudaFunction, CudaKernelProvider, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig,
    SemanticAdmission, SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticHypergraph,
    SemanticHypergraphCapacities, SemanticHypergraphError, SemanticInsertOutcome,
    SemanticRecordRole, SemanticRootHandle, SemanticRootSnapshot,
};

#[cfg(feature = "semantic-policy")]
type PreparedPolicyStorage = (
    PolicyBuffers,
    TrackedCudaSlice<u8>,
    TrackedCudaSlice<SemanticTransitionReceipt>,
    TrackedCudaSlice<DeviceState>,
);

/// Canonical 256-bit identity, preserving byte order without native-word conversion.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Identity256([u8; 32]);

impl Identity256 {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A native observer executed during cold task binding. Implementations belong to
/// the program executor, not the CUDA substrate. Python accepts executable XLOG
/// source through that executor, never caller-provided answers or receipts.
/// A successful observation must complete all submitted work before returning
/// host results. An error may follow device submission, so the calling session
/// is poisoned and cannot reuse its previous task binding.
pub trait SemanticTaskProgram: fmt::Debug + Send + Sync {
    fn observe(
        &self,
        provider: Arc<CudaKernelProvider>,
    ) -> Result<SemanticTaskObservation, SemanticTransitionError>;
}

/// Actual native observer output and the exact source/input/result custody used
/// to derive it. These bytes describe execution; they grant no application rights.
pub struct SemanticTaskObservation {
    pub program_source: Vec<u8>,
    pub input_bytes: Vec<u8>,
    pub result_bytes: Vec<u8>,
    pub expected_truth: [crate::SemanticTruth; 3],
}

/// Explicit nonnegative coefficients for query agreement and measured work.
/// Selection maximizes the resulting value, retaining the earlier candidate on
/// ties. The final return separately prices improvement, refusals, and spent work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticTaskScoring {
    pub correct_weight: u32,
    pub all_correct_weight: u32,
    pub work_weight: u32,
    pub improvement_weight: u32,
    pub refusal_weight: u32,
    pub spent_weight: u32,
}

impl SemanticTaskScoring {
    fn words(self) -> [u64; 6] {
        [
            self.correct_weight,
            self.all_correct_weight,
            self.work_weight,
            self.improvement_weight,
            self.refusal_weight,
            self.spent_weight,
        ]
        .map(u64::from)
    }

    fn validate(self) -> Result<(), SemanticTransitionError> {
        // Two learned candidates each perform at most two commands, attachments,
        // and defined truth changes. At most nine queries plus eight discarded
        // command/truth units contribute to spent work. Check the entire signed
        // return before upload.
        let value_span = 3 * u128::from(self.correct_weight)
            + u128::from(self.all_correct_weight)
            + 6 * u128::from(self.work_weight);
        let return_bound = value_span * u128::from(self.improvement_weight)
            + 2 * u128::from(self.refusal_weight)
            + 17 * u128::from(self.spent_weight);
        if return_bound > i64::MAX as u128 {
            return Err(publication_input_error(
                "task scoring can overflow the signed return",
            ));
        }
        Ok(())
    }
}

/// Three ordered query selections for the bounded transition ABI. Predicate,
/// arguments, observer program, eligibility, and scoring are application inputs.
/// This specification carries no authority; only the application controller may
/// attach an authorized task to its retained semantic session.
#[derive(Clone, Debug)]
pub struct SemanticTaskEvaluationSpec {
    /// Original statement records in the retained admission, in observer order.
    pub statement_records: [u32; 3],
    /// Admitted support records, each retaining its original statement association.
    pub allowed_support_records: Vec<u32>,
    pub program: Arc<dyn SemanticTaskProgram>,
    pub scoring: SemanticTaskScoring,
    /// Bit n permits Truth4 value n for the corresponding candidate query.
    pub admissible_truth_masks: [u8; 3],
}

/// Owner-validated cold read coverage for a task's queried semantic heads.
/// These records grant no use or mutation rights and are not current device
/// query receipts. Runtime feedback must still establish its actual live view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTaskObservationRoots {
    /// Logical root whose canonical insertion material was inspected.
    pub root_digest: Identity256,
    pub root_extents: [u32; 3],
    /// Original admitted query selections, retaining their order.
    pub query_records: [u32; 3],
    /// Query ordinal, genuine original insertion target (absent for a derived
    /// target), and original support record. Ordered by query, then insertion.
    /// A matching query value never supplies an absent original target.
    pub contributors: Vec<(u32, Option<u32>, u32)>,
}

impl SemanticTaskEvaluationSpec {
    fn validate_records(
        &self,
        records: &SemanticAdmissionRecords,
    ) -> Result<(), SemanticTransitionError> {
        let invalid = |detail: &str| SemanticTransitionError::InvalidInput {
            detail: format!("task binding: {detail}"),
        };
        self.scoring.validate()?;
        if self
            .admissible_truth_masks
            .iter()
            .any(|mask| *mask == 0 || *mask & !15 != 0)
        {
            return Err(invalid(
                "truth eligibility requires a nonempty four-valued mask",
            ));
        }
        for index in self.statement_records {
            let record = records
                .records
                .get(index as usize)
                .ok_or_else(|| invalid("query statement is outside the admission"))?;
            let predicate = records
                .predicates
                .iter()
                .find(|predicate| predicate.predicate == record.predicate)
                .ok_or_else(|| invalid("query predicate is outside the admission"))?;
            if predicate.role != SemanticRecordRole::Statement {
                return Err(invalid("query selection is not a statement"));
            }
        }
        if self.allowed_support_records.len() > records.supports.len() {
            return Err(invalid(
                "support selections exceed the admitted support bank",
            ));
        }
        for &index in &self.allowed_support_records {
            let support = records
                .supports
                .get(index as usize)
                .ok_or_else(|| invalid("support selection is outside the admission"))?;
            if !self.statement_records.contains(&support.statement) {
                return Err(invalid(
                    "support belongs to a statement outside the task scope",
                ));
            }
        }
        Ok(())
    }
}

fn feedback_statement_payload(
    records: &SemanticAdmissionRecords,
    encoded: &[crate::semantic_hypergraph::SemanticRecordEncoding],
    record: u32,
) -> Result<Vec<u8>, SemanticTransitionError> {
    let statement = records
        .records
        .get(record as usize)
        .ok_or(SemanticTransitionError::InvalidTaskBank)?;
    let mut payload = Vec::new();
    let append = |payload: &mut Vec<u8>, index: u32| -> Result<(), SemanticTransitionError> {
        let atom = encoded
            .get(index as usize)
            .ok_or(SemanticTransitionError::InvalidTaskBank)?;
        // Canonical admission retains exact argument ranges. The predicate and
        // arity occupy the eight bytes immediately before the first argument;
        // zero-argument atoms end immediately after those same two u32 values.
        // No symbol lookup, numeric conversion, domain or digest enters features.
        let argument_begin = atom
            .arguments
            .first()
            .map_or(atom.bytes.len(), |range| range.start);
        let value_begin = argument_begin
            .checked_sub(8)
            .ok_or(SemanticTransitionError::InvalidTaskBank)?;
        let values = atom
            .bytes
            .get(value_begin..)
            .ok_or(SemanticTransitionError::InvalidTaskBank)?;
        payload.extend_from_slice(&(values.len() as u64).to_le_bytes());
        payload.extend_from_slice(values);
        Ok(())
    };
    append(&mut payload, record)?;
    let qualifier_count = u32::try_from(statement.qualifiers.len())
        .map_err(|_| SemanticTransitionError::InvalidTaskBank)?;
    payload.extend_from_slice(&qualifier_count.to_le_bytes());
    for &qualifier in &statement.qualifiers {
        append(&mut payload, qualifier)?;
    }
    Ok(payload)
}

/// Immutable native task content, separate from the controller's authority and
/// from every operand visible to the proposal policy.
pub(crate) struct TaskEvaluationBinding {
    admission_identity: Identity256,
    schema_generation: Identity256,
    spec: SemanticTaskEvaluationSpec,
    statements: [crate::SemanticStatementKey; 3],
    statement_bytes: BTreeMap<u32, Vec<u8>>,
    allowed_supports: Vec<(u32, [u8; 32])>,
    observation: SemanticTaskObservation,
}

impl TaskEvaluationBinding {
    pub(crate) fn bind(
        admission: &SemanticAdmission,
        spec: SemanticTaskEvaluationSpec,
        observation: SemanticTaskObservation,
    ) -> Result<Self, SemanticTransitionError> {
        spec.validate_records(admission.records())?;
        let statements = [
            admission
                .statement_key(spec.statement_records[0])
                .map_err(SemanticTransitionError::Semantic)?,
            admission
                .statement_key(spec.statement_records[1])
                .map_err(SemanticTransitionError::Semantic)?,
            admission
                .statement_key(spec.statement_records[2])
                .map_err(SemanticTransitionError::Semantic)?,
        ];
        let statement_bytes = BTreeMap::from([
            (
                spec.statement_records[0],
                feedback_statement_payload(
                    admission.records(),
                    &admission.encoded_records,
                    spec.statement_records[0],
                )?,
            ),
            (
                spec.statement_records[1],
                feedback_statement_payload(
                    admission.records(),
                    &admission.encoded_records,
                    spec.statement_records[1],
                )?,
            ),
            (
                spec.statement_records[2],
                feedback_statement_payload(
                    admission.records(),
                    &admission.encoded_records,
                    spec.statement_records[2],
                )?,
            ),
        ]);
        let mut allowed_supports = BTreeSet::new();
        for &index in &spec.allowed_support_records {
            let record = &admission.records().supports[index as usize];
            let statement = admission
                .statement_key(record.statement)
                .map_err(SemanticTransitionError::Semantic)?;
            let event = admission
                .support_event(index)
                .map_err(SemanticTransitionError::Semantic)?;
            allowed_supports.insert((index, *event.identity(statement.identity()).as_bytes()));
        }
        if observation.program_source.is_empty() || observation.result_bytes.is_empty() {
            return Err(publication_input_error(
                "native task observer omitted source or result custody",
            ));
        }
        Ok(Self {
            admission_identity: admission.identity(),
            schema_generation: admission.schema_generation(),
            spec,
            statements,
            statement_bytes,
            allowed_supports: allowed_supports.into_iter().collect(),
            observation,
        })
    }

    pub(crate) fn identity(&self) -> Identity256 {
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.task-evaluation.v5\0");
        hash.update(self.admission_identity.as_bytes());
        hash.update(self.schema_generation.as_bytes());
        // Keep source occurrence and explicit selection order separate from
        // the distinct original support records consumed by the device policy.
        hash.update((self.spec.statement_records.len() as u64).to_le_bytes());
        for index in self.spec.statement_records {
            hash.update(index.to_le_bytes());
        }
        hash.update((self.spec.allowed_support_records.len() as u64).to_le_bytes());
        for index in &self.spec.allowed_support_records {
            hash.update(index.to_le_bytes());
        }
        hash.update((self.observation.program_source.len() as u64).to_le_bytes());
        hash.update(&self.observation.program_source);
        hash.update((self.observation.input_bytes.len() as u64).to_le_bytes());
        hash.update(&self.observation.input_bytes);
        hash.update((self.observation.result_bytes.len() as u64).to_le_bytes());
        hash.update(&self.observation.result_bytes);
        for truth in self.observation.expected_truth {
            hash.update((truth as u64).to_le_bytes());
        }
        hash.update(self.spec.admissible_truth_masks);
        for weight in self.spec.scoring.words() {
            hash.update(weight.to_le_bytes());
        }
        for statement in self.statements {
            hash.update(statement.identity().as_bytes());
        }
        hash.update((self.allowed_supports.len() as u64).to_le_bytes());
        for (record, support) in &self.allowed_supports {
            hash.update(record.to_le_bytes());
            hash.update(support);
        }
        Identity256::from_bytes(hash.finalize().into())
    }

    pub(crate) fn spec(&self) -> &SemanticTaskEvaluationSpec {
        &self.spec
    }

    pub(crate) fn words(&self, owner: u64) -> Vec<u64> {
        let mut words = vec![4, owner];
        words.extend(identity_words(*self.identity().as_bytes()));
        for statement in self.statements {
            words.extend(identity_words(*statement.identity().as_bytes()));
        }
        words.extend(self.observation.expected_truth.map(|truth| truth as u64));
        words.push(self.allowed_supports.len() as u64);
        words.extend(self.spec.statement_records.map(u64::from));
        words.extend(self.spec.scoring.words());
        words.extend(self.spec.admissible_truth_masks.map(u64::from));
        for (record, support) in &self.allowed_supports {
            words.push(u64::from(*record));
            words.extend(identity_words(*support));
        }
        words
    }
}

/// One row of the generated roster. Text fields are text positions; edit fields
/// index the eighteen-field action signature. Offsets count input elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticComponent {
    pub offset: u64,
    pub ordinal: u32,
    pub lane: u32,
    pub slot: u32,
    pub field: u32,
    pub kind: u32,
    pub cardinality: u32,
}

impl SemanticComponent {
    pub const fn is_text(&self) -> bool {
        self.kind == COMPONENT_KIND_TEXT as u32
    }
}

/// Field codebook role and NULL-only cold cardinality in this catalogue.
pub struct SemanticCatalogueField {
    pub ordinal: u32,
    pub cardinality: u32,
    pub name: &'static str,
    pub role: &'static str,
}

/// Declarative action/opcode signature, including dependent opcode selection.
pub struct SemanticCatalogueSignature {
    pub value: u32,
    pub kind: &'static str,
    pub name: &'static str,
    pub signature: &'static str,
}

include!(concat!(
    env!("OUT_DIR"),
    "/semantic_transition_catalogue.rs"
));

pub const SEMANTIC_TRANSITION_GENERATION: u64 = CATALOGUE_GENERATION;
pub const SEMANTIC_TRANSITION_COMPONENT_COUNT: usize = COMPONENT_COUNT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticCatalogueBinding {
    pub generation: u64,
    pub digest: Identity256,
}

/// The sole built-in action law; both Rust and CUDA are generated from its declaration.
pub struct SemanticActionCatalogue;

impl SemanticActionCatalogue {
    pub const fn current() -> Self {
        Self
    }
    pub const fn binding(&self) -> SemanticCatalogueBinding {
        SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest: Identity256::from_bytes(CATALOGUE_DIGEST),
        }
    }
    pub const fn canonical_bytes(&self) -> &'static [u8] {
        CATALOGUE_ENCODING
    }
    pub const fn components(&self) -> &'static [SemanticComponent] {
        COMPONENTS
    }
    pub const fn fields(&self) -> &'static [SemanticCatalogueField] {
        CATALOGUE_FIELDS
    }
    pub const fn signatures(&self) -> &'static [SemanticCatalogueSignature] {
        CATALOGUE_SIGNATURES
    }
    pub const fn text_cardinality(&self) -> usize {
        TEXT_CARDINALITY
    }
    pub const fn scratch_bytes(&self) -> usize {
        SCRATCH_BYTES
    }
    /// (active fields, forced-NULL fields) for NO_EDIT and INSERT_SUPPORT.
    pub const fn field_masks(&self) -> [(u32, u32); 2] {
        [
            (NO_EDIT_ACTIVE_MASK as u32, NO_EDIT_NULL_MASK as u32),
            (
                INSERT_SUPPORT_ACTIVE_MASK as u32,
                INSERT_SUPPORT_NULL_MASK as u32,
            ),
        ]
    }
}

/// Cold-bound FP32 logits and binary product support in original category order.
pub struct SemanticComponentInput {
    pub logits: Vec<f32>,
    pub product_support: Vec<u8>,
}

/// FP32 element ranges for one field in the native policy parameter snapshot.
/// NULL is a category, but has no stored embedding row. The action-class field
/// has no NULL: its category zero is the trainable no-edit action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticPolicyFieldLayout {
    pub embeddings: std::ops::Range<usize>,
    pub biases: std::ops::Range<usize>,
    pub cardinality: usize,
    pub null_category: Option<usize>,
}

/// Packing of the immutable policy inputs consumed by the serial component loop.
/// These ranges describe storage, not authority to substitute a model generation.
/// A session derives the layout from its retained admission's actual codebooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticPolicyLayout {
    pub z: std::ops::Range<usize>,
    pub recurrence: std::ops::Range<usize>,
    pub positions: std::ops::Range<usize>,
    pub fields: Vec<SemanticPolicyFieldLayout>,
    pub parameter_cells: usize,
}

impl SemanticPolicyLayout {
    fn from_components(components: &[SemanticComponent]) -> Result<Self, SemanticTransitionError> {
        let invalid = || SemanticTransitionError::InvalidInput {
            detail: "policy layout requires the complete native component roster".into(),
        };
        if components.len() != COMPONENT_COUNT {
            return Err(invalid());
        }
        let z = 0usize..4 * 128;
        let recurrence = z.end..z.end + 128 * 128;
        let positions = recurrence.end..recurrence.end + 18 * 128;
        let mut cursor = positions.end;
        let mut fields = Vec::with_capacity(18);
        for field in 0..18 {
            let cardinality = components[32 + field].cardinality as usize;
            if cardinality == 0 {
                return Err(invalid());
            }
            for lane in 0..2 {
                for slot in 0..2 {
                    let component = components[lane * 68 + 32 + slot * 18 + field];
                    if component.kind != COMPONENT_KIND_EDIT as u32
                        || component.field as usize != field
                        || component.lane as usize != lane + 1
                        || component.slot as usize != slot
                        || component.cardinality as usize != cardinality
                    {
                        return Err(invalid());
                    }
                }
            }
            let null_category = (field != 0).then_some(0);
            let embedding_cells = (cardinality - usize::from(null_category.is_some()))
                .checked_mul(128)
                .ok_or_else(invalid)?;
            let end = cursor.checked_add(embedding_cells).ok_or_else(invalid)?;
            let embeddings = cursor..end;
            cursor = end;
            let bias_cells = if null_category.is_some() && cardinality == 1 {
                0
            } else {
                cardinality
            };
            let end = cursor.checked_add(bias_cells).ok_or_else(invalid)?;
            let biases = cursor..end;
            cursor = end;
            fields.push(SemanticPolicyFieldLayout {
                embeddings,
                biases,
                cardinality,
                null_category,
            });
        }
        Ok(Self {
            z,
            recurrence,
            positions,
            fields,
            parameter_cells: cursor,
        })
    }

    /// Both lanes retain the initial state and all thirty-six field successors.
    pub const fn recurrent_cells(&self) -> usize {
        2 * 37 * 128
    }

    /// The same score bank is reused serially by every structural component.
    pub fn score_cells(&self) -> usize {
        self.fields
            .iter()
            .map(|field| field.cardinality)
            .max()
            .unwrap_or(0)
    }
}

/// Immutable descriptor meanings derived from the graph's admitted records.
/// Category zero is NULL. Nonzero categories are sorted by logical typed bytes,
/// independent of arena slots and digests. Equal-valued support records retain
/// separate categories, ordered by their original admission index.
#[derive(Clone, Debug)]
pub enum SemanticActionDescriptor {
    Target {
        predicate: xlog_core::RelId,
    },
    Operand {
        scalar_type: xlog_core::ScalarType,
        sort_label: String,
        encoded_value: Vec<u8>,
    },
    QualifierBundle {
        records: Vec<u32>,
    },
    SupportEvent {
        source_record: u32,
    },
}

struct ActionCodebooks {
    words: Vec<u64>,
    targets: Vec<SemanticActionDescriptor>,
    operands: Vec<SemanticActionDescriptor>,
    qualifiers: Vec<SemanticActionDescriptor>,
    leaves: Vec<SemanticActionDescriptor>,
    components: Vec<SemanticComponent>,
    binding: SemanticCatalogueBinding,
    input_cells: usize,
}

impl ActionCodebooks {
    fn support_layout(&self) -> (Vec<std::ops::Range<usize>>, usize) {
        (
            self.components
                .iter()
                .map(|component| {
                    component.offset as usize
                        ..component.offset as usize + component.cardinality as usize
                })
                .collect(),
            self.input_cells,
        )
    }

    fn derive(admission: &SemanticAdmission, owner: u64) -> Result<Self, SemanticTransitionError> {
        let records = admission.records();
        let predicates: BTreeMap<_, _> = records
            .predicates
            .iter()
            .map(|p| (p.predicate.0, p))
            .collect();
        let sorts: BTreeSet<_> = records
            .predicates
            .iter()
            .flat_map(|p| p.schema.sort_labels().iter().cloned())
            .collect();
        let sorts: BTreeMap<_, _> = sorts
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u64 + 1))
            .collect();
        let mut targets = BTreeMap::new();
        for p in predicates
            .values()
            .filter(|p| p.role == SemanticRecordRole::Statement && p.schema.arity() <= 4)
        {
            let mut row = vec![p.predicate.0 as u64, p.schema.arity() as u64];
            row.extend((0..4).map(|i| {
                p.schema
                    .columns
                    .get(i)
                    .map_or(0, |(_, t)| t.to_code() as u64 + 1)
            }));
            row.extend((0..4).map(|i| p.schema.sort_labels().get(i).map_or(0, |s| sorts[s])));
            targets.insert(
                p.predicate.0.to_le_bytes(),
                (
                    row,
                    SemanticActionDescriptor::Target {
                        predicate: p.predicate,
                    },
                ),
            );
        }
        let mut operands = BTreeMap::new();
        for (record_index, (record, encoded)) in records
            .records
            .iter()
            .zip(&admission.encoded_records)
            .enumerate()
        {
            let schema = &predicates[&record.predicate.0].schema;
            for (i, range) in encoded.arguments.iter().enumerate() {
                let value = encoded.bytes[range.clone()].to_vec();
                let label = &schema.sort_labels()[i];
                let mut key = value.clone();
                key.extend_from_slice(&(label.len() as u64).to_le_bytes());
                key.extend_from_slice(label.as_bytes());
                operands.entry(key).or_insert_with(|| {
                    (
                        value.clone(),
                        schema.columns[i].1.to_code() as u64 + 1,
                        sorts[label],
                        record_index as u64,
                        i as u64,
                        SemanticActionDescriptor::Operand {
                            scalar_type: schema.columns[i].1,
                            sort_label: label.clone(),
                            encoded_value: value,
                        },
                    )
                });
            }
        }
        let logical = |indices: &[u32]| {
            let mut bytes = (indices.len() as u32).to_le_bytes().to_vec();
            for &index in indices {
                let atom = &admission.encoded_records[index as usize].bytes;
                bytes.extend_from_slice(&(atom.len() as u64).to_le_bytes());
                bytes.extend_from_slice(atom);
            }
            bytes
        };
        let mut qualifiers = BTreeMap::new();
        qualifiers.insert(logical(&[]), (Vec::new(), u32::MAX));
        for (index, record) in records.records.iter().enumerate() {
            if predicates[&record.predicate.0].role == SemanticRecordRole::Statement {
                qualifiers
                    .entry(logical(&record.qualifiers))
                    .or_insert_with(|| (record.qualifiers.clone(), index as u32));
            }
        }
        let mut leaves = BTreeMap::new();
        for (index, support) in records.supports.iter().enumerate() {
            let polarity = match support.polarity {
                crate::SemanticPolarity::Pro => 1u64,
                crate::SemanticPolarity::Contra => 2,
            };
            let refs = [
                support.provenance,
                support.source,
                support.context,
                support.scope,
            ];
            let mut key = polarity.to_le_bytes().to_vec();
            key.extend(logical(&refs));
            let mut words = vec![polarity];
            for record in refs {
                words.extend(identity_words(
                    Sha256::digest(&admission.encoded_records[record as usize].bytes).into(),
                ));
            }
            words.push(index as u64);
            leaves.insert(
                (key, index),
                (
                    words,
                    SemanticActionDescriptor::SupportEvent {
                        source_record: index as u32,
                    },
                ),
            );
        }
        if [
            targets.len(),
            operands.len(),
            qualifiers.len(),
            leaves.len(),
        ]
        .into_iter()
        .any(|n| n >= 65536)
        {
            return Err(SemanticTransitionError::InvalidInput {
                detail: "descriptor codebook exceeds 65535 non-NULL entries".into(),
            });
        }
        let mut words = vec![0; 25];
        let mut bytes = Vec::new();
        words[0] = words.len() as u64;
        words[1] = targets.len() as u64 + 1;
        words.extend([0; 10]);
        let target_views = targets
            .into_values()
            .map(|(row, view)| {
                words.extend(row);
                view
            })
            .collect();
        words[2] = words.len() as u64;
        words[3] = operands.len() as u64 + 1;
        words.extend([0; 6]);
        let operand_views = operands
            .into_values()
            .map(|(value, ty, sort, record, argument, view)| {
                words.extend([
                    bytes.len() as u64,
                    value.len() as u64,
                    ty,
                    sort,
                    record,
                    argument,
                ]);
                bytes.extend(value);
                view
            })
            .collect();
        words[4] = words.len() as u64;
        words[5] = qualifiers.len() as u64 + 1;
        words.extend([0; 3]);
        let qualifier_views = qualifiers
            .into_values()
            .map(|(refs, source)| {
                let offset = bytes.len();
                bytes.extend_from_slice(&(refs.len() as u32).to_le_bytes());
                for &index in &refs {
                    bytes.extend_from_slice(&Sha256::digest(
                        &admission.encoded_records[index as usize].bytes,
                    ));
                }
                words.extend([
                    offset as u64,
                    (bytes.len() - offset) as u64,
                    u64::from(source),
                ]);
                SemanticActionDescriptor::QualifierBundle { records: refs }
            })
            .collect();
        words[6] = words.len() as u64;
        words[7] = leaves.len() as u64 + 1;
        words.extend([0; 18]);
        let leaf_views = leaves
            .into_values()
            .map(|(row, view)| {
                words.extend(row);
                view
            })
            .collect();
        words[8] = words.len() as u64 * 8;
        words[9] = bytes.len() as u64;
        words[10..14].copy_from_slice(&identity_words(*admission.schema_generation().as_bytes()));
        words[14] = owner;
        words[15] = admission.base().slot() as u64;
        words[16] = admission.base().generation();
        words[17..21].copy_from_slice(&identity_words(
            *admission.base_snapshot().digest().as_bytes(),
        ));
        bytes.resize(bytes.len().div_ceil(8) * 8, 0);
        words.extend(
            bytes
                .chunks_exact(8)
                .map(|v| u64::from_le_bytes(v.try_into().unwrap())),
        );
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.action-binding.v3\0");
        hash.update(CATALOGUE_DIGEST);
        hash.update(admission.identity().as_bytes());
        for &word in &words {
            hash.update(word.to_le_bytes());
        }
        let digest = Identity256::from_bytes(hash.finalize().into());
        words[21..25].copy_from_slice(&identity_words(*digest.as_bytes()));
        let mut offset = 0u64;
        let components = COMPONENTS
            .iter()
            .map(|&c| {
                let cardinality = if c.is_text() {
                    c.cardinality
                } else {
                    match c.field {
                        2 => words[1] as u32,
                        3..=6 => words[3] as u32,
                        11 => words[5] as u32,
                        17 => words[7] as u32,
                        _ => c.cardinality,
                    }
                };
                let result = SemanticComponent {
                    offset,
                    cardinality,
                    ..c
                };
                offset += cardinality as u64;
                result
            })
            .collect();
        Ok(Self {
            words,
            targets: target_views,
            operands: operand_views,
            qualifiers: qualifier_views,
            leaves: leaf_views,
            components,
            binding: SemanticCatalogueBinding {
                generation: CATALOGUE_GENERATION,
                digest,
            },
            input_cells: offset as usize,
        })
    }
}

fn identity_words(bytes: [u8; 32]) -> [u64; 4] {
    std::array::from_fn(|i| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap()))
}

// SAFETY: fixed C layout, integral fields only.
unsafe impl DeviceRepr for SemanticComponent {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticRngBinding {
    pub model_generation: u32,
    pub stream_serial: u64,
    pub family_id: u8,
    pub proposal: u32,
}

/// Device-produced receipt. Integers use three little-endian 64-bit limbs.
/// The exact probability is p/q; the exact importance factor is
/// p/factor_denominator. Singleton rows canonically use 1/1 for both.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTransitionReceipt {
    pub proposal: u64,
    pub catalogue_generation: u64,
    pub catalogue_digest: Identity256,
    /// Binds the retained admission, acquired base, owner and logical codebooks.
    pub admission_binding: Identity256,
    pub ordinal: u32,
    pub lane: u32,
    pub slot: u32,
    pub field: u32,
    pub kind: u32,
    pub choice: u32,
    pub legal_count: u32,
    pub active_count: u32,
    pub key: [u32; 2],
    pub counter: [u32; 4],
    pub random_words: [u32; 4],
    pub draw: u64,
    pub cdf_start: u64,
    pub cdf_end: u64,
    pub mass: u64,
    pub p: [u64; 3],
    pub q: [u64; 3],
    pub factor_denominator: [u64; 3],
    pub active_fields: u32,
    pub null_fields: u32,
}

// SAFETY: C layout, integral fields only; all bit patterns are valid.
unsafe impl DeviceRepr for SemanticTransitionReceipt {}

fn canonical_text_null(receipt: &SemanticTransitionReceipt) -> bool {
    receipt.kind == COMPONENT_KIND_TEXT as u32
        && receipt.choice == TEXT_NULL as u32
        && receipt.legal_count == 1
        && receipt.active_count == 1
        && receipt.active_fields == 0
        && receipt.null_fields == 0
        && receipt.p == [1, 0, 0]
        && receipt.q == [1, 0, 0]
        && receipt.factor_denominator == [1, 0, 0]
        && receipt.cdf_start == 0
        && receipt.cdf_end == 1 << 63
        && receipt.mass == 1 << 63
}

/// Device-computed facts, using the three actual canonical truth-query results.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTaskFacts {
    /// Equality of each actual Truth4 answer to the protected observer result.
    pub correct: [u64; 3],
    /// Number of correct subgoals, in the inclusive range zero to three.
    pub g: u64,
    /// Parent completion: all three subgoals are correct (zero or one).
    pub p: u64,
    /// Actual edit commands plus added supports plus defined truth changes.
    pub c: u64,
    /// Query agreement minus measured work, using the bound task coefficients.
    pub v: i64,
    /// Whether the candidate satisfies the task scope and hard constraints.
    pub eligible: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceTaskEvaluation {
    lane_refusal: [u64; 2],
    query_receipts: [[[u64; 42]; 3]; 3],
    query_count: u64,
    winner: u64,
    return_value: i64,
    facts: [SemanticTaskFacts; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceState {
    model_generation: u32,
    family_id: u32,
    stream_serial: u64,
    proposal: u64,
    next_proposal: u64,
    blocks: u64,
    status: u64,
    catalogue_generation: u64,
    catalogue_digest: Identity256,
    binding_digest: Identity256,
    semantic_receipts: [[u64; 42]; 7],
    work: [SemanticTransitionWork; 2],
    importance_weight: f64,
    retired_roots_mask: u64,
    cleanup_receipts: [[u64; 42]; 3],
    task_evaluation: DeviceTaskEvaluation,
    execution_work: ExecutionWork,
}

impl Default for DeviceState {
    fn default() -> Self {
        // SAFETY: the scalar ABI admits every bit pattern; zero initializes all
        // integers and the floating-point importance weight.
        unsafe { std::mem::zeroed() }
    }
}

// SAFETY: C layout, integer and FP64 fields; all bit patterns are valid.
unsafe impl DeviceRepr for DeviceState {}

/// Original active-ring source slot. Kind and provenance are independent: a
/// filled model-input row is not an assertion that the token was observed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTextSlot {
    pub token: u64,
    pub logical_position: u64,
    /// Padding = 0, filled = 1, mask = 2. Active validity is derived from kind.
    pub kind: u64,
    /// Absent = 0, source = 1, generated = 2; only filled rows have a subtype.
    pub provenance: u64,
    pub valid: u64,
    pub committed: u64,
    pub recomputed: u64,
    /// Row in the retained full source-provenance table, not an authority bit.
    pub provenance_record: u64,
}

type SourceSlot = SemanticTextSlot;

// SAFETY: C layout containing only u64 values; all bit patterns are valid.
unsafe impl DeviceRepr for SourceSlot {}

/// Semantic role of a retained state range. Indices name actual layers, tensors,
/// or records inside the cold-bound role roster, never arbitrary device pointers.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticStateRole {
    PrefixSource = 1,
    TokenProvenanceRecords = 2,
    PrefixIdentity = 3,
    AttentionKeys = 4,
    AttentionValues = 5,
    ConvolutionState = 6,
    RecurrentState = 7,
    ValueRunningMax = 8,
    ValueDenominator = 9,
    ValueNumerator = 10,
    EditRunningMax = 11,
    EditDenominator = 12,
    EditNumerator = 13,
    CompletionCoverage = 14,
    RawFeedback = 15,
    FeedbackStatements = 16,
    FeedbackSupportProvenance = 17,
    SlowWeights = 18,
    FastAdapters = 19,
    SymbolicHeads = 20,
    OptimizerState = 21,
    LossScaler = 22,
    TrainingSchedule = 23,
    Gradients = 24,
    GradientAccumulation = 25,
    OtherRng = 26,
    ReplayEntries = 27,
    ReplayPayload = 28,
    TrainingView = 29,
    IntentEntries = 30,
    IntentPayload = 31,
    Acknowledgements = 32,
    AttemptReceipt = 33,
    GoalState = 34,
    WorldLineage = 35,
    EditJournal = 36,
    Environment = 37,
    AuthorityEnvelope = 38,
    AuthorityDecisions = 39,
    CheckpointLineage = 40,
    SchemaContract = 41,
    TokenizerContract = 42,
    RuntimeContract = 43,
    ModelContract = 44,
    TopologyContract = 45,
    PositionTable = 46,
    ActionCatalogue = 47,
    ActionCodebooks = 48,
    ProofRecords = 49,
    LearningCopyReset = 50,
    ActiveAttentionKeys = 51,
    ActiveAttentionValues = 52,
    ActiveConvolution = 53,
    ActiveRecurrent = 54,
    TensorLayout = 55,
}

impl SemanticStateRole {
    pub fn from_code(code: u64) -> Option<Self> {
        const ROLES: [SemanticStateRole; 55] = [
            SemanticStateRole::PrefixSource,
            SemanticStateRole::TokenProvenanceRecords,
            SemanticStateRole::PrefixIdentity,
            SemanticStateRole::AttentionKeys,
            SemanticStateRole::AttentionValues,
            SemanticStateRole::ConvolutionState,
            SemanticStateRole::RecurrentState,
            SemanticStateRole::ValueRunningMax,
            SemanticStateRole::ValueDenominator,
            SemanticStateRole::ValueNumerator,
            SemanticStateRole::EditRunningMax,
            SemanticStateRole::EditDenominator,
            SemanticStateRole::EditNumerator,
            SemanticStateRole::CompletionCoverage,
            SemanticStateRole::RawFeedback,
            SemanticStateRole::FeedbackStatements,
            SemanticStateRole::FeedbackSupportProvenance,
            SemanticStateRole::SlowWeights,
            SemanticStateRole::FastAdapters,
            SemanticStateRole::SymbolicHeads,
            SemanticStateRole::OptimizerState,
            SemanticStateRole::LossScaler,
            SemanticStateRole::TrainingSchedule,
            SemanticStateRole::Gradients,
            SemanticStateRole::GradientAccumulation,
            SemanticStateRole::OtherRng,
            SemanticStateRole::ReplayEntries,
            SemanticStateRole::ReplayPayload,
            SemanticStateRole::TrainingView,
            SemanticStateRole::IntentEntries,
            SemanticStateRole::IntentPayload,
            SemanticStateRole::Acknowledgements,
            SemanticStateRole::AttemptReceipt,
            SemanticStateRole::GoalState,
            SemanticStateRole::WorldLineage,
            SemanticStateRole::EditJournal,
            SemanticStateRole::Environment,
            SemanticStateRole::AuthorityEnvelope,
            SemanticStateRole::AuthorityDecisions,
            SemanticStateRole::CheckpointLineage,
            SemanticStateRole::SchemaContract,
            SemanticStateRole::TokenizerContract,
            SemanticStateRole::RuntimeContract,
            SemanticStateRole::ModelContract,
            SemanticStateRole::TopologyContract,
            SemanticStateRole::PositionTable,
            SemanticStateRole::ActionCatalogue,
            SemanticStateRole::ActionCodebooks,
            SemanticStateRole::ProofRecords,
            SemanticStateRole::LearningCopyReset,
            SemanticStateRole::ActiveAttentionKeys,
            SemanticStateRole::ActiveAttentionValues,
            SemanticStateRole::ActiveConvolution,
            SemanticStateRole::ActiveRecurrent,
            SemanticStateRole::TensorLayout,
        ];
        let index = usize::try_from(code.checked_sub(1)?).ok()?;
        ROLES.get(index).copied()
    }
}

/// One actual canonical control record or queue payload retained by a parent.
/// Mutable queue capacity is explicit; bytes contain the real encoded header
/// and entries, including an empty queue's header rather than an invented state.
pub struct SemanticStateRecord {
    pub role: SemanticStateRole,
    pub index: u64,
    pub bytes: Vec<u8>,
    pub capacity_bytes: usize,
}

/// Byte spans in the producer's one ModelContract record. The native owner
/// hashes the immutable schema bytes without interpreting their format, then
/// binds them to the actual model generation and device numerical identity.
/// Digest fields are raw 32 bytes; generation is an unaligned little-endian u64.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticModelContractLayout {
    pub schema_begin: u64,
    pub schema_bytes: u64,
    pub schema_digest_offset: u64,
    pub generation_offset: u64,
    pub numerical_digest_offset: u64,
    pub identity_offset: u64,
}

impl SemanticModelContractLayout {
    fn validate(self, record_bytes: u64) -> Result<(), SemanticTransitionError> {
        let spans = [
            (self.schema_begin, self.schema_bytes),
            (self.schema_digest_offset, 32),
            (self.generation_offset, 8),
            (self.numerical_digest_offset, 32),
            (self.identity_offset, 32),
        ];
        for (index, &(begin, bytes)) in spans.iter().enumerate() {
            if bytes == 0
                || begin > record_bytes
                || bytes > record_bytes - begin
                || spans[..index]
                    .iter()
                    .any(|&(other, length)| begin < other + length && other < begin + bytes)
            {
                return Err(publication_input_error(
                    "model contract spans must be nonempty, disjoint and within its actual bytes",
                ));
            }
        }
        Ok(())
    }
}

/// A genuine producer-owned model tensor and its exact cold model contract.
/// No numeric conversion occurs. Logical intervals describe covered positions;
/// non-positional weights and optimizer state use the explicit interval 0..0.
pub struct SemanticTensorInput {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub tensor: DlpackManagedTensor,
    /// Retained native allocation origin of this actual producer, if any.
    /// This is not a reader grant or a tensor version-counter identity.
    pub native_allocation: Option<DeviceAllocationProvenance>,
}

/// A producer-declared storage object within one complete backing allocation.
/// Storage identity is its ordinal, not an address or a tensor version counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticModelStorage {
    pub allocation: u64,
    pub byte_offset: u64,
    pub span_bytes: u64,
}

/// A typed publication coordinate within a producer-declared storage object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticModelView {
    pub role: u64,
    pub index: u64,
    pub storage: u64,
    pub byte_offset: u64,
}

/// Full producer-owned backing buffers and their independent storage/view map.
/// Allocation inputs are contiguous U8 vectors with role zero and ordinal index.
/// Tensor and version-counter identities remain in the opaque model contract.
pub struct SemanticModelMemory {
    pub allocations: Vec<SemanticTensorInput>,
    pub storages: Vec<SemanticModelStorage>,
    pub views: Vec<SemanticModelView>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ModelMemoryGeometry {
    allocation_bytes: Vec<u64>,
    storages: Vec<SemanticModelStorage>,
    views: Vec<SemanticModelView>,
}

impl ModelMemoryGeometry {
    fn location(&self, role: u64, index: u64) -> Result<(usize, usize), SemanticTransitionError> {
        let view = self
            .views
            .iter()
            .find(|view| (view.role, view.index) == (role, index))
            .ok_or_else(|| {
                publication_input_error("model tensor lacks its producer storage view")
            })?;
        let storage = usize::try_from(view.storage)
            .ok()
            .and_then(|index| self.storages.get(index))
            .ok_or_else(|| publication_input_error("model view names an absent storage object"))?;
        let allocation = usize::try_from(storage.allocation)
            .ok()
            .filter(|&index| index < self.allocation_bytes.len())
            .ok_or_else(|| {
                publication_input_error("model storage names an absent backing allocation")
            })?;
        let offset = storage
            .byte_offset
            .checked_add(view.byte_offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| publication_input_error("model storage view offset overflows"))?;
        Ok((allocation, offset))
    }

    fn validate(
        &self,
        layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    ) -> Result<(), SemanticTransitionError> {
        let mut allocations = BTreeSet::new();
        for storage in &self.storages {
            let capacity = usize::try_from(storage.allocation)
                .ok()
                .and_then(|index| self.allocation_bytes.get(index))
                .ok_or_else(|| publication_input_error("model storage allocation is absent"))?;
            if usize::try_from(*capacity).is_err()
                || storage
                    .byte_offset
                    .checked_add(storage.span_bytes)
                    .is_none_or(|end| end > *capacity)
            {
                return Err(publication_input_error(
                    "model storage exceeds its complete backing allocation",
                ));
            }
            allocations.insert(storage.allocation);
        }
        let mut storages = BTreeSet::new();
        let mut previous = None;
        for view in &self.views {
            let key = (view.role, view.index);
            if !matches!(view.role, 18..=25) || previous.is_some_and(|previous| previous >= key) {
                return Err(publication_input_error(
                    "model views must enumerate unique model tensor coordinates in order",
                ));
            }
            previous = Some(key);
            let layout = layouts.get(&key).ok_or_else(|| {
                publication_input_error("model storage view lacks its original tensor layout")
            })?;
            let (_, offset) = self.location(view.role, view.index)?;
            let storage = &self.storages[view.storage as usize];
            let span = tensor_layout_bytes(layout)? as u64;
            if layout.logical_axis != u64::MAX
                || !(offset as u64).is_multiple_of(layout.element_bytes)
                || view
                    .byte_offset
                    .checked_add(span)
                    .is_none_or(|end| end > storage.span_bytes)
            {
                return Err(publication_input_error(
                    "typed model view exceeds or misaligns its declared storage",
                ));
            }
            storages.insert(view.storage);
        }
        if allocations.len() != self.allocation_bytes.len()
            || storages.len() != self.storages.len()
            || self.views.len()
                != layouts
                    .values()
                    .filter(|layout| matches!(layout.role, 18..=25))
                    .count()
        {
            return Err(publication_input_error(
                "model memory map omits a tensor or contains unreachable storage",
            ));
        }
        Ok(())
    }

    fn encode_into(&self, bytes: &mut Vec<u8>) -> Result<(), SemanticTransitionError> {
        for count in [
            self.allocation_bytes.len(),
            self.storages.len(),
            self.views.len(),
        ] {
            material_u32(
                bytes,
                u32::try_from(count).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            );
        }
        for &extent in &self.allocation_bytes {
            material_u64(bytes, extent);
        }
        for storage in &self.storages {
            for field in [storage.allocation, storage.byte_offset, storage.span_bytes] {
                material_u64(bytes, field);
            }
        }
        for view in &self.views {
            for field in [view.role, view.index, view.storage, view.byte_offset] {
                material_u64(bytes, field);
            }
        }
        Ok(())
    }

    fn digest(
        &self,
        layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    ) -> Result<Identity256, SemanticTransitionError> {
        self.validate(layouts)?;
        let mut bytes = b"xlog.semantic.model-memory.v1\0".to_vec();
        self.encode_into(&mut bytes)?;
        let model = layouts
            .values()
            .filter(|layout| matches!(layout.role, 18..=25));
        material_u64(&mut bytes, model.clone().count() as u64);
        for layout in model {
            for field in [
                layout.role,
                layout.index,
                layout.element_bytes,
                layout.scalar_type,
                layout.rank,
                layout.logical_axis,
            ]
            .into_iter()
            .chain(layout.dimensions)
            .chain(layout.strides_bytes)
            {
                material_u64(&mut bytes, field);
            }
        }
        Ok(Identity256::from_bytes(Sha256::digest(&bytes).into()))
    }

    fn decode_from(
        reader: &mut SemanticMaterialReader<'_>,
    ) -> Result<Self, SemanticTransitionError> {
        let allocations = reader.count(8).map_err(SemanticTransitionError::Semantic)?;
        let storages = reader
            .count(24)
            .map_err(SemanticTransitionError::Semantic)?;
        let views = reader
            .count(32)
            .map_err(SemanticTransitionError::Semantic)?;
        let mut geometry = Self::default();
        for _ in 0..allocations {
            geometry
                .allocation_bytes
                .push(reader.u64().map_err(SemanticTransitionError::Semantic)?);
        }
        for _ in 0..storages {
            geometry.storages.push(SemanticModelStorage {
                allocation: reader.u64().map_err(SemanticTransitionError::Semantic)?,
                byte_offset: reader.u64().map_err(SemanticTransitionError::Semantic)?,
                span_bytes: reader.u64().map_err(SemanticTransitionError::Semantic)?,
            });
        }
        for _ in 0..views {
            geometry.views.push(SemanticModelView {
                role: reader.u64().map_err(SemanticTransitionError::Semantic)?,
                index: reader.u64().map_err(SemanticTransitionError::Semantic)?,
                storage: reader.u64().map_err(SemanticTransitionError::Semantic)?,
                byte_offset: reader.u64().map_err(SemanticTransitionError::Semantic)?,
            });
        }
        Ok(geometry)
    }
}

fn prepare_model_memory(
    allocations: &[PreparedSemanticTensor],
    storages: Vec<SemanticModelStorage>,
    views: Vec<SemanticModelView>,
    tensors: &[PreparedSemanticTensor],
) -> Result<ModelMemoryGeometry, SemanticTransitionError> {
    let mut geometry = ModelMemoryGeometry {
        allocation_bytes: Vec::new(),
        storages,
        views,
    };
    for (index, allocation) in allocations.iter().enumerate() {
        let layout = allocation.layout;
        if layout.role != 0
            || layout.index != index as u64
            || layout.rank != 1
            || layout.scalar_type != 1
            || layout.element_bytes != 1
            || layout.logical_axis != u64::MAX
            || layout.strides_bytes != [1, 0, 0, 0]
            || allocation.logical_begin != 0
            || allocation.logical_end != 0
        {
            return Err(publication_input_error(
                "model backing producer must be its complete ordinal U8 vector",
            ));
        }
        let bytes = tensor_layout_bytes(&layout)? as u64;
        let end = allocation
            .data
            .checked_add(bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        // Validate distinct declared owners; never infer identity from addresses.
        for (prior, &prior_bytes) in allocations[..index].iter().zip(&geometry.allocation_bytes) {
            if bytes != 0
                && prior_bytes != 0
                && allocation.data < prior.data + prior_bytes
                && prior.data < end
            {
                return Err(publication_input_error(
                    "declared model backing allocations overlap",
                ));
            }
        }
        geometry.allocation_bytes.push(bytes);
    }
    let layouts = tensors
        .iter()
        .map(|tensor| ((tensor.layout.role, tensor.layout.index), tensor.layout))
        .collect();
    geometry.validate(&layouts)?;
    for tensor in tensors
        .iter()
        .filter(|tensor| matches!(tensor.layout.role, 18..=25))
    {
        let (allocation, offset) = geometry.location(tensor.layout.role, tensor.layout.index)?;
        if allocations[allocation].data.checked_add(offset as u64) != Some(tensor.data) {
            return Err(publication_input_error(
                "original tensor does not occupy its producer-declared backing view",
            ));
        }
    }
    Ok(geometry)
}

/// Initial canonical parent only. A model continuation cannot be supplied here:
/// it is admitted after a real Acquire and a forward using that acquired parent.
/// PrefixIdentity is native-owned: fresh publication derives its 32 bytes from
/// the actual PrefixSource. Supplying a caller record for that role is an error.
pub struct SemanticParentBinding {
    pub recovered_instance: Option<Identity256>,
    pub source: [SemanticTextSlot; 32],
    pub prefix: Vec<SemanticTextSlot>,
    pub ring_head: u64,
    pub provenance_records: u64,
    pub prefix_capacity: u64,
    pub feedback_capacity: u64,
    pub max_position: u64,
    pub pad_token: u64,
    pub terminal_tokens: Vec<u64>,
    pub final_intent_payload_bytes: u64,
    /// Actual application effect descriptor. The native owner constructs the
    /// empty intent queue; callers never encode its private header or entries.
    pub intent_effect: Vec<u8>,
    pub intent_entry_capacity: u64,
    pub intent_payload_capacity_bytes: usize,
    pub model_generation: u64,
    /// Complete versioned numerical-mode contribution from the model owner.
    /// The Python binding uses exact canonical ColdValue metadata, not a digest
    /// or repr. Native code owns the surrounding RuntimeContract record.
    pub model_numerical_mode: Vec<u8>,
    pub model_contract_layout: SemanticModelContractLayout,
    pub policy_generation: u64,
    pub neural_generation: u64,
    pub cache_generation: u64,
    pub authority_generation: u64,
    pub training_cursor: u64,
    pub training_rng: [u64; 4],
    pub fuel: u64,
    pub rng: SemanticRngBinding,
    pub topology_identity: Identity256,
    pub table_identity: Identity256,
    /// Every one of the 55 roles is explicit, including actual zero tensor counts.
    pub role_counts: [u64; 55],
    pub records: Vec<SemanticStateRecord>,
    pub tensors: Vec<SemanticTensorInput>,
    pub model_memory: SemanticModelMemory,
    /// Model-declared 51..54 capacities only; no active output exists before
    /// Acquire and the original forward. Initial ranges are unreachable.
    pub active_layouts: Vec<SemanticTensorLayout>,
}

/// An observed token payload admitted by the trusted application controller.
/// `origin` retains the complete source/manifest/transformation record, not an
/// independently supplied native digest or a model-generated provenance row.
#[derive(Clone, Debug)]
pub struct SemanticObservedSource {
    pub identity: String,
    pub tokens: Vec<u64>,
    pub origin: Vec<u8>,
}

/// A token is selected only by offset into its admitted observed payload.
#[derive(Clone, Debug)]
pub struct SemanticSourceMapping {
    pub source: String,
    pub offset: u64,
    pub logical_position: u64,
}

impl SemanticParentBinding {
    /// Construct the private initial SOURCE ledger from admitted observations.
    /// The complete authority record is retained in the parent by the same
    /// controller. Fresh GENERATED rows must instead enter through restoration.
    pub fn bind_initial_sources(
        &mut self,
        sources: &[SemanticObservedSource],
        mapping: &[SemanticSourceMapping],
        authority: &[u8],
        capacity_records: u64,
    ) -> Result<(), SemanticTransitionError> {
        if self.recovered_instance.is_some()
            || self.provenance_records != 0
            || self
                .records
                .iter()
                .any(|record| record.role == SemanticStateRole::TokenProvenanceRecords)
            || self.role_counts[SemanticStateRole::TokenProvenanceRecords as usize - 1] != 1
        {
            return Err(publication_input_error(
                "fresh source provenance must be constructed by the controller",
            ));
        }
        let (source, prefix, record) = initial_source_ledger(
            self.source,
            self.prefix.clone(),
            sources,
            mapping,
            authority,
            self.authority_generation,
            capacity_records,
        )?;
        self.source = source;
        self.prefix = prefix;
        self.provenance_records = mapping.len() as u64;
        self.records.push(record);
        Ok(())
    }
}

fn initial_source_ledger(
    mut ring: [SemanticTextSlot; 32],
    mut prefix: Vec<SemanticTextSlot>,
    sources: &[SemanticObservedSource],
    mapping: &[SemanticSourceMapping],
    authority: &[u8],
    generation: u64,
    capacity: u64,
) -> Result<
    (
        [SemanticTextSlot; 32],
        Vec<SemanticTextSlot>,
        SemanticStateRecord,
    ),
    SemanticTransitionError,
> {
    let invalid = publication_input_error;
    let source_count = sources.len();
    let sources = sources
        .iter()
        .map(|source| (source.identity.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    if sources.len() != source_count
        || sources.values().any(|source| {
            source.identity.is_empty()
                || source.origin.is_empty()
                || source.tokens.is_empty()
                || source
                    .tokens
                    .iter()
                    .any(|&token| token >= TEXT_CARDINALITY as u64 || token == MASK_TOKEN as u64)
        })
    {
        return Err(invalid(
            "observed sources require unique identities and complete valid token payloads",
        ));
    }
    if authority.is_empty()
        || generation == 0
        || capacity
            < (mapping.len() as u64)
                .checked_add(32)
                .ok_or_else(|| invalid("source ledger capacity overflows"))?
    {
        return Err(invalid(
            "source ledger requires complete authority and append capacity",
        ));
    }
    let mut positions = BTreeMap::new();
    for entry in mapping {
        let source = sources
            .get(entry.source.as_str())
            .ok_or_else(|| invalid("source mapping names an unadmitted payload"))?;
        let offset =
            usize::try_from(entry.offset).map_err(|_| invalid("source token offset overflows"))?;
        if offset >= source.tokens.len()
            || positions.insert(entry.logical_position, entry).is_some()
        {
            return Err(invalid(
                "source mapping has invalid payload, offset, or duplicate logical position",
            ));
        }
    }
    let authority_digest = Identity256::from_bytes(Sha256::digest(authority).into());
    let mut bytes = Vec::with_capacity(
        mapping
            .len()
            .checked_mul(size_of::<TokenProvenance>())
            .ok_or_else(|| invalid("source ledger extent overflows"))?,
    );
    let mut seen = BTreeSet::new();
    for (physical_slot, slot) in prefix.iter_mut().map(|slot| (u64::MAX, slot)).chain(
        ring.iter_mut()
            .enumerate()
            .map(|(index, slot)| (index as u64, slot)),
    ) {
        if slot.kind != 1 {
            if slot.kind > 2 || slot.provenance != 0 || slot.provenance_record != 0 {
                return Err(invalid(
                    "MASK or PAD cannot carry initial SOURCE provenance",
                ));
            }
            continue;
        }
        let entry = positions.get(&slot.logical_position).ok_or_else(|| {
            invalid("every initial FILLED position requires its exact source mapping")
        })?;
        let source = sources[entry.source.as_str()];
        let token = source.tokens[entry.offset as usize];
        if slot.provenance != 1
            || slot.provenance_record != 0
            || slot.token != token
            || !seen.insert(slot.logical_position)
        {
            return Err(invalid(
                "initial FILLED token differs from its admitted SOURCE payload or position",
            ));
        }
        slot.provenance_record = (bytes.len() / size_of::<TokenProvenance>()) as u64;
        let mut record = TokenProvenance {
            source_slot: physical_slot,
            logical_position: entry.logical_position,
            token,
            // For observed inputs ordinal is the exact payload token offset;
            // no action receipt is invented before a real generated transition.
            ordinal: entry.offset,
            authority_generation: generation,
            origin_logical_digest: Identity256::from_bytes(Sha256::digest(&source.origin).into()),
            authority_closure_digest: authority_digest,
            ..TokenProvenance::default()
        };
        // SAFETY: TokenProvenance is a padding-free repr(C) integral record.
        // Every field was initialized; the private record digest starts at zero.
        let logical = unsafe {
            std::slice::from_raw_parts(
                (&record as *const TokenProvenance).cast::<u8>(),
                size_of::<TokenProvenance>(),
            )
        };
        record.record_digest = Identity256::from_bytes(Sha256::digest(logical).into());
        // SAFETY: the same padding-free initialized record is copied, not borrowed.
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                (&record as *const TokenProvenance).cast::<u8>(),
                size_of::<TokenProvenance>(),
            )
        });
    }
    if seen.len() != mapping.len() {
        return Err(invalid(
            "source mapping includes a MASK, PAD, or absent logical position",
        ));
    }
    let capacity_bytes = usize::try_from(capacity)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<TokenProvenance>()))
        .ok_or_else(|| invalid("source ledger capacity overflows"))?;
    Ok((
        ring,
        prefix,
        SemanticStateRecord {
            role: SemanticStateRole::TokenProvenanceRecords,
            index: 0,
            bytes,
            capacity_bytes,
        },
    ))
}

/// Original model outputs for one acquired parent. The Session supplies their
/// instance, publication word, task and generation binding from its real lease.
pub struct SemanticContinuationInput {
    pub tensors: Vec<SemanticTensorInput>,
    /// Exact fresh controller snapshot used to authorize this invocation.
    /// This is not an independently issuable model input or authority claim.
    pub authority_decisions: Vec<u8>,
    /// Original U64[32,2] mapping and U64[1] count, never host-compacted.
    pub text_rows: DlpackManagedTensor,
    pub text_row_count: DlpackManagedTensor,
    /// Original Bool8[32], indexed by acquired source slot.
    pub selected_text: DlpackManagedTensor,
    /// Original U64[A,4] physical mapping and U64[1] count; holes remain holes.
    pub active_rows: DlpackManagedTensor,
    pub active_row_count: DlpackManagedTensor,
    /// Original Bool8[1] ordinary numerical admission predicate.
    pub numerical_admissibility: DlpackManagedTensor,
}

fn continuation_service_layouts(
    first_index: usize,
    capacity: u64,
) -> Result<[SemanticTensorLayout; 6], SemanticTransitionError> {
    let first_index = u64::try_from(first_index)
        .ok()
        .filter(|index| index.checked_add(5).is_some())
        .ok_or_else(|| publication_input_error("continuation service roster overflow"))?;
    if capacity < 34 {
        return Err(publication_input_error(
            "continuation active capacity lacks its source and feedback slots",
        ));
    }
    let layout = |offset, scalar_type, element_bytes, rank, dimensions| {
        canonical_tensor_layout(SemanticTensorLayout {
            role: 0,
            index: first_index + offset,
            scalar_type,
            element_bytes,
            rank,
            dimensions,
            logical_axis: u64::MAX,
            strides_bytes: [0; 4],
        })
    };
    Ok([
        layout(0, 3, 8, 2, [32, 2, 0, 0])?,
        layout(1, 3, 8, 1, [1, 0, 0, 0])?,
        layout(2, 8, 1, 1, [32, 0, 0, 0])?,
        layout(3, 3, 8, 2, [capacity, 4, 0, 0])?,
        layout(4, 3, 8, 1, [1, 0, 0, 0])?,
        layout(5, 8, 1, 1, [1, 0, 0, 0])?,
    ])
}

/// Identity of one packed active-cache row in its original forward. Kinds are
/// FILLED=1, MASK=2 and FEEDBACK=3; physical_row is not the packed coordinate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticActiveRow {
    pub physical_row: u64,
    pub source_slot: u64,
    pub logical_position: u64,
    pub kind: u64,
}

/// Acquired bank-local active-cache state. A computed empty result has no rows
/// but is distinct from the initial parent which has never consumed a forward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticActiveRows {
    pub computed: bool,
    pub rows: Vec<SemanticActiveRow>,
}

/// One original compact model row; source_slot is the acquired ring identity,
/// never the packed model coordinate. The row's array index identifies logits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTextRow {
    pub source_slot: u64,
    pub logical_position: u64,
}
// SAFETY: padding-free C representation containing only two initialized u64s.
unsafe impl DeviceRepr for SemanticTextRow {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TextBinding {
    rows: u64,
    count: u64,
    selected: u64,
}

struct TextBindingStorage {
    inputs: [PreparedSemanticTensor; 6],
    _witness: SemanticTensorContentWitness,
    consumer_stream: u64,
    parent: TextBindingParent,
}

#[expect(
    clippy::large_enum_variant,
    reason = "the published header stays inline on the latency-sensitive export path"
)]
enum TextBindingParent {
    Published {
        bank: usize,
        header: PublicationHeader,
    },
    Prepared(Arc<PreparedStepInputs>),
}

impl TextBindingStorage {
    fn published_parent(&self) -> Result<(usize, PublicationHeader), SemanticTransitionError> {
        match &self.parent {
            TextBindingParent::Published { bank, header } => Ok((*bank, *header)),
            TextBindingParent::Prepared(_) => Err(publication_input_error(
                "a prepared invocation has no host-observed parent header",
            )),
        }
    }
    fn descriptor(&self) -> TextBinding {
        TextBinding {
            rows: self.inputs[0].data,
            count: self.inputs[1].data,
            selected: self.inputs[2].data,
        }
    }

    fn record(&self, recorder: &mut LaunchRecorder) {
        for input in &self.inputs {
            if let Some(source) = &input.source {
                recorder.read(source);
            }
        }
    }

    fn continuation_inputs(
        &self,
        kind: SemanticTransitionKind,
        authority_bytes: u64,
        training_selection: u64,
        model_update_bindings: u64,
        model_update_binding_count: u64,
        model_update_admissibility: u64,
    ) -> ContinuationInputs {
        ContinuationInputs {
            text: self.descriptor(),
            active_rows: self.inputs[3].data,
            active_row_count: self.inputs[4].data,
            numerical_admissibility: self.inputs[5].data,
            transition_kind: kind.code(),
            authority_bytes,
            training_selection,
            model_update_bindings,
            model_update_binding_count,
            model_update_admissibility,
        }
    }
}

/// Actual device-validated publication identity. This identifies content and
/// lineage; it is not a use-authority token or an externally issuable lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticPublishedIdentity {
    pub instance: Identity256,
    pub word: u64,
    pub logical_digest: Identity256,
    pub state_digest: Identity256,
}

/// Model metadata projected from one acquired publication, not supplied by the
/// model. Content identities grant no authority independently of that reader.
pub struct SemanticModelContext {
    pub parent: SemanticPublishedIdentity,
    pub descriptor_digest: Identity256,
    /// Original storage/view geometry and device-sealed numerical state. These
    /// do not replace the opaque ModelContract or grant independent authority.
    pub numerical_state: (Identity256, Identity256),
    pub semantic_root: (u64, u64, u64, Identity256, [u64; 3]),
    /// Original prefix identity U8[32] and committed extent U64[1] device views.
    pub prefix: (DlpackManagedTensor, DlpackManagedTensor),
    /// Original ring head U64[1], not a host scalar or a source rotation.
    pub ring_head: DlpackManagedTensor,
    /// Original source slots U64[32,8], in physical slot order.
    pub source: DlpackManagedTensor,
    /// Model generation, neural bank, neural generation, cache and authority.
    pub generations: (u64, u64, u64, u64, u64),
    pub topology_identity: Identity256,
    pub table_identity: Identity256,
    /// Prefix capacity, feedback capacity and exclusive position-table bound.
    pub capacities: (u64, u64, u64),
}

/// Direct read-only aliases of the selected resident neural bank. Each backing
/// allocation is exported exactly once; storages, views and layouts preserve
/// the producer's original alias geometry. The acquired reader remains the use
/// owner and therefore prevents reuse of this physical bank through final use.
pub struct SemanticResidentModelMemory {
    pub parent: SemanticPublishedIdentity,
    /// Model generation, neural bank and neural generation.
    pub generations: (u64, u64, u64),
    /// Original storage/view geometry and sealed numerical contents.
    pub numerical_state: (Identity256, Identity256),
    pub allocations: Vec<(DeviceAllocationProvenance, DlpackManagedTensor)>,
    pub storages: Vec<SemanticModelStorage>,
    pub views: Vec<(SemanticModelView, SemanticTensorLayout)>,
}

/// Admission cost of the next operation, checked against one acquired parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTransitionKind {
    Proposal,
    Recompute,
    Drain,
    Update,
}

impl SemanticTransitionKind {
    fn code(self) -> u64 {
        match self {
            Self::Proposal => 1,
            Self::Recompute => 2,
            Self::Drain => 3,
            Self::Update => 4,
        }
    }
}

fn is_tensor_role(role: u64) -> bool {
    matches!(role, 4..=13 | 18..=25 | 51..=54)
}

fn validate_publication_counts(counts: &[u64; 55]) -> Result<usize, SemanticTransitionError> {
    let mut total = 0u64;
    for (index, &count) in counts.iter().enumerate() {
        let role = index as u64 + 1;
        if (!is_tensor_role(role) && count != if role == 16 { 3 } else { 1 })
            || (matches!(role, 8..=13) && count != 1)
            || (role == 18 && count == 0)
        {
            return Err(publication_input_error(
                "publication role count omits or duplicates a required owner",
            ));
        }
        total = total
            .checked_add(count)
            .ok_or_else(|| publication_input_error("role count overflow"))?;
    }
    for (left, right) in [(4, 5), (6, 7), (4, 51), (5, 52), (6, 53), (7, 54)] {
        if counts[left - 1] != counts[right - 1] {
            return Err(publication_input_error(
                "paired cache and active tensor layer counts differ",
            ));
        }
    }
    usize::try_from(total).map_err(|_| publication_input_error("role count exceeds host extent"))
}

fn tensor_layout_bytes(layout: &SemanticTensorLayout) -> Result<usize, SemanticTransitionError> {
    if layout.rank > 4
        || !matches!(
            (layout.scalar_type, layout.element_bytes),
            (1, 1) | (2, 4) | (3, 8) | (4, 2) | (5, 2) | (6, 4) | (7, 8) | (8, 1)
        )
        || (layout.logical_axis != u64::MAX && layout.logical_axis >= layout.rank)
        || layout.dimensions[layout.rank as usize..]
            .iter()
            .any(|&n| n != 0)
        || layout.strides_bytes[layout.rank as usize..]
            .iter()
            .any(|&n| n != 0)
    {
        return Err(publication_input_error(
            "invalid tensor rank, scalar type, or unused dimensions",
        ));
    }
    if layout.dimensions[..layout.rank as usize].contains(&0) {
        return Ok(0);
    }
    let mut axes: Vec<_> = (0..layout.rank as usize)
        .filter(|&i| layout.dimensions[i] > 1)
        .collect();
    axes.sort_by_key(|&i| layout.strides_bytes[i]);
    let mut span = layout.element_bytes;
    for axis in axes {
        let stride = layout.strides_bytes[axis];
        if stride < span || !stride.is_multiple_of(layout.element_bytes) {
            return Err(publication_input_error(
                "overlapping or unaligned tensor layout",
            ));
        }
        span = stride
            .checked_mul(layout.dimensions[axis] - 1)
            .and_then(|n| span.checked_add(n))
            .ok_or_else(|| publication_input_error("tensor byte extent overflow"))?;
    }
    usize::try_from(span).map_err(|_| publication_input_error("tensor extent exceeds host size"))
}

fn empty_model_tensor_layout(layout: &SemanticTensorLayout) -> bool {
    matches!(layout.role, 18..=25)
        && layout.logical_axis == u64::MAX
        && tensor_layout_bytes(layout).is_ok_and(|bytes| bytes == 0)
}

fn prefix_capacity_layout(
    mut layout: SemanticTensorLayout,
    capacity: u64,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    if !matches!(layout.role, 4 | 5)
        || layout.rank != 4
        || layout.logical_axis != 2
        || layout.dimensions[0] != 1
        || layout.dimensions[2] > capacity
    {
        return Err(publication_input_error(
            "prefix cache layout must be batch-one rank-four with position axis two",
        ));
    }
    layout.dimensions[2] = capacity;
    canonical_tensor_layout(layout)
}

fn canonical_tensor_layout(
    mut layout: SemanticTensorLayout,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    if layout.rank > 4 {
        return Err(publication_input_error("canonical tensor rank is invalid"));
    }
    let mut stride = layout.element_bytes;
    for axis in (0..layout.rank as usize).rev() {
        layout.strides_bytes[axis] = stride;
        stride = stride
            .checked_mul(layout.dimensions[axis].max(1))
            .ok_or_else(|| publication_input_error("canonical tensor stride overflow"))?;
    }
    tensor_layout_bytes(&layout)?;
    Ok(layout)
}

fn validate_active_capacity(
    layout: &SemanticTensorLayout,
    capacity: u64,
) -> Result<(), SemanticTransitionError> {
    tensor_layout_bytes(layout)?;
    let axis = if matches!(layout.role, 51 | 52) {
        if layout.rank != 4 || layout.dimensions[0] != 1 {
            return Err(publication_input_error(
                "active attention must be batch-one rank-four",
            ));
        }
        2
    } else if matches!(layout.role, 53 | 54) {
        0
    } else {
        return Err(publication_input_error(
            "active capacity requires an active tensor role",
        ));
    };
    if layout.logical_axis != axis
        || layout.dimensions[axis as usize] != capacity
        || capacity == 0
        || layout.dimensions[..layout.rank as usize].contains(&0)
        || !matches!(layout.scalar_type, 4..=6)
    {
        return Err(publication_input_error(
            "active tensor row axis must have the exact window plus feedback capacity",
        ));
    }
    Ok(())
}

fn continuation_capacity_layout(
    mut model: SemanticTensorLayout,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    tensor_layout_bytes(&model)?;
    if !continuation_role(model.role) || !is_tensor_role(model.role) {
        return Err(publication_input_error(
            "continuation layout has no forward tensor owner",
        ));
    }
    let active = matches!(model.role, 51..=54);
    let suffix = matches!(model.role, 4 | 5);
    let axis = if active {
        if matches!(model.role, 51 | 52) {
            2
        } else {
            0
        }
    } else if suffix {
        2
    } else {
        u64::MAX
    };
    if model.logical_axis != axis || (axis != u64::MAX && axis >= model.rank) {
        return Err(publication_input_error(
            "continuation tensor has the wrong original row axis",
        ));
    }
    if suffix {
        model.dimensions[2] = 0;
        prefix_capacity_layout(model, 32)
    } else {
        canonical_tensor_layout(model)
    }
}

fn continuation_allocation_layout(
    model: SemanticTensorLayout,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    continuation_capacity_layout(model)?;
    if matches!(model.role, 4 | 5) {
        canonical_tensor_layout(model)
    } else {
        continuation_capacity_layout(model)
    }
}

fn continuation_tensor_layout_for_kind(
    actual: &SemanticTensorLayout,
    begin: u64,
    end: u64,
    model: &SemanticTensorLayout,
    kind: SemanticTransitionKind,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    if kind != SemanticTransitionKind::Update || !matches!(model.role, 4 | 5) {
        return continuation_tensor_layout(actual, begin, end, model);
    }
    let expected = canonical_tensor_layout(*model)?;
    let destination = canonical_tensor_layout(*actual)?;
    if destination != expected {
        return Err(publication_input_error(
            "continuation tensor differs from its transition-specific fixed-capacity layout",
        ));
    }
    tensor_layout_bytes(actual)?;
    let capacity = if expected.logical_axis == u64::MAX {
        0
    } else {
        expected.dimensions[expected.logical_axis as usize]
    };
    if (begin, end) != (0, capacity) {
        return Err(publication_input_error(
            "continuation transport differs from its transition-specific fixed-capacity output",
        ));
    }
    Ok(destination)
}

fn continuation_tensor_layout(
    actual: &SemanticTensorLayout,
    begin: u64,
    end: u64,
    model: &SemanticTensorLayout,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    let expected = continuation_capacity_layout(*model)?;
    let destination = canonical_tensor_layout(*actual)?;
    if destination != expected {
        return Err(publication_input_error(
            "continuation tensor differs from its cold fixed-capacity layout",
        ));
    }
    // Validate original strides too: canonicalization must not conceal an
    // invalid producer view before the recorded device-to-device snapshot.
    tensor_layout_bytes(actual)?;
    let capacity = if expected.logical_axis == u64::MAX {
        0
    } else {
        expected.dimensions[expected.logical_axis as usize]
    };
    if (begin, end) != (0, capacity) {
        return Err(publication_input_error(
            "continuation transport differs from its original fixed-capacity output",
        ));
    }
    Ok(destination)
}

/// Cold copy geometry only: no tensor value is inspected or copied through host memory.
fn tensor_copy_plan(
    source: &SemanticTensorLayout,
    destination: &SemanticTensorLayout,
) -> Result<Vec<(u64, u64, usize)>, SemanticTransitionError> {
    tensor_layout_bytes(source)?;
    tensor_layout_bytes(destination)?;
    if source.rank != destination.rank
        || source.scalar_type != destination.scalar_type
        || source.element_bytes != destination.element_bytes
        || (0..source.rank as usize)
            .any(|axis| source.dimensions[axis] > destination.dimensions[axis])
    {
        return Err(publication_input_error(
            "tensor snapshot layout does not fit its owned destination",
        ));
    }
    if source.dimensions[..source.rank as usize].contains(&0) {
        return Ok(Vec::new());
    }
    let mut first_outer = source.rank as usize;
    let mut chunk = source.element_bytes;
    while first_outer > 0 {
        let axis = first_outer - 1;
        if source.strides_bytes[axis] != chunk || destination.strides_bytes[axis] != chunk {
            break;
        }
        chunk = chunk
            .checked_mul(source.dimensions[axis])
            .ok_or_else(|| publication_input_error("copy chunk overflow"))?;
        first_outer = axis;
    }
    let count = source.dimensions[..first_outer]
        .iter()
        .try_fold(1u64, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| publication_input_error("copy row count overflow"))?;
    let mut result = Vec::new();
    result
        .try_reserve(
            usize::try_from(count)
                .map_err(|_| publication_input_error("copy row count exceeds host extent"))?,
        )
        .map_err(|error| runtime_error("copy plan allocation", error))?;
    for mut linear in 0..count {
        let (mut source_offset, mut destination_offset) = (0u64, 0u64);
        for axis in (0..first_outer).rev() {
            let position = linear % source.dimensions[axis];
            linear /= source.dimensions[axis];
            source_offset += position * source.strides_bytes[axis];
            destination_offset += position * destination.strides_bytes[axis];
        }
        result.push((source_offset, destination_offset, chunk as usize));
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct TensorMetadata<'a> {
    device_type: i32,
    device_id: i32,
    dtype: crate::dlpack::DLDataType,
    shape: &'a [i64],
    strides: Option<&'a [i64]>,
    data: u64,
    byte_offset: u64,
}

// SAFETY: the caller retains the genuine managed producer and its immutable
// shape/stride arrays for the returned borrow. A scalar has no shape entries;
// its DLPack shape and strides may be null and must never form a raw Rust slice.
unsafe fn semantic_tensor_metadata(
    actual: &crate::dlpack::DLTensor,
) -> Result<TensorMetadata<'_>, SemanticTransitionError> {
    if !(0..=4).contains(&actual.ndim) || (actual.ndim != 0 && actual.shape.is_null()) {
        return Err(publication_input_error(
            "semantic tensor requires an original scalar through rank-four shape",
        ));
    }
    let rank = actual.ndim as usize;
    Ok(TensorMetadata {
        device_type: actual.device.device_type,
        device_id: actual.device.device_id,
        dtype: actual.dtype,
        // SAFETY: a nonempty shape was checked above and is owned by the caller.
        shape: if rank == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(actual.shape, rank) }
        },
        strides: if rank == 0 || actual.strides.is_null() {
            None
        } else {
            // SAFETY: DLPack's retained producer owns ndim stride elements.
            Some(unsafe { std::slice::from_raw_parts(actual.strides, rank) })
        },
        data: actual.data as usize as u64,
        byte_offset: actual.byte_offset,
    })
}

fn publication_input_error(detail: &str) -> SemanticTransitionError {
    SemanticTransitionError::InvalidInput {
        detail: detail.into(),
    }
}

fn validate_tensor_metadata(
    declared: &SemanticTensorLayout,
    actual: &TensorMetadata<'_>,
    device_id: i32,
) -> Result<(u64, usize), SemanticTransitionError> {
    let invalid = || {
        publication_input_error("semantic tensor metadata differs from its exact model contract")
    };
    let (dtype_code, bits) = match declared.scalar_type {
        1 => (crate::dlpack::K_DLUINT, 8),
        2 => (crate::dlpack::K_DLUINT, 32),
        3 => (crate::dlpack::K_DLUINT, 64),
        4 => (crate::dlpack::K_DLFLOAT, 16),
        5 => (4, 16), // DLPack's distinct bfloat type code, not IEEE float16.
        6 => (crate::dlpack::K_DLFLOAT, 32),
        7 => (0, 64), // Signed token indices are not unsigned storage aliases.
        8 => (6, 8),  // DLPack's distinct Boolean code.
        _ => return Err(invalid()),
    };
    let rank = actual.shape.len();
    if actual.device_type != crate::dlpack::K_DLCUDA
        || actual.device_id != device_id
        || rank > 4
        || declared.rank != rank as u64
        || actual.dtype.code != dtype_code
        || actual.dtype.bits != bits
        || actual.dtype.lanes != 1
        || declared.element_bytes != u64::from(bits / 8)
        || (declared.role != 0 && SemanticStateRole::from_code(declared.role).is_none())
        || (declared.logical_axis != u64::MAX && declared.logical_axis >= declared.rank)
        || declared.dimensions[rank..]
            .iter()
            .any(|&dimension| dimension != 0)
        || declared.strides_bytes[rank..]
            .iter()
            .any(|&stride| stride != 0)
    {
        return Err(invalid());
    }
    let element_bytes = declared.element_bytes;
    let mut dimensions = [0u64; 4];
    for (index, &dimension) in actual.shape.iter().enumerate() {
        dimensions[index] = u64::try_from(dimension).map_err(|_| invalid())?;
    }
    if dimensions != declared.dimensions {
        return Err(invalid());
    }
    let mut strides = [0u64; 4];
    if let Some(actual_strides) = actual.strides {
        if actual_strides.len() != rank {
            return Err(invalid());
        }
        for (index, &stride) in actual_strides.iter().enumerate() {
            strides[index] = u64::try_from(stride)
                .ok()
                .and_then(|stride| stride.checked_mul(element_bytes))
                .ok_or_else(invalid)?;
        }
    } else {
        let mut stride = element_bytes;
        for index in (0..rank).rev() {
            strides[index] = stride;
            stride = stride.checked_mul(dimensions[index]).ok_or_else(invalid)?;
        }
    }
    if strides != declared.strides_bytes {
        return Err(invalid());
    }
    let empty = dimensions[..rank].contains(&0);
    let bytes = if empty {
        0
    } else {
        // Positive non-overlapping strides support real transposed and padded
        // views. Broadcast/overlapping cells are not a mutable state contract.
        let mut axes = (0..rank)
            .filter(|&axis| dimensions[axis] > 1)
            .collect::<Vec<_>>();
        axes.sort_by_key(|&axis| strides[axis]);
        let mut span = element_bytes;
        for axis in axes {
            if strides[axis] < span {
                return Err(invalid());
            }
            span = strides[axis]
                .checked_mul(dimensions[axis] - 1)
                .and_then(|bytes| span.checked_add(bytes))
                .ok_or_else(invalid)?;
        }
        span
    };
    if !actual.byte_offset.is_multiple_of(element_bytes) || (bytes != 0 && actual.data == 0) {
        return Err(invalid());
    }
    let pointer = actual
        .data
        .checked_add(actual.byte_offset)
        .ok_or_else(invalid)?;
    if !pointer.is_multiple_of(element_bytes) || pointer.checked_add(bytes).is_none() {
        return Err(invalid());
    }
    let bytes = usize::try_from(bytes).map_err(|_| invalid())?;
    usize::try_from(pointer).map_err(|_| invalid())?;
    Ok((pointer, bytes))
}

#[derive(Clone)]
struct PreparedSemanticTensor {
    layout: SemanticTensorLayout,
    logical_begin: u64,
    logical_end: u64,
    data: u64,
    source: Option<DeviceMemoryView<u8>>,
    native_allocation: Option<DeviceAllocationProvenance>,
    // Empty views still retain their genuine producer; releasing its token can
    // retire a larger backing allocation whose producer work is still pending.
    _empty_owner: Option<Arc<DlpackManagedTensor>>,
}

#[cfg(feature = "semantic-policy")]
unsafe fn policy_score_cotangent_metadata(
    actual: &crate::dlpack::DLTensor,
    device_id: i32,
) -> Result<(u64, usize), SemanticTransitionError> {
    // SAFETY: the caller retains the original shape/stride descriptor arrays.
    let (rows, scalar, data, bytes) =
        unsafe { crate::dlpack::dlpack_tensor_metadata(device_id, actual) }.map_err(|error| {
            publication_input_error(&format!("selected-score cotangent metadata: {error}"))
        })?;
    if rows != COMPONENT_COUNT as u64
        || scalar != xlog_core::ScalarType::F64
        || data.checked_add(bytes as u64).is_none()
    {
        return Err(publication_input_error(
            "selected-score cotangents require exact contiguous FP64 storage",
        ));
    }
    Ok((data, bytes))
}

struct RetainedTensorAdmission<T: Send + 'static> {
    inputs: Option<Vec<T>>,
    stream: Arc<CudaStream>,
    ready: bool,
}

impl<T: Send + 'static> Drop for RetainedTensorAdmission<T> {
    fn drop(&mut self) {
        let Some(inputs) = self.inputs.take() else {
            return;
        };
        if self.ready {
            drop(inputs);
            return;
        }
        // Preserve the genuine producer tokens through a failed/unknown cold
        // handoff. Canonical retirement retries readiness and runs deleters once.
        crate::cuda_graph::retire_resources_after_completion(
            (Some(inputs), Arc::clone(&self.stream)),
            |owners| owners.1.synchronize(),
            |owners| {
                drop(owners.0.take());
                Ok(())
            },
            |_| Ok(()),
            |_, error| eprintln!("semantic tensor producer retirement incomplete: {error}"),
        );
    }
}

// Metadata/ownership admission only. The caller must order producer readiness
// before reading any device cells; cold import above waits, content guards use
// device event edges and retain all owners until explicit reader release.
fn prepare_semantic_tensors(
    provider: &CudaKernelProvider,
    inputs: Vec<SemanticTensorInput>,
) -> Result<Vec<PreparedSemanticTensor>, SemanticTransitionError> {
    let stream = Arc::clone(provider.device().inner().stream());
    let mut pending = RetainedTensorAdmission {
        inputs: Some(inputs),
        stream: Arc::clone(&stream),
        ready: false,
    };
    let mut geometry = Vec::new();
    for input in pending
        .inputs
        .as_ref()
        .expect("retained original producer roster")
    {
        let pointer = input.tensor.as_ptr();
        if pointer.is_null() {
            return Err(publication_input_error(
                "semantic tensor managed owner is null",
            ));
        }
        // SAFETY: the genuine managed producer owns valid immutable DLPack
        // metadata for this import. Rank is bounded before reading its arrays;
        // the caller owns the managed record and orders its producer before use.
        let actual = unsafe { &(*pointer).dl_tensor };
        // SAFETY: the complete original producer roster remains in pending.
        let metadata = unsafe { semantic_tensor_metadata(actual) }?;
        let (data, bytes) =
            validate_tensor_metadata(&input.layout, &metadata, provider.device().ordinal() as i32)?;
        geometry.push((data, bytes));
    }
    // Every fallible metadata check precedes transfer of any original owner.
    // Failure retires the complete producer roster on its negotiated stream.
    // Success transfers ownership to the views, without a host wait.
    pending.ready = true;
    let inputs = pending
        .inputs
        .take()
        .expect("retained original producer roster");
    let mut imported = Vec::with_capacity(inputs.len());
    for (input, (data, bytes)) in inputs.into_iter().zip(geometry) {
        let (source, empty_owner) = if bytes == 0 {
            // No invented device cells, but keep the actual managed owner until
            // the caller's cold completion or content-reader retirement.
            (None, Some(Arc::new(input.tensor)))
        } else {
            // SAFETY: exact original dtype/shape/stride/offset/device bounds and
            // producer readiness ordering is the caller's responsibility. This is the
            // canonical managed owner, not a borrowed raw-pointer slice.
            let column =
                unsafe { CudaColumn::dlpack(data, bytes, Arc::clone(&stream), input.tensor) };
            (Some(column.device_view()), None)
        };
        imported.push(PreparedSemanticTensor {
            layout: input.layout,
            logical_begin: input.logical_begin,
            logical_end: input.logical_end,
            data,
            source,
            native_allocation: input.native_allocation,
            _empty_owner: empty_owner,
        });
    }
    Ok(imported)
}

fn complete_tensor_handoff<T: Send + 'static>(
    provider: &CudaKernelProvider,
    inputs: Vec<T>,
) -> Result<Vec<T>, SemanticTransitionError> {
    let stream = Arc::clone(provider.device().inner().stream());
    let mut pending = RetainedTensorAdmission {
        inputs: Some(inputs),
        stream,
        ready: false,
    };
    let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&pending.stream)
        .map_err(|error| runtime_error("tensor producer stream admission", error))?;
    pending
        .stream
        .synchronize()
        .map_err(|error| runtime_error("tensor producer handoff", error))?;
    pending.ready = true;
    Ok(pending
        .inputs
        .take()
        .expect("retained tensor admission owns its inputs"))
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationStorageEntry {
    pointer: u64,
    bytes: u64,
    generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationRange {
    role: u64,
    index: u64,
    storage_slot: u64,
    generation: u64,
    offset_bytes: u64,
    length_bytes: u64,
    logical_begin: u64,
    logical_end: u64,
    digest: Identity256,
    backing_digest: Identity256,
}

/// Descriptor integrity includes local handles; logical content identities do
/// not. Every physical range resolves through the owner's fixed storage table.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct PublicationHeader {
    abi: u64,
    instance: Identity256,
    recovered_instance: Identity256,
    sealed_epoch: u64,
    base_word: u64,
    publication_word: u64,
    ring_head: u64,
    prefix_extent: u64,
    semantic_owner: u64,
    semantic_slot: u64,
    semantic_generation: u64,
    semantic_digest: Identity256,
    semantic_extents: [u64; 3],
    model_generation: u64,
    neural_bank: u64,
    neural_generation: u64,
    cache_generation: u64,
    authority_generation: u64,
    terminal: u64,
    terminal_position: u64,
    fuel: u64,
    proposal: u64,
    stream_serial: u64,
    family_id: u64,
    training_cursor: u64,
    training_rng: [u64; 4],
    range_count: u64,
    model_geometry_digest: Identity256,
    model_numerical_digest: Identity256,
    logical_digest: Identity256,
    state_digest: Identity256,
    descriptor_digest: Identity256,
}

impl PublicationHeader {
    fn rng_binding(&self) -> Result<SemanticRngBinding, SemanticTransitionError> {
        if self.stream_serial > u64::MAX >> 8 {
            return Err(SemanticTransitionError::GenerationExhausted);
        }
        Ok(SemanticRngBinding {
            model_generation: u32::try_from(self.model_generation)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            family_id: u8::try_from(self.family_id)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            stream_serial: self.stream_serial,
            proposal: u32::try_from(self.proposal)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationControl {
    abi: u64,
    word: u64,
    instance: Identity256,
    banks: [u64; 2],
    directories: [u64; 2],
    storage: u64,
    storage_count: u64,
    contract: u64,
    continuation: u64,
    reader_counts: [u64; 2],
    reader_gate: u64,
    refusal: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PublicationBank {
    header: PublicationHeader,
    source: [SourceSlot; 32],
    state: DeviceState,
    receipts: [SemanticTransitionReceipt; COMPONENT_COUNT],
}

#[derive(Clone, Copy)]
enum PublicationBankField {
    Source,
    PrefixExtent,
    RingHead,
    Terminal,
}

impl PublicationBankField {
    fn layout(self) -> (std::ops::Range<usize>, Vec<i64>, Vec<i64>) {
        let (offset, bytes, shape, strides) = match self {
            Self::Source => (
                std::mem::offset_of!(PublicationBank, source),
                size_of::<[SourceSlot; 32]>(),
                vec![32, 8],
                vec![8, 1],
            ),
            Self::PrefixExtent => (
                std::mem::offset_of!(PublicationBank, header)
                    + std::mem::offset_of!(PublicationHeader, prefix_extent),
                size_of::<u64>(),
                vec![1],
                vec![1],
            ),
            Self::RingHead => (
                std::mem::offset_of!(PublicationBank, header)
                    + std::mem::offset_of!(PublicationHeader, ring_head),
                size_of::<u64>(),
                vec![1],
                vec![1],
            ),
            Self::Terminal => (
                std::mem::offset_of!(PublicationBank, header)
                    + std::mem::offset_of!(PublicationHeader, terminal),
                size_of::<u64>(),
                vec![1],
                vec![1],
            ),
        };
        (offset..offset + bytes, shape, strides)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationRoleCount {
    role: u64,
    count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationContract {
    abi: u64,
    range_capacity: u64,
    prefix_capacity: u64,
    window_capacity: u64,
    feedback_capacity: u64,
    max_position: u64,
    pad_token: u64,
    terminal_tokens: u64,
    terminal_token_count: u64,
    final_intent_payload_bytes: u64,
    model_generation: u64,
    policy_generation: u64,
    authority_generation: u64,
    semantic_owner: u64,
    task_identity: Identity256,
    topology_identity: Identity256,
    table_identity: Identity256,
    role_counts: u64,
    role_count: u64,
    model_contract_layout: SemanticModelContractLayout,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PendingContinuation {
    abi: u64,
    instance: Identity256,
    base_word: u64,
    model_generation: u64,
    authority_generation: u64,
    topology_identity: Identity256,
    table_identity: Identity256,
    prefix_identity: Identity256,
    ranges: u64,
    range_count: u64,
    transition_kind: u64,
    text: TextBinding,
    numerical_admissibility: u64,
    training_selection: u64,
    model_update_bindings: u64,
    model_update_binding_count: u64,
    model_update_admissibility: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContinuationInputs {
    text: TextBinding,
    active_rows: u64,
    active_row_count: u64,
    numerical_admissibility: u64,
    transition_kind: u64,
    authority_bytes: u64,
    training_selection: u64,
    model_update_bindings: u64,
    model_update_binding_count: u64,
    model_update_admissibility: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModelUpdateBinding {
    source: u64,
    bytes: u64,
    slots: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationCommand {
    control: u64,
    lease: u64,
    operation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationLease {
    abi: u64,
    status: u64,
    instance: Identity256,
    word: u64,
    bank: u64,
    epoch: u64,
    active: u64,
    transition_kind: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CompletionCoverage {
    abi: u64,
    instance: Identity256,
    base_word: u64,
    model_generation: u64,
    topology_identity: Identity256,
    table_identity: Identity256,
    prefix_identity: Identity256,
    prefix_begin: u64,
    prefix_end: u64,
    source_digest: Identity256,
    attention_digest: Identity256,
    linear_digest: Identity256,
    value_digest: Identity256,
    edit_digest: Identity256,
    receipt_digest: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawFeedbackRecord {
    valid: u64,
    statement_index: u64,
    statement_offset_bytes: u64,
    statement_length_bytes: u64,
    pro: u64,
    contra: u64,
    query_receipt: [u64; 42],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttemptReceipt {
    abi: u64,
    instance: Identity256,
    base_word: u64,
    next_word: u64,
    logical_digest: Identity256,
    action_receipts_digest: Identity256,
    semantic_receipts_digest: Identity256,
    coverage_digest: Identity256,
    replay_head_digest: Identity256,
    intent_head_digest: Identity256,
    acknowledgement_head_digest: Identity256,
    previous_attempt_digest: Identity256,
    receipt_digest: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IntentQueueHeader {
    abi: u64,
    count: u64,
    capacity: u64,
    payload_used_bytes: u64,
    payload_capacity_bytes: u64,
    effect_offset_bytes: u64,
    effect_length_bytes: u64,
    chain_head: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IntentEntry {
    stable_identity: Identity256,
    checkpoint_lineage: Identity256,
    base_logical: Identity256,
    effective_delta: Identity256,
    result_logical: Identity256,
    effect_digest: Identity256,
    payload_digest: Identity256,
    payload_offset: u64,
    payload_len: u64,
    audit_instance: Identity256,
    audit_epoch: u64,
    previous_chain: Identity256,
    chain: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TokenProvenance {
    source_slot: u64,
    logical_position: u64,
    token: u64,
    base_word: u64,
    proposal: u64,
    ordinal: u64,
    action_receipt_digest: Identity256,
    authority_generation: u64,
    origin_logical_digest: Identity256,
    authority_closure_digest: Identity256,
    record_digest: Identity256,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTensorLayout {
    pub role: u64,
    pub index: u64,
    pub element_bytes: u64,
    // U8, U32, U64, F16, BF16, F32, I64, Bool8 respectively.
    // Equal widths are not equal types.
    pub scalar_type: u64,
    pub rank: u64,
    pub logical_axis: u64,
    pub dimensions: [u64; 4],
    pub strides_bytes: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TensorLayoutTableHeader {
    abi: u64,
    layout_count: u64,
    active_computed: u64,
    active_row_count: u64,
}

fn tensor_table_capacity(
    layout_count: usize,
    row_count: u64,
) -> Result<usize, SemanticTransitionError> {
    layout_count
        .checked_mul(size_of::<SemanticTensorLayout>())
        .and_then(|n| n.checked_add(size_of::<TensorLayoutTableHeader>()))
        .and_then(|n| {
            usize::try_from(row_count)
                .ok()?
                .checked_mul(size_of::<SemanticActiveRow>())?
                .checked_add(n)
        })
        .ok_or_else(|| publication_input_error("tensor layout table extent overflow"))
}

fn tensor_table_bytes(
    layouts: &[SemanticTensorLayout],
    computed: bool,
    rows: &[SemanticActiveRow],
) -> Result<Vec<u8>, SemanticTransitionError> {
    if !computed && !rows.is_empty() {
        return Err(publication_input_error(
            "uncomputed active cache cannot contain rows",
        ));
    }
    let length = tensor_table_capacity(layouts.len(), rows.len() as u64)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| runtime_error("tensor layout table allocation", error))?;
    bytes.extend(publication_abi_bytes(&[TensorLayoutTableHeader {
        abi: 1,
        layout_count: layouts.len() as u64,
        active_computed: u64::from(computed),
        active_row_count: rows.len() as u64,
    }]));
    bytes.extend(publication_abi_bytes(layouts));
    bytes.extend(publication_abi_bytes(rows));
    Ok(bytes)
}

fn decode_tensor_table(
    bytes: &[u8],
) -> Result<
    (
        TensorLayoutTableHeader,
        Vec<SemanticTensorLayout>,
        SemanticActiveRows,
    ),
    SemanticTransitionError,
> {
    if bytes.len() < size_of::<TensorLayoutTableHeader>() {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    // SAFETY: checked complete integer-only header; byte storage need not align.
    let header = unsafe {
        bytes
            .as_ptr()
            .cast::<TensorLayoutTableHeader>()
            .read_unaligned()
    };
    let count = usize::try_from(header.layout_count)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    if header.abi != 1
        || header.active_computed > 1
        || (header.active_computed == 0 && header.active_row_count != 0)
        || tensor_table_capacity(count, header.active_row_count)? != bytes.len()
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let layouts_end =
        size_of::<TensorLayoutTableHeader>() + count * size_of::<SemanticTensorLayout>();
    let layouts = bytes[size_of::<TensorLayoutTableHeader>()..layouts_end]
        .chunks_exact(size_of::<SemanticTensorLayout>())
        // SAFETY: each chunk is a complete integer-only C record.
        .map(|chunk| unsafe {
            chunk
                .as_ptr()
                .cast::<SemanticTensorLayout>()
                .read_unaligned()
        })
        .collect();
    let rows = bytes[layouts_end..]
        .chunks_exact(size_of::<SemanticActiveRow>())
        // SAFETY: each chunk is a complete integer-only C record.
        .map(|chunk| unsafe { chunk.as_ptr().cast::<SemanticActiveRow>().read_unaligned() })
        .collect();
    Ok((
        header,
        layouts,
        SemanticActiveRows {
            computed: header.active_computed == 1,
            rows,
        },
    ))
}

// SAFETY: these C-layout records contain only integers, fixed byte arrays and
// the already checked scalar state/receipt ABI. Every bit pattern is valid.
unsafe impl DeviceRepr for PublicationStorageEntry {}
unsafe impl DeviceRepr for PublicationRange {}
unsafe impl DeviceRepr for PublicationHeader {}
unsafe impl DeviceRepr for PublicationControl {}
unsafe impl DeviceRepr for PublicationBank {}
unsafe impl DeviceRepr for PublicationRoleCount {}
unsafe impl DeviceRepr for PublicationContract {}
unsafe impl DeviceRepr for PendingContinuation {}
unsafe impl DeviceRepr for ModelUpdateBinding {}
unsafe impl DeviceRepr for PublicationCommand {}
unsafe impl DeviceRepr for PublicationLease {}
unsafe impl DeviceRepr for SemanticTensorLayout {}
// SAFETY: padding-free C records containing initialized u64 values only.
unsafe impl DeviceRepr for TensorLayoutTableHeader {}
unsafe impl DeviceRepr for SemanticActiveRow {}
unsafe impl DeviceRepr for CompletionCoverage {}
unsafe impl DeviceRepr for RawFeedbackRecord {}
unsafe impl DeviceRepr for AttemptReceipt {}
unsafe impl DeviceRepr for IntentQueueHeader {}
unsafe impl DeviceRepr for IntentEntry {}
unsafe impl DeviceRepr for TokenProvenance {}

const _: () = assert!(size_of::<SourceSlot>() == 64);
const _: () = assert!(size_of::<PublicationStorageEntry>() == 24);
const _: () = assert!(size_of::<PublicationRange>() == 128);
const _: () = assert!(size_of::<PublicationHeader>() == 488);
const _: () = assert!(size_of::<PublicationControl>() == 144);
const _: () = assert!(size_of::<PublicationBank>() == 45392);
const _: () = assert!(size_of::<PublicationRoleCount>() == 16);
const _: () = assert!(size_of::<SemanticModelContractLayout>() == 48);
const _: () = assert!(size_of::<PublicationContract>() == 272);
const _: () = assert!(size_of::<SemanticTextRow>() == 16);
const _: () = assert!(size_of::<TextBinding>() == 24);
const _: () = assert!(size_of::<PendingContinuation>() == 248);
const _: () = assert!(size_of::<ContinuationInputs>() == 96);
const _: () = assert!(size_of::<ModelUpdateBinding>() == 32);
const _: () = assert!(size_of::<PublicationCommand>() == 24);
const _: () = assert!(size_of::<PublicationLease>() == 88);
const _: () = assert!(size_of::<SemanticTensorLayout>() == 112);
const _: () = assert!(size_of::<TensorLayoutTableHeader>() == 32);
const _: () = assert!(size_of::<SemanticActiveRow>() == 32);
const _: () = assert!(size_of::<CompletionCoverage>() == 360);
const _: () = assert!(size_of::<RawFeedbackRecord>() == 384);
const _: () = assert!(size_of::<AttemptReceipt>() == 344);
const _: () = assert!(size_of::<IntentQueueHeader>() == 88);
const _: () = assert!(size_of::<IntentEntry>() == 344);
const _: () = assert!(size_of::<TokenProvenance>() == 184);

#[derive(Clone)]
enum PublicationPayload {
    Metadata(Vec<u8>),
    Tensor(PreparedSemanticTensor),
    Uncomputed,
}

struct PublicationAllocationPlan {
    role: u64,
    index: u64,
    capacity: usize,
    length: usize,
    logical_begin: u64,
    logical_end: u64,
    payload: PublicationPayload,
}

/// One reachable range of a cold publication material. Allocation capacity is
/// retained for future appends; bytes outside the sealed extent are not exported.
#[derive(Clone)]
struct PublicationMaterialRange {
    range: PublicationRange,
    capacity: usize,
    bytes: Vec<u8>,
}

/// Original non-tensor feedback material from a completed acquired step.
/// `identity` is its native range seal, not SHA-256 of `bytes` alone. Raw slots
/// (role 15), statement bytes (16) and provenance records (17) stay separate;
/// encoded feature tensors are not a replacement for any of these records.
pub struct SemanticFeedbackRecordMaterial {
    pub role: SemanticStateRole,
    pub index: u64,
    pub generation: u64,
    pub capacity_bytes: u64,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub identity: Identity256,
    pub bytes: Vec<u8>,
}

impl PublicationMaterialRange {
    fn into_feedback_record(
        self,
    ) -> Result<SemanticFeedbackRecordMaterial, SemanticTransitionError> {
        if !matches!(self.range.role, 15..=17)
            || self.range.generation != 1
            || self.range.length_bytes != self.bytes.len() as u64
            || self.bytes.is_empty()
            || self.bytes.len() > self.capacity
            || self.range.logical_begin != 0
            || (self.range.role != 15 && self.range.logical_end != 0)
            || (self.range.role == 15
                && (!self
                    .bytes
                    .len()
                    .is_multiple_of(size_of::<RawFeedbackRecord>())
                    || self.range.logical_end
                        > (self.bytes.len() / size_of::<RawFeedbackRecord>()) as u64))
            || self.original_record_digest() != self.range.digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(SemanticFeedbackRecordMaterial {
            role: SemanticStateRole::from_code(self.range.role).expect("checked feedback role"),
            index: self.range.index,
            generation: self.range.generation,
            capacity_bytes: self.capacity as u64,
            logical_begin: self.range.logical_begin,
            logical_end: self.range.logical_end,
            identity: self.range.digest,
            bytes: self.bytes,
        })
    }

    // Exact opaque-record branch of publication_content_view. Runtime words
    // must remain present here, including those normalized out of Hlogical.
    fn original_record_digest(&self) -> Identity256 {
        debug_assert!(!is_tensor_role(self.range.role));
        self.record_digest(&self.bytes)
    }

    fn record_digest(&self, bytes: &[u8]) -> Identity256 {
        let mut prefix = [0u64; 16];
        prefix[..5].copy_from_slice(&[
            0x786c6f6772616e31,
            self.range.role,
            self.range.index,
            self.range.logical_begin,
            self.range.logical_end,
        ]);
        let mut hash = Sha256::new();
        for word in prefix {
            hash.update(word.to_le_bytes());
        }
        hash.update(bytes);
        Identity256::from_bytes(hash.finalize().into())
    }

    fn attempt(&self) -> Result<AttemptReceipt, SemanticTransitionError> {
        if self.range.role != 33 || self.bytes.len() != size_of::<AttemptReceipt>() {
            return Err(publication_input_error(
                "replay evidence requires the whole native attempt receipt",
            ));
        }
        // SAFETY: the checked, padding-free ABI contains only integer/digest fields.
        Ok(unsafe { std::ptr::read_unaligned(self.bytes.as_ptr().cast::<AttemptReceipt>()) })
    }

    // Exact PublicationLogicalBytes normalization for the publication-only
    // records and completion coverage. This is a cold comparison, never a
    // substitute for the actual device publication seals.
    fn logical_record_digest(&self) -> Result<Identity256, SemanticTransitionError> {
        let mut bytes = self.bytes.clone();
        match self.range.role {
            14 | 33 => {
                let (extent, first, last, tail) = if self.range.role == 14 {
                    (size_of::<CompletionCoverage>(), 1, 5, 41)
                } else {
                    (size_of::<AttemptReceipt>(), 1, 6, 39)
                };
                if bytes.len() != extent {
                    return Err(publication_input_error(
                        "replay record has another native ABI extent",
                    ));
                }
                bytes[first * 8..(last + 1) * 8].fill(0);
                bytes[tail * 8..].fill(0);
            }
            30 => {
                if bytes.len() < size_of::<IntentQueueHeader>() {
                    return Err(publication_input_error(
                        "replay intent queue omits its native header",
                    ));
                }
                // SAFETY: the checked prefix is an all-integer, padding-free ABI.
                let header =
                    unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<IntentQueueHeader>()) };
                if header.abi != 1
                    || header.count > header.capacity
                    || header
                        .count
                        .checked_mul(size_of::<IntentEntry>() as u64)
                        .and_then(|n| n.checked_add(size_of::<IntentQueueHeader>() as u64))
                        != Some(bytes.len() as u64)
                {
                    return Err(publication_input_error(
                        "replay intent queue differs from its exact native extent",
                    ));
                }
                for entry in bytes[size_of::<IntentQueueHeader>()..]
                    .chunks_exact_mut(size_of::<IntentEntry>())
                {
                    entry[30 * 8..35 * 8].fill(0);
                }
            }
            27 | 28 | 31 | 32 => {}
            _ => {
                return Err(publication_input_error(
                    "record is not a native replay comparison record",
                ))
            }
        }
        Ok(self.record_digest(&bytes))
    }

    fn encode_into(&self, bytes: &mut Vec<u8>) -> Result<(), SemanticTransitionError> {
        material_bytes(bytes, &publication_abi_bytes(&[self.range]))
            .map_err(SemanticTransitionError::Semantic)?;
        material_u64(bytes, self.capacity as u64);
        material_bytes(bytes, &self.bytes).map_err(SemanticTransitionError::Semantic)
    }

    fn decode_from(
        reader: &mut SemanticMaterialReader<'_>,
    ) -> Result<Self, SemanticTransitionError> {
        // SAFETY: the range ABI contains only integer/digest fields.
        let range = unsafe { publication_material_record::<PublicationRange>(reader)? };
        let capacity = usize::try_from(reader.u64().map_err(SemanticTransitionError::Semantic)?)
            .map_err(|_| {
                publication_input_error("publication material capacity exceeds host extent")
            })?;
        let bytes = reader
            .bytes()
            .map_err(SemanticTransitionError::Semantic)?
            .to_vec();
        Ok(Self {
            range,
            capacity,
            bytes,
        })
    }
}

// publication_fold hashes digest bytes followed by two little-endian ABI words
// and the content digest. Identity256 preserves the CUDA SHA output byte order.
fn publication_fold_digest(
    digest: Identity256,
    role: u64,
    index: u64,
    content: Identity256,
) -> Identity256 {
    let mut hash = Sha256::new();
    hash.update(digest.as_bytes());
    hash.update(role.to_le_bytes());
    hash.update(index.to_le_bytes());
    hash.update(content.as_bytes());
    Identity256::from_bytes(hash.finalize().into())
}

fn publication_action_receipts_digest(
    codebook: &PublicationMaterialRange,
    receipts: &[SemanticTransitionReceipt],
) -> Result<Identity256, SemanticTransitionError> {
    // publication_logical_codebook_digest and PublicationActionReceiptBytes:
    // owner/slot/generation and the runtime admission binding are audit data.
    let bytes = &codebook.bytes;
    if codebook.range.role != 48 || bytes.len() < 25 * 8 || !bytes.len().is_multiple_of(8) {
        return Err(publication_input_error(
            "replay action codebook has another native extent",
        ));
    }
    let word =
        |index: usize| u64::from_le_bytes(bytes[index * 8..(index + 1) * 8].try_into().unwrap());
    let (offset, length) = (word(8), word(9));
    if offset % 8 != 0
        || offset > bytes.len() as u64
        || length > bytes.len() as u64 - offset
        || (offset + length).div_ceil(8) != (bytes.len() / 8) as u64
    {
        return Err(publication_input_error(
            "replay action codebook omits its exact statement payload",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"xlog.semantic.logical-codebooks.v1\0");
    hash.update(CATALOGUE_DIGEST);
    hash.update(((bytes.len() / 8) as u64).to_le_bytes());
    for (index, bytes) in bytes.chunks_exact(8).enumerate() {
        hash.update(if matches!(index, 14..=16 | 21..=24) {
            &[0; 8]
        } else {
            bytes
        });
    }
    let binding = Identity256::from_bytes(hash.finalize().into());
    let mut hash = Sha256::new();
    for receipt in receipts {
        let mut normalized = *receipt;
        if normalized.catalogue_generation == CATALOGUE_GENERATION {
            normalized.admission_binding = binding;
        }
        hash.update(publication_abi_bytes(&[normalized]));
    }
    Ok(Identity256::from_bytes(hash.finalize().into()))
}

fn publication_semantic_receipts_digest(receipts: &[[u64; 42]; 7]) -> Identity256 {
    let mut hash = Sha256::new();
    for receipt in receipts {
        for (index, &word) in receipt.iter().enumerate() {
            hash.update(
                if matches!(index, 3..=12 | 34..=41) {
                    0
                } else {
                    word
                }
                .to_le_bytes(),
            );
        }
    }
    Identity256::from_bytes(hash.finalize().into())
}

const REPLAY_EVIDENCE_ROLES: [u64; 6] = [27, 28, 30, 31, 32, 33];
const REPLAY_EVIDENCE_MAGIC: &[u8] = b"XLOG-REPLAY-EVIDENCE\0";
const REPLAY_PROVENANCE_MAGIC: &[u8] = b"XLOG-REPLAY-PROVENANCE\0";

fn encode_publication_identity(bytes: &mut Vec<u8>, identity: SemanticPublishedIdentity) {
    bytes.extend_from_slice(identity.instance.as_bytes());
    material_u64(bytes, identity.word);
    bytes.extend_from_slice(identity.logical_digest.as_bytes());
    bytes.extend_from_slice(identity.state_digest.as_bytes());
}

fn decode_publication_identity(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
    let instance = Identity256::from_bytes(
        reader
            .take(32)
            .map_err(SemanticTransitionError::Semantic)?
            .try_into()
            .unwrap(),
    );
    let word = reader.u64().map_err(SemanticTransitionError::Semantic)?;
    let logical_digest = Identity256::from_bytes(
        reader
            .take(32)
            .map_err(SemanticTransitionError::Semantic)?
            .try_into()
            .unwrap(),
    );
    let state_digest = Identity256::from_bytes(
        reader
            .take(32)
            .map_err(SemanticTransitionError::Semantic)?
            .try_into()
            .unwrap(),
    );
    Ok(SemanticPublishedIdentity {
        instance,
        word,
        logical_digest,
        state_digest,
    })
}

// The reconstruction provenance root contains original authority bytes and the
// original continuation decision, never a new grant or the successor snapshot.
struct PublicationReplayProvenance {
    predecessor: SemanticPublishedIdentity,
    authority: PublicationMaterialRange,
    decision: PublicationMaterialRange,
}

impl PublicationReplayProvenance {
    fn encode(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        let mut bytes = REPLAY_PROVENANCE_MAGIC.to_vec();
        material_u32(&mut bytes, 1);
        bytes.extend_from_slice(&publication_material_runtime());
        encode_publication_identity(&mut bytes, self.predecessor);
        self.authority.encode_into(&mut bytes)?;
        self.decision.encode_into(&mut bytes)?;
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        let mut reader = SemanticMaterialReader::new(bytes);
        if reader
            .take(REPLAY_PROVENANCE_MAGIC.len())
            .map_err(SemanticTransitionError::Semantic)?
            != REPLAY_PROVENANCE_MAGIC
            || reader.u32().map_err(SemanticTransitionError::Semantic)? != 1
            || reader.take(32).map_err(SemanticTransitionError::Semantic)?
                != publication_material_runtime()
        {
            return Err(publication_input_error(
                "replay provenance requires its exact pinned native runtime",
            ));
        }
        let predecessor = decode_publication_identity(&mut reader)?;
        let authority = PublicationMaterialRange::decode_from(&mut reader)?;
        let decision = PublicationMaterialRange::decode_from(&mut reader)?;
        reader.finish().map_err(SemanticTransitionError::Semantic)?;
        Ok(Self {
            predecessor,
            authority,
            decision,
        })
    }

    fn validate(&self, predecessor: &PublicationMaterial) -> Result<(), SemanticTransitionError> {
        let header = predecessor.bank.header;
        let original = SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        };
        if self.predecessor != original {
            return Err(publication_input_error(
                "replay provenance names another original native predecessor",
            ));
        }
        for (item, role) in [(&self.authority, 38), (&self.decision, 39)] {
            let saved = predecessor
                .ranges
                .iter()
                .find(|item| item.range.role == role && item.range.index == 0)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if (item.range.role, item.range.index) != (role, 0)
                || item.range.generation != 1
                || item.range.length_bytes != item.bytes.len() as u64
                || item.capacity != saved.capacity
                || item.capacity == 0
                || item.bytes.is_empty()
                || item.bytes.len() > item.capacity
                || item.range.logical_begin > item.range.logical_end
                || item.original_record_digest() != item.range.digest
            {
                return Err(publication_input_error(
                    "replay provenance changes an originally sealed authority record",
                ));
            }
            if role == 38
                && (item.bytes != saved.bytes
                    || publication_abi_bytes(&[item.range])
                        != publication_abi_bytes(&[saved.range]))
            {
                return Err(publication_input_error("replay provenance differs from the predecessor's whole original authority envelope"));
            }
        }
        Ok(())
    }
}

struct PublicationReplayEvidence {
    successor: SemanticPublishedIdentity,
    ranges: Vec<PublicationMaterialRange>,
}

impl PublicationReplayEvidence {
    fn range(&self, role: u64) -> Result<&PublicationMaterialRange, SemanticTransitionError> {
        self.ranges
            .iter()
            .find(|item| item.range.role == role && item.range.index == 0)
            .ok_or_else(|| publication_input_error("replay evidence omits a publication record"))
    }

    fn state_digest(&self) -> Result<Identity256, SemanticTransitionError> {
        let mut digest = self.successor.logical_digest;
        for item in &self.ranges {
            digest = publication_fold_digest(
                digest,
                item.range.role,
                item.range.index,
                item.logical_record_digest()?,
            );
        }
        Ok(digest)
    }

    fn encode(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        let mut bytes = REPLAY_EVIDENCE_MAGIC.to_vec();
        material_u32(&mut bytes, 1);
        bytes.extend_from_slice(&publication_material_runtime());
        encode_publication_identity(&mut bytes, self.successor);
        material_u32(
            &mut bytes,
            u32::try_from(self.ranges.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        );
        for item in &self.ranges {
            item.encode_into(&mut bytes)?;
        }
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        let mut reader = SemanticMaterialReader::new(bytes);
        if reader
            .take(REPLAY_EVIDENCE_MAGIC.len())
            .map_err(SemanticTransitionError::Semantic)?
            != REPLAY_EVIDENCE_MAGIC
            || reader.u32().map_err(SemanticTransitionError::Semantic)? != 1
            || reader.take(32).map_err(SemanticTransitionError::Semantic)?
                != publication_material_runtime()
        {
            return Err(publication_input_error(
                "replay evidence requires its exact pinned native runtime",
            ));
        }
        let successor = decode_publication_identity(&mut reader)?;
        if reader.u32().map_err(SemanticTransitionError::Semantic)?
            != REPLAY_EVIDENCE_ROLES.len() as u32
        {
            return Err(publication_input_error(
                "replay evidence requires all six native publication records",
            ));
        }
        let mut ranges = Vec::with_capacity(REPLAY_EVIDENCE_ROLES.len());
        for _ in REPLAY_EVIDENCE_ROLES {
            ranges.push(PublicationMaterialRange::decode_from(&mut reader)?);
        }
        reader.finish().map_err(SemanticTransitionError::Semantic)?;
        Ok(Self { successor, ranges })
    }

    fn validate(
        &self,
        predecessor: &PublicationMaterial,
    ) -> Result<SemanticTransitionKind, SemanticTransitionError> {
        let base = predecessor.bank.header;
        let successor_word = (base.publication_word >> 1)
            .checked_add(1)
            .filter(|&epoch| epoch <= u64::MAX >> 1)
            .map(|epoch| (epoch << 1) | ((base.publication_word & 1) ^ 1))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if self.successor.instance != base.instance
            || self.successor.word != successor_word
            || self.ranges.len() != REPLAY_EVIDENCE_ROLES.len()
        {
            return Err(publication_input_error(
                "replay evidence is not the original predecessor's immediate successor",
            ));
        }
        for (item, role) in self.ranges.iter().zip(REPLAY_EVIDENCE_ROLES) {
            let previous = predecessor
                .ranges
                .iter()
                .find(|item| item.range.role == role && item.range.index == 0)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if (item.range.role, item.range.index) != (role, 0)
                || item.range.generation != 1
                || item.range.length_bytes != item.bytes.len() as u64
                || item.capacity != previous.capacity
                || item.capacity == 0
                || item.bytes.len() > item.capacity
                || item.range.logical_begin > item.range.logical_end
                || item.original_record_digest() != item.range.digest
            {
                return Err(publication_input_error(
                    "replay evidence changes or omits an originally sealed publication record",
                ));
            }
        }
        let attempt_range = self.range(33)?;
        let attempt = attempt_range.attempt()?;
        let previous_attempt = predecessor
            .ranges
            .iter()
            .find(|item| item.range.role == 33 && item.range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if attempt.abi != 1
            || attempt.instance != self.successor.instance
            || attempt.base_word != base.publication_word
            || attempt.next_word != self.successor.word
            || attempt.logical_digest != self.successor.logical_digest
            || attempt.receipt_digest != attempt_range.logical_record_digest()?
            || attempt.previous_attempt_digest != previous_attempt.logical_record_digest()?
            || attempt.replay_head_digest != self.range(27)?.logical_record_digest()?
            || attempt.intent_head_digest != self.range(30)?.logical_record_digest()?
            || attempt.acknowledgement_head_digest != self.range(32)?.logical_record_digest()?
            || self.successor.state_digest != self.state_digest()?
        {
            return Err(publication_input_error(
                "replay evidence differs from its whole native attempt and state links",
            ));
        }
        let no_draw =
            attempt.action_receipts_digest == Identity256::from_bytes(Sha256::digest([]).into());
        match (base.terminal, no_draw) {
            (0, false) => Ok(SemanticTransitionKind::Proposal),
            (0, true) => Ok(SemanticTransitionKind::Recompute),
            (1, true) => Ok(SemanticTransitionKind::Drain),
            _ => Err(publication_input_error(
                "replay attempt cannot follow this predecessor's native terminal state",
            )),
        }
    }

    fn validate_observed(
        &self,
        predecessor: &PublicationMaterial,
        bank: &PublicationBank,
        coverage: &PublicationMaterialRange,
        codebook: &PublicationMaterialRange,
    ) -> Result<(), SemanticTransitionError> {
        let kind = self.validate(predecessor)?;
        let base = predecessor.bank.header;
        let header = bank.header;
        let draws = if kind == SemanticTransitionKind::Proposal {
            COMPONENT_COUNT
        } else {
            0
        };
        let proposal = base
            .proposal
            .checked_add(u64::from(draws != 0))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if header.abi != 1
            || header.instance != self.successor.instance
            || header.publication_word != self.successor.word
            || header.base_word != base.publication_word
            || header.sealed_epoch != self.successor.word >> 1
            || header.logical_digest != self.successor.logical_digest
            || header.state_digest != self.successor.state_digest
            || header.proposal != proposal
            || bank.state.proposal != base.proposal
            || bank.state.next_proposal != proposal
            || bank.state.blocks != draws as u64
            || bank.state.status != 0
            || (kind == SemanticTransitionKind::Drain && header.terminal != 2)
            || (kind == SemanticTransitionKind::Recompute && header.terminal != 0)
            || (kind == SemanticTransitionKind::Proposal && header.terminal > 1)
        {
            return Err(publication_input_error(
                "replay evidence does not describe the actual acquired native transition",
            ));
        }
        let attempt = self.range(33)?.attempt()?;
        if coverage.range.role != 14
            || coverage.bytes.len() != size_of::<CompletionCoverage>()
            || coverage.original_record_digest() != coverage.range.digest
            || codebook.original_record_digest() != codebook.range.digest
        {
            return Err(publication_input_error(
                "replay evidence sources differ from their actual native seals",
            ));
        }
        // SAFETY: the exact coverage extent was checked and all ABI fields are integers.
        let completion = unsafe {
            std::ptr::read_unaligned(coverage.bytes.as_ptr().cast::<CompletionCoverage>())
        };
        let coverage_digest = coverage.logical_record_digest()?;
        if completion.abi != 1
            || completion.instance != header.instance
            || completion.base_word != base.publication_word
            || completion.receipt_digest != coverage_digest
            || attempt.coverage_digest != coverage_digest
            || attempt.action_receipts_digest
                != publication_action_receipts_digest(codebook, &bank.receipts[..draws])?
            || attempt.semantic_receipts_digest
                != publication_semantic_receipts_digest(&bank.state.semantic_receipts)
        {
            return Err(publication_input_error(
                "whole native attempt differs from actual action, semantic or completion receipts",
            ));
        }
        Ok(())
    }
}

/// Owned cold replay inputs: complete predecessor material and the successor's
/// external native publication evidence. Hash consistency does not establish
/// historical execution, current rights, or permission to use a model.
pub struct SemanticReplayMaterial {
    predecessor: PublicationMaterial,
    evidence: PublicationReplayEvidence,
    kind: SemanticTransitionKind,
}

impl SemanticReplayMaterial {
    /// Check the original coordinates and seals before any runtime relocation.
    pub fn decode(
        predecessor_bytes: &[u8],
        evidence_bytes: &[u8],
        expected_predecessor_logical: Identity256,
        expected_successor_logical: Identity256,
    ) -> Result<Self, SemanticTransitionError> {
        let predecessor = PublicationMaterial::decode(predecessor_bytes)?;
        let evidence = PublicationReplayEvidence::decode(evidence_bytes)?;
        if predecessor.bank.header.logical_digest != expected_predecessor_logical
            || evidence.successor.logical_digest != expected_successor_logical
        {
            return Err(publication_input_error(
                "replay native materials differ from the execution's logical identities",
            ));
        }
        let kind = evidence.validate(&predecessor)?;
        Ok(Self {
            predecessor,
            evidence,
            kind,
        })
    }

    pub fn predecessor_identity(&self) -> SemanticPublishedIdentity {
        let header = self.predecessor.bank.header;
        SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        }
    }

    pub fn successor_identity(&self) -> SemanticPublishedIdentity {
        self.evidence.successor
    }

    pub fn transition_kind(&self) -> SemanticTransitionKind {
        self.kind
    }

    pub fn training_view_origin(
        &self,
    ) -> Result<SemanticTrainingViewOrigin, SemanticTransitionError> {
        let header = self.predecessor.bank.header;
        Ok(SemanticTrainingViewOrigin {
            transition: self.kind,
            predecessor: self.predecessor_identity(),
            successor: self.successor_identity(),
            invocation: header.rng_binding()?,
            model_geometry_digest: header.model_geometry_digest,
            model_numerical_digest: header.model_numerical_digest,
        })
    }

    /// Original model-owned contribution, after complete material validation.
    /// Its consumer must validate the supported model format before restoring
    /// numerical state; these bytes are not current settings or an execution grant.
    pub fn model_numerical_mode(&self) -> Result<&[u8], SemanticTransitionError> {
        self.predecessor.model_numerical_mode()
    }

    /// Return the archived continuation decision after checking its native
    /// provenance root against the original complete predecessor. The caller
    /// must separately bind this root through the canonical reconstruction
    /// closure and check current rights; these bytes are not current authority.
    pub fn verify_provenance(&self, bytes: &[u8]) -> Result<Vec<u8>, SemanticTransitionError> {
        let provenance = PublicationReplayProvenance::decode(bytes)?;
        provenance.validate(&self.predecessor)?;
        Ok(provenance.decision.bytes)
    }
}

/// Complete native logical-state material inside an execution's material
/// closure. This is not an episode, a use grant, or proof of past publication.
struct PublicationMaterial {
    bank: PublicationBank,
    contract: PublicationContract,
    role_counts: [u64; 55],
    terminals: Vec<u64>,
    layouts: BTreeMap<(u64, u64), SemanticTensorLayout>,
    model_memory: ModelMemoryGeometry,
    model_allocations: Vec<Vec<u8>>,
    ranges: Vec<PublicationMaterialRange>,
    graph: SemanticRootMaterial,
}

fn relocate_publication_codebooks(
    saved: &[u8],
    actual: &[u64],
) -> Result<Vec<u8>, SemanticTransitionError> {
    if actual.len() < 25
        || saved.len()
            != actual
                .len()
                .checked_mul(8)
                .ok_or(SemanticTransitionError::GenerationExhausted)?
    {
        return Err(publication_input_error(
            "restored codebook has another actual extent",
        ));
    }
    for (index, (saved, &actual)) in saved.chunks_exact(8).zip(actual).enumerate() {
        if !matches!(index, 14..=16 | 21..=24)
            && u64::from_le_bytes(saved.try_into().unwrap()) != actual
        {
            return Err(publication_input_error(
                "restored codebook changes its admitted logical content or original base",
            ));
        }
    }
    Ok(publication_abi_bytes(actual))
}

fn publication_material_runtime() -> [u8; 32] {
    static IDENTITY: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    *IDENTITY.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"xlog.publication.material-runtime.v1\0");
        hash.update(include_bytes!("semantic_transition.rs"));
        hash.update(include_bytes!("../kernels/semantic_transition.cu"));
        let policy_identity: [u8; 32] =
            include!(concat!(env!("OUT_DIR"), "/semantic_policy_identity.rs"));
        hash.update(policy_identity);
        #[cfg(feature = "semantic-policy")]
        hash.update(include_bytes!("../kernels/semantic_policy_binding.cuh"));
        hash.update(include_bytes!("semantic_hypergraph.rs"));
        hash.update(include_bytes!("../kernels/semantic_hypergraph.cu"));
        hash.finalize().into()
    })
}

const RUNTIME_CONTRACT_MAGIC: &[u8] = b"XLOG-PUBLICATION-RUNTIME\0";

/// Portable projection of the actual cold publication envelope. Mutable
/// extents, active rows, bank/instance identities and executable addresses do
/// not belong here. The original model contribution is retained in full.
fn runtime_contract_bytes(
    contract: PublicationContract,
    counts: &[u64; 55],
    terminals: &[u64],
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    capacities: &[(u64, u64, usize)],
    model_mode: &[u8],
    model_memory: &ModelMemoryGeometry,
) -> Result<Vec<u8>, SemanticTransitionError> {
    if model_mode.is_empty() || model_mode.len() > 16 * 1024 * 1024 {
        return Err(publication_input_error(
            "runtime contract requires the complete bounded model numerical mode",
        ));
    }
    let mut bytes = RUNTIME_CONTRACT_MAGIC.to_vec();
    material_u32(&mut bytes, 1);
    bytes.extend_from_slice(&publication_material_runtime());
    material_bytes(&mut bytes, model_mode).map_err(SemanticTransitionError::Semantic)?;
    for field in [
        contract.abi,
        contract.range_capacity,
        contract.prefix_capacity,
        contract.window_capacity,
        contract.feedback_capacity,
        contract.max_position,
        contract.pad_token,
        contract.terminal_token_count,
        contract.final_intent_payload_bytes,
        contract.model_generation,
        contract.policy_generation,
        contract.authority_generation,
        contract.role_count,
    ] {
        material_u64(&mut bytes, field);
    }
    let model = contract.model_contract_layout;
    for field in [
        model.schema_begin,
        model.schema_bytes,
        model.schema_digest_offset,
        model.generation_offset,
        model.numerical_digest_offset,
        model.identity_offset,
    ] {
        material_u64(&mut bytes, field);
    }
    for identity in [
        contract.task_identity,
        contract.topology_identity,
        contract.table_identity,
    ] {
        bytes.extend_from_slice(identity.as_bytes());
    }
    for &count in counts {
        material_u64(&mut bytes, count);
    }
    for &token in terminals {
        material_u64(&mut bytes, token);
    }
    material_u64(&mut bytes, layouts.len() as u64);
    for layout in layouts.values() {
        for field in [
            layout.role,
            layout.index,
            layout.element_bytes,
            layout.scalar_type,
            layout.rank,
            layout.logical_axis,
        ]
        .into_iter()
        .chain(layout.dimensions)
        .chain(layout.strides_bytes)
        {
            material_u64(&mut bytes, field);
        }
    }
    material_u64(&mut bytes, capacities.len() as u64);
    for &(role, index, capacity) in capacities {
        for field in [role, index, capacity as u64] {
            material_u64(&mut bytes, field);
        }
    }
    model_memory.encode_into(&mut bytes)?;
    Ok(bytes)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the authenticated model mode is defined by independent publication owners"
)]
fn verified_runtime_model_mode<'a>(
    bytes: &'a [u8],
    contract: PublicationContract,
    counts: &[u64; 55],
    terminals: &[u64],
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    capacities: &[(u64, u64, usize)],
    table: &PublicationMaterialRange,
    model_memory: &ModelMemoryGeometry,
) -> Result<&'a [u8], SemanticTransitionError> {
    // The host layout map is not evidence of published device bytes: public
    // record aliases retain the actual mutable allocation. Authenticate that
    // parent's entire original table, including its active-row state, first.
    if (table.range.role, table.range.index, table.range.generation) != (55, 0, 1)
        || table.range.length_bytes != table.bytes.len() as u64
        || table.bytes.len() > table.capacity
        || table.original_record_digest() != table.range.digest
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let (_, actual_layouts, _) = decode_tensor_table(&table.bytes)?;
    if actual_layouts != layouts.values().copied().collect::<Vec<_>>() {
        return Err(publication_input_error(
            "publication material layout differs from its sealed table",
        ));
    }
    let mut reader = SemanticMaterialReader::new(bytes);
    if reader
        .take(RUNTIME_CONTRACT_MAGIC.len())
        .map_err(SemanticTransitionError::Semantic)?
        != RUNTIME_CONTRACT_MAGIC
        || reader.u32().map_err(SemanticTransitionError::Semantic)? != 1
        || reader.take(32).map_err(SemanticTransitionError::Semantic)?
            != publication_material_runtime()
    {
        return Err(publication_input_error(
            "unsupported native runtime contract format or identity",
        ));
    }
    let mode = reader.bytes().map_err(SemanticTransitionError::Semantic)?;
    // Comparing the complete explicit encoding validates the suffix, including
    // its exact length and all cold fields. Never overwrite the saved record.
    if runtime_contract_bytes(
        contract,
        counts,
        terminals,
        layouts,
        capacities,
        mode,
        model_memory,
    )? != bytes
    {
        return Err(publication_input_error(
            "runtime contract differs from actual cold publication state",
        ));
    }
    Ok(mode)
}

fn initial_runtime_contract_record(
    contract: PublicationContract,
    counts: &[u64; 55],
    terminals: &[u64],
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    mut capacities: Vec<(u64, u64, usize)>,
    model_mode: &[u8],
    model_memory: &ModelMemoryGeometry,
) -> Result<SemanticStateRecord, SemanticTransitionError> {
    if capacities.iter().any(|&(role, _, _)| role == 43) || counts[42] != 1 {
        return Err(publication_input_error(
            "runtime contract is generated exactly once by the native owner",
        ));
    }
    capacities.push((43, 0, 0));
    capacities.sort_by_key(|&(role, index, _)| (role, index));
    if capacities.len() != validate_publication_counts(counts)?
        || contract.range_capacity != capacities.len() as u64
    {
        return Err(publication_input_error(
            "runtime capacity roster differs from the actual publication directory",
        ));
    }
    // Only the value of a fixed-width capacity cell changes between passes.
    let capacity = runtime_contract_bytes(
        contract,
        counts,
        terminals,
        layouts,
        &capacities,
        model_mode,
        model_memory,
    )?
    .len();
    capacities.iter_mut().find(|entry| entry.0 == 43).unwrap().2 = capacity;
    let bytes = runtime_contract_bytes(
        contract,
        counts,
        terminals,
        layouts,
        &capacities,
        model_mode,
        model_memory,
    )?;
    debug_assert_eq!(bytes.len(), capacity);
    Ok(SemanticStateRecord {
        role: SemanticStateRole::RuntimeContract,
        index: 0,
        capacity_bytes: capacity,
        bytes,
    })
}

// SAFETY: callers below name only fixed publication ABI records containing
// integer/IEEE scalar fields with all bit patterns valid. The exact ABI size is
// checked before reading; no pointer-bearing control records enter this codec.
unsafe fn publication_material_record<T: Copy>(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<T, SemanticTransitionError> {
    let bytes = reader.bytes().map_err(SemanticTransitionError::Semantic)?;
    if bytes.len() != size_of::<T>() {
        return Err(publication_input_error(
            "publication material record has another ABI extent",
        ));
    }
    Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) })
}

impl PublicationMaterial {
    fn model_numerical_mode(&self) -> Result<&[u8], SemanticTransitionError> {
        let record = self
            .ranges
            .iter()
            .find(|item| item.range.role == 43 && item.range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let table = self
            .ranges
            .iter()
            .find(|item| item.range.role == 55 && item.range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacities = self
            .ranges
            .iter()
            .map(|item| (item.range.role, item.range.index, item.capacity))
            .collect::<Vec<_>>();
        verified_runtime_model_mode(
            &record.bytes,
            self.contract,
            &self.role_counts,
            &self.terminals,
            &self.layouts,
            &capacities,
            table,
            &self.model_memory,
        )
    }

    fn encode(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        self.validate()?;
        let mut bytes = b"XLOG-PUBLICATION-MATERIAL\0".to_vec();
        material_u32(&mut bytes, 1);
        bytes.extend_from_slice(&publication_material_runtime());
        material_bytes(
            &mut bytes,
            &self
                .graph
                .encode()
                .map_err(SemanticTransitionError::Semantic)?,
        )
        .map_err(SemanticTransitionError::Semantic)?;
        for record in [
            publication_abi_bytes(&[self.bank]),
            publication_abi_bytes(&[self.contract]),
            publication_abi_bytes(&self.role_counts),
            publication_abi_bytes(&self.terminals),
        ] {
            material_bytes(&mut bytes, &record).map_err(SemanticTransitionError::Semantic)?;
        }
        material_u32(
            &mut bytes,
            u32::try_from(self.layouts.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        );
        for layout in self.layouts.values() {
            material_bytes(&mut bytes, &publication_abi_bytes(&[*layout]))
                .map_err(SemanticTransitionError::Semantic)?;
        }
        self.model_memory.encode_into(&mut bytes)?;
        for allocation in &self.model_allocations {
            material_bytes(&mut bytes, allocation).map_err(SemanticTransitionError::Semantic)?;
        }
        material_u32(
            &mut bytes,
            u32::try_from(self.ranges.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        );
        for range in &self.ranges {
            if matches!(range.range.role, 18..=25) {
                // Model bytes occur once per full backing allocation, including
                // padding and cells outside any individual typed view.
                material_bytes(&mut bytes, &publication_abi_bytes(&[range.range]))
                    .map_err(SemanticTransitionError::Semantic)?;
                material_u64(&mut bytes, range.capacity as u64);
            } else {
                range.encode_into(&mut bytes)?;
            }
        }
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        let mut reader = SemanticMaterialReader::new(bytes);
        if reader.take(26).map_err(SemanticTransitionError::Semantic)?
            != b"XLOG-PUBLICATION-MATERIAL\0"
            || reader.u32().map_err(SemanticTransitionError::Semantic)? != 1
            || reader.take(32).map_err(SemanticTransitionError::Semantic)?
                != publication_material_runtime()
        {
            return Err(publication_input_error(
                "publication material requires its exact pinned native runtime",
            ));
        }
        let graph_bytes = reader.bytes().map_err(SemanticTransitionError::Semantic)?;
        let limit = u32::try_from(graph_bytes.len()).unwrap_or(u32::MAX);
        let graph = SemanticRootMaterial::decode(
            graph_bytes,
            SemanticAdmissionLimits {
                max_records: limit,
                max_terms: limit,
                max_references: limit,
                max_utf8_bytes: graph_bytes.len(),
            },
        )
        .map_err(SemanticTransitionError::Semantic)?;
        // SAFETY: these fixed ABI records contain only all-bit-valid scalars.
        let bank = unsafe { publication_material_record::<PublicationBank>(&mut reader)? };
        let contract = unsafe { publication_material_record::<PublicationContract>(&mut reader)? };
        let role_counts = unsafe { publication_material_record::<[u64; 55]>(&mut reader)? };
        let terminal_bytes = reader.bytes().map_err(SemanticTransitionError::Semantic)?;
        if terminal_bytes.len() % 8 != 0 {
            return Err(publication_input_error(
                "publication terminal material is not integral",
            ));
        }
        let terminals = terminal_bytes
            .chunks_exact(8)
            .map(|item| u64::from_le_bytes(item.try_into().unwrap()))
            .collect();
        let count = reader
            .count(4 + size_of::<SemanticTensorLayout>())
            .map_err(SemanticTransitionError::Semantic)?;
        let mut layouts = BTreeMap::new();
        let mut previous = None;
        for _ in 0..count {
            // SAFETY: the tensor-layout ABI contains only integer fields.
            let layout =
                unsafe { publication_material_record::<SemanticTensorLayout>(&mut reader)? };
            let key = (layout.role, layout.index);
            if previous.is_some_and(|previous| previous >= key) {
                return Err(publication_input_error(
                    "publication material tensor layouts are not in canonical order",
                ));
            }
            previous = Some(key);
            layouts.insert(key, layout);
        }
        let model_memory = ModelMemoryGeometry::decode_from(&mut reader)?;
        model_memory.validate(&layouts)?;
        let mut model_allocations = Vec::new();
        for &extent in &model_memory.allocation_bytes {
            let allocation = reader.bytes().map_err(SemanticTransitionError::Semantic)?;
            if allocation.len() as u64 != extent {
                return Err(publication_input_error(
                    "publication material truncates a complete model backing allocation",
                ));
            }
            model_allocations.push(allocation.to_vec());
        }
        let count = reader
            .count(12 + size_of::<PublicationRange>())
            .map_err(SemanticTransitionError::Semantic)?;
        let mut ranges = Vec::with_capacity(count);
        for _ in 0..count {
            // SAFETY: the range ABI contains only integer/digest fields.
            let range = unsafe { publication_material_record::<PublicationRange>(&mut reader)? };
            let capacity =
                usize::try_from(reader.u64().map_err(SemanticTransitionError::Semantic)?).map_err(
                    |_| publication_input_error("publication material range capacity overflows"),
                )?;
            let payload = if matches!(range.role, 18..=25) {
                let (allocation, offset) = model_memory.location(range.role, range.index)?;
                let end = usize::try_from(range.length_bytes)
                    .ok()
                    .and_then(|bytes| offset.checked_add(bytes))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                model_allocations[allocation]
                    .get(offset..end)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
            } else {
                reader.bytes().map_err(SemanticTransitionError::Semantic)?
            };
            ranges.push(PublicationMaterialRange {
                range,
                capacity,
                bytes: payload.to_vec(),
            });
        }
        reader.finish().map_err(SemanticTransitionError::Semantic)?;
        let material = Self {
            bank,
            contract,
            role_counts,
            terminals,
            layouts,
            model_memory,
            model_allocations,
            ranges,
            graph,
        };
        material.validate()?;
        for range in &material.ranges {
            if !is_tensor_role(range.range.role)
                && range.original_record_digest() != range.range.digest
            {
                return Err(publication_input_error(
                    "publication material changes an originally sealed native record",
                ));
            }
        }
        if material.original_descriptor_digest() != material.bank.header.descriptor_digest {
            return Err(publication_input_error(
                "publication material differs from its original descriptor seal",
            ));
        }
        Ok(material)
    }

    // Match publication_descriptor_digest before any owner/slot relocation.
    // This unkeyed consistency seal is not proof of historical execution. The
    // native restore still hashes actual payloads, and the trusted importer
    // separately verifies the complete execution evidence and current rights.
    fn original_descriptor_digest(&self) -> Identity256 {
        let mut bank = self.bank;
        bank.header.abi = 1;
        bank.header.descriptor_digest = Identity256::default();
        let mut digest =
            Identity256::from_bytes(Sha256::digest(publication_abi_bytes(&[bank])).into());
        for material in &self.ranges {
            let range = material.range;
            let content =
                Identity256::from_bytes(Sha256::digest(publication_abi_bytes(&[range])).into());
            digest = publication_fold_digest(digest, range.role, range.index, content);
        }
        digest
    }

    fn validate(&self) -> Result<(), SemanticTransitionError> {
        self.model_memory.validate(&self.layouts)?;
        let model = self
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (44, 0))
            .ok_or_else(|| {
                publication_input_error("publication material lacks its complete model contract")
            })?;
        self.contract
            .model_contract_layout
            .validate(model.bytes.len() as u64)?;
        if self.bank.header.model_geometry_digest != self.model_memory.digest(&self.layouts)? {
            return Err(publication_input_error(
                "publication model geometry differs from its original sealed map",
            ));
        }
        if self.model_allocations.len() != self.model_memory.allocation_bytes.len()
            || self
                .model_allocations
                .iter()
                .zip(&self.model_memory.allocation_bytes)
                .any(|(bytes, &extent)| bytes.len() as u64 != extent)
        {
            return Err(publication_input_error(
                "publication material lacks complete model backing bytes",
            ));
        }
        let mut model_slots = BTreeMap::new();
        let mut slot_allocations = BTreeMap::new();
        let backing_digests = self
            .model_allocations
            .iter()
            .map(|bytes| Identity256::from_bytes(Sha256::digest(bytes).into()))
            .collect::<Vec<_>>();
        for item in self
            .ranges
            .iter()
            .filter(|item| matches!(item.range.role, 18..=25))
        {
            let (allocation, offset) = self
                .model_memory
                .location(item.range.role, item.range.index)?;
            let bytes = &self.model_allocations[allocation];
            let end = offset
                .checked_add(item.bytes.len())
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if item.range.offset_bytes != offset as u64
                || item.capacity != bytes.len() - offset
                || bytes.get(offset..end) != Some(item.bytes.as_slice())
                || item.range.backing_digest != backing_digests[allocation]
                || model_slots
                    .insert(allocation, item.range.storage_slot)
                    .is_some_and(|slot| slot != item.range.storage_slot)
                || slot_allocations
                    .insert(item.range.storage_slot, allocation)
                    .is_some_and(|prior| prior != allocation)
            {
                return Err(publication_input_error(
                    "publication material changes model backing or view relationships",
                ));
            }
        }
        if self.ranges.iter().any(|item| {
            !matches!(item.range.role, 18..=25)
                && (slot_allocations.contains_key(&item.range.storage_slot)
                    || item.range.backing_digest != Identity256::default())
        }) {
            return Err(publication_input_error(
                "model backing aliases native control or cache storage",
            ));
        }
        let header = self.bank.header;
        let contract = self.contract;
        if !cfg!(target_endian = "little")
            || header.abi != 1
            || contract.abi != 1
            || header.sealed_epoch != header.publication_word >> 1
            || contract.terminal_tokens != 0
            || contract.role_counts != 0
            || contract.semantic_owner != 0
            || contract.window_capacity != 32
            || contract.role_count != 55
            || contract.terminal_token_count != self.terminals.len() as u64
            || contract.model_generation == 0
            || header.model_generation < contract.model_generation
            || u32::try_from(header.model_generation).is_err()
            || header.authority_generation != contract.authority_generation
            || header.semantic_digest.as_bytes() != &self.graph.digest
            || header.semantic_extents != self.graph.extents.map(u64::from)
            || validate_publication_counts(&self.role_counts)? != self.ranges.len()
            || header.range_count != self.ranges.len() as u64
            || contract.range_capacity != header.range_count
        {
            return Err(publication_input_error(
                "publication material header or graph differs from its complete owned state",
            ));
        }
        let mut ranges = self.ranges.iter();
        for (ordinal, &count) in self.role_counts.iter().enumerate() {
            let role = ordinal as u64 + 1;
            for index in 0..count {
                let item = ranges
                    .next()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if (item.range.role, item.range.index) != (role, index)
                    || item.range.generation != 1
                    || item.range.length_bytes != item.bytes.len() as u64
                    || (item.capacity == 0
                        && !self
                            .layouts
                            .get(&(role, index))
                            .is_some_and(empty_model_tensor_layout))
                    || item.bytes.len() > item.capacity
                    || item.range.logical_begin > item.range.logical_end
                {
                    return Err(publication_input_error(
                        "publication material omits, reorders or changes a reachable range",
                    ));
                }
                if is_tensor_role(role) {
                    let layout = self.layouts.get(&(role, index)).ok_or_else(|| {
                        publication_input_error("publication material omits a tensor layout")
                    })?;
                    if tensor_layout_bytes(layout)? > item.capacity
                        || (empty_model_tensor_layout(layout)
                            && (item.range.logical_begin != 0 || item.range.logical_end != 0))
                    {
                        return Err(publication_input_error(
                            "publication material tensor exceeds its actual capacity",
                        ));
                    }
                } else if self.layouts.contains_key(&(role, index)) {
                    return Err(publication_input_error(
                        "publication material gives a tensor layout to a native record",
                    ));
                }
            }
        }
        if self.layouts.len()
            != self
                .ranges
                .iter()
                .filter(|item| is_tensor_role(item.range.role))
                .count()
        {
            return Err(publication_input_error(
                "publication material has an undeclared tensor layout",
            ));
        }
        self.model_numerical_mode()?;
        Ok(())
    }
}

struct PublicationStorage {
    control: TrackedCudaSlice<PublicationControl>,
    banks: [TrackedCudaSlice<PublicationBank>; 2],
    directories: [TrackedCudaSlice<PublicationRange>; 2],
    storage: TrackedCudaSlice<PublicationStorageEntry>,
    contract: TrackedCudaSlice<PublicationContract>,
    role_counts: TrackedCudaSlice<PublicationRoleCount>,
    terminals: TrackedCudaSlice<u64>,
    continuation: TrackedCudaSlice<PendingContinuation>,
    continuation_directory: TrackedCudaSlice<PublicationRange>,
    allocations: Vec<TrackedCudaSlice<u8>>,
    model_memory: ModelMemoryGeometry,
    model_slots: Vec<[usize; 2]>,
    bank_templates: [Vec<PublicationRange>; 2],
    continuation_templates: Vec<PublicationRange>,
    layouts: BTreeMap<(u64, u64), SemanticTensorLayout>,
    contract_value: PublicationContract,
    instance: Identity256,
}

fn allocate_publication<T: DeviceRepr>(
    provider: &CudaKernelProvider,
    count: usize,
) -> Result<TrackedCudaSlice<T>, SemanticTransitionError> {
    provider
        .memory()
        .alloc::<T>(count)
        .map_err(|error| runtime_error("publication allocation", error))
}

fn upload_publication<T: DeviceRepr>(
    provider: &CudaKernelProvider,
    values: &[T],
    destination: &TrackedCudaSlice<T>,
) -> Result<(), SemanticTransitionError> {
    let mut destination = destination.view();
    provider
        .htod_launch_metadata_sync_copy_into(values, &mut destination)
        .map_err(|error| runtime_error("publication cold metadata upload", error))
}

impl PublicationStorage {
    #[expect(
        clippy::too_many_arguments,
        reason = "the guard records each authenticated publication coordinate explicitly"
    )]
    fn enqueue_content_guard(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        reader: DeviceMemoryView<PublicationLease>,
        execute: &CudaFunction,
        role: u64,
        index: u64,
        layout: SemanticTensorLayout,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.read(&reader);
        recorder.read(&self.control);
        recorder.read(&self.storage);
        recorder.read(&self.contract);
        for bank in &self.banks {
            recorder.read(bank);
        }
        for directory in &self.directories {
            recorder.read(directory);
        }
        for allocation in &self.allocations {
            recorder.read(allocation);
        }
        let arguments = (
            self.control.device_ptr_value(),
            role,
            index,
            layout,
            u64::from(is_tensor_role(role)),
            *reader.device_ptr(),
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: the original device lease selects its actual directory;
            // every selectable bank, range, seal and allocation remains owned.
            unsafe {
                execute.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "cold allocation consumes the complete independently owned publication contract"
    )]
    fn allocate(
        provider: &CudaKernelProvider,
        plans: &[PublicationAllocationPlan],
        layouts: BTreeMap<(u64, u64), SemanticTensorLayout>,
        model_memory: ModelMemoryGeometry,
        model_payloads: &[PublicationPayload],
        mut contract: PublicationContract,
        terminal_tokens: &[u64],
        instance: Identity256,
    ) -> Result<(Self, Vec<(usize, PublicationPayload)>), SemanticTransitionError> {
        let mut allocations = Vec::new();
        let mut bank_templates = [Vec::new(), Vec::new()];
        let mut uploads = Vec::new();
        model_memory.validate(&layouts)?;
        if model_payloads.len() != model_memory.allocation_bytes.len() {
            return Err(publication_input_error(
                "model backing upload roster differs from its complete memory map",
            ));
        }
        let mut model_slots = Vec::new();
        for (&bytes, payload) in model_memory.allocation_bytes.iter().zip(model_payloads) {
            let mut slots = [0; 2];
            for slot in &mut slots {
                *slot = allocations.len();
                allocations.push(allocate_publication::<u8>(provider, bytes as usize)?);
                uploads.push((*slot, payload.clone()));
            }
            model_slots.push(slots);
        }
        for plan in plans {
            if (plan.capacity == 0
                && !layouts
                    .get(&(plan.role, plan.index))
                    .is_some_and(empty_model_tensor_layout))
                || plan.length > plan.capacity
                || (layouts
                    .get(&(plan.role, plan.index))
                    .is_some_and(empty_model_tensor_layout)
                    && (plan.logical_begin != 0 || plan.logical_end != 0))
            {
                return Err(publication_input_error(
                    "owned publication allocation has invalid capacity",
                ));
            }
            let mut slot = 0;
            let model_view = if matches!(plan.role, 18..=25) {
                Some(model_memory.location(plan.role, plan.index)?)
            } else {
                None
            };
            for (bank, directory) in bank_templates.iter_mut().enumerate() {
                if let Some((allocation, offset)) = model_view {
                    slot = model_slots[allocation][bank];
                    if plan.capacity != allocations[slot].len() - offset {
                        return Err(publication_input_error(
                            "model range capacity differs from its actual backing view",
                        ));
                    }
                } else if bank == 0 || publication_mutable_role(plan.role) {
                    slot = allocations.len();
                    allocations.push(allocate_publication::<u8>(provider, plan.capacity)?);
                    uploads.push((slot, plan.payload.clone()));
                }
                directory.push(PublicationRange {
                    role: plan.role,
                    index: plan.index,
                    storage_slot: slot as u64,
                    generation: 1,
                    offset_bytes: model_view.map_or(0, |(_, offset)| offset as u64),
                    length_bytes: plan.length as u64,
                    logical_begin: plan.logical_begin,
                    logical_end: plan.logical_end,
                    digest: Identity256::default(),
                    backing_digest: Identity256::default(),
                });
            }
        }
        let pending_layouts = layouts
            .values()
            .filter(|layout| continuation_role(layout.role))
            .map(|layout| continuation_allocation_layout(*layout))
            .collect::<Result<Vec<_>, _>>()?;
        let pending_table = tensor_table_bytes(&pending_layouts, true, &[])?;
        let mut continuation_templates = Vec::new();
        for plan in plans.iter().filter(|plan| continuation_role(plan.role)) {
            let capacity = if plan.role == 1 {
                32 * size_of::<SemanticTextSlot>()
            } else if plan.role == 39 {
                plan.capacity
            } else if plan.role == 55 {
                tensor_table_capacity(pending_layouts.len(), 32 + contract.feedback_capacity)?
            } else {
                tensor_layout_bytes(&continuation_allocation_layout(
                    layouts[&(plan.role, plan.index)],
                )?)?
            };
            if capacity == 0 {
                return Err(publication_input_error(
                    "continuation destination has no actual capacity",
                ));
            }
            let slot = allocations.len();
            allocations.push(allocate_publication::<u8>(provider, capacity)?);
            let length = if plan.role == 55 {
                uploads.push((slot, PublicationPayload::Metadata(pending_table.clone())));
                pending_table.len()
            } else if matches!(plan.role, 1 | 39) {
                0
            } else {
                capacity
            };
            continuation_templates.push(PublicationRange {
                role: plan.role,
                index: plan.index,
                storage_slot: slot as u64,
                generation: 1,
                length_bytes: length as u64,
                ..PublicationRange::default()
            });
        }
        let terminals = allocate_publication::<u64>(provider, terminal_tokens.len())?;
        let role_counts = allocate_publication::<PublicationRoleCount>(provider, 55)?;
        contract.terminal_tokens = terminals.device_ptr_value();
        contract.role_counts = role_counts.device_ptr_value();
        let contract_value = contract;
        Ok((
            Self {
                control: allocate_publication(provider, 1)?,
                banks: [
                    allocate_publication(provider, 1)?,
                    allocate_publication(provider, 1)?,
                ],
                directories: [
                    allocate_publication(provider, plans.len())?,
                    allocate_publication(provider, plans.len())?,
                ],
                storage: allocate_publication(provider, allocations.len())?,
                contract: allocate_publication(provider, 1)?,
                role_counts,
                terminals,
                continuation: allocate_publication(provider, 1)?,
                continuation_directory: allocate_publication(
                    provider,
                    continuation_templates.len(),
                )?,
                allocations,
                model_memory,
                model_slots,
                bank_templates,
                continuation_templates,
                layouts,
                contract_value,
                instance,
            },
            uploads,
        ))
    }

    fn record(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.control);
        for bank in &self.banks {
            recorder.read_write(bank);
        }
        for directory in &self.directories {
            recorder.read_write(directory);
        }
        recorder.read(&self.storage);
        recorder.read(&self.contract);
        recorder.read(&self.role_counts);
        recorder.read(&self.terminals);
        recorder.read_write(&self.continuation);
        recorder.read_write(&self.continuation_directory);
        for allocation in &self.allocations {
            recorder.read_write(allocation);
        }
    }
}

struct PreparedContinuation {
    storage: Arc<PublicationStorage>,
    reader: DeviceMemoryView<PublicationLease>,
    text_binding: Arc<TextBindingStorage>,
    sources: Vec<PreparedSemanticTensor>,
    copies: Vec<(u64, u64, usize)>,
    inputs: ContinuationInputs,
    training_selection: Option<DeviceMemoryView<SemanticTrainingViewSelection>>,
    execute: CudaFunction,
}

type ContinuationPayloadPlan = (
    Vec<(usize, PublicationPayload)>,
    BTreeMap<(u64, u64), SemanticTensorLayout>,
);

fn prepare_continuation_payloads(
    storage: &PublicationStorage,
    original: &[PreparedSemanticTensor],
    authority_decisions: &[u8],
    kind: SemanticTransitionKind,
) -> Result<ContinuationPayloadPlan, SemanticTransitionError> {
    let mut pending_layouts = BTreeMap::new();
    let mut inputs = BTreeMap::new();
    for tensor in original {
        let key = (tensor.layout.role, tensor.layout.index);
        let model = storage.layouts.get(&key).ok_or_else(|| {
            publication_input_error("continuation tensor has no cold model owner")
        })?;
        if !continuation_role(key.0) || !is_tensor_role(key.0) || inputs.contains_key(&key) {
            return Err(publication_input_error(
                "continuation tensor type or role differs from the model",
            ));
        }
        let destination = continuation_tensor_layout_for_kind(
            &tensor.layout,
            tensor.logical_begin,
            tensor.logical_end,
            model,
            kind,
        )?;
        pending_layouts.insert(key, destination);
        inputs.insert(key, tensor.clone());
    }
    let mut uploads = Vec::new();
    for range in &storage.continuation_templates {
        let key = (range.role, range.index);
        if matches!(range.role, 1 | 55) {
            continue;
        }
        let payload = if range.role == 39 {
            PublicationPayload::Metadata(authority_decisions.to_vec())
        } else {
            PublicationPayload::Tensor(inputs.remove(&key).ok_or_else(|| {
                publication_input_error("continuation omits an actual required forward tensor")
            })?)
        };
        let length = match &payload {
            PublicationPayload::Metadata(bytes) => bytes.len(),
            PublicationPayload::Tensor(_) => tensor_layout_bytes(&pending_layouts[&key])?,
            PublicationPayload::Uncomputed => {
                unreachable!("continuation accepts original computed tensors only")
            }
        };
        if length > storage.allocations[range.storage_slot as usize].len() {
            return Err(publication_input_error(
                "actual continuation exceeds its owned fixed destination capacity",
            ));
        }
        uploads.push((range.storage_slot as usize, payload));
    }
    if !inputs.is_empty() {
        return Err(publication_input_error(
            "continuation contains an undeclared tensor owner",
        ));
    }
    Ok((uploads, pending_layouts))
}

impl PreparedContinuation {
    // Cold geometry and owner preparation. Counts remain original device
    // pointers in inputs; this never snapshots their values or changes layout.
    #[expect(
        clippy::too_many_arguments,
        reason = "continuation preparation binds all retained producer and publication owners"
    )]
    fn prepare(
        provider: &CudaKernelProvider,
        storage: Arc<PublicationStorage>,
        reader: DeviceMemoryView<PublicationLease>,
        text_binding: Arc<TextBindingStorage>,
        inputs: ContinuationInputs,
        training_selection: Option<DeviceMemoryView<SemanticTrainingViewSelection>>,
        uploads: &[(usize, PublicationPayload)],
        layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    ) -> Result<Self, SemanticTransitionError> {
        let mut sources = Vec::new();
        let mut copies = Vec::new();
        for (slot, payload) in uploads {
            let PublicationPayload::Tensor(tensor) = payload else {
                continue;
            };
            let destination = storage
                .allocations
                .get(*slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if let Some(source) = &tensor.source {
                let layout = layouts
                    .get(&(tensor.layout.role, tensor.layout.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                for (source_offset, destination_offset, bytes) in
                    tensor_copy_plan(&tensor.layout, layout)?
                {
                    if source_offset
                        .checked_add(bytes as u64)
                        .is_none_or(|end| end > source.len() as u64)
                        || destination_offset
                            .checked_add(bytes as u64)
                            .is_none_or(|end| end > destination.len() as u64)
                    {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    let source = source
                        .device_ptr()
                        .checked_add(source_offset)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let destination = destination
                        .device_ptr_value()
                        .checked_add(destination_offset)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    copies.push((source, destination, bytes));
                }
            }
            // Empty tensors retain their real producer token as well.
            sources.push(tensor.clone());
        }
        let execute = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_prepare_continuation",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "continuation preparation unavailable")
            })?;
        Ok(Self {
            storage,
            reader,
            text_binding,
            sources,
            copies,
            inputs,
            training_selection,
            execute,
        })
    }

    // Enqueue only: original source owners, fixed copy spans and the kernel
    // are already prepared. Authority upload and stream joins belong outside.
    fn enqueue(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        self.storage.record(&mut recorder);
        recorder.read(&self.reader);
        self.text_binding.record(&mut recorder);
        if let Some(selection) = &self.training_selection {
            recorder.read(selection);
        }
        for tensor in &self.sources {
            if let Some(source) = &tensor.source {
                recorder.read(source);
            }
        }
        let arguments = (
            self.storage.control.device_ptr_value(),
            *self.reader.device_ptr(),
            self.inputs,
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            for &(source, destination, bytes) in &self.copies {
                // SAFETY: cold geometry bounds both addresses to the original
                // witnessed source and retained fixed destination allocation.
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        destination,
                        source,
                        bytes,
                        enqueue.stream().cu_stream(),
                    )
                }
                .result()
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            // SAFETY: the actual acquired reader, original service producers
            // and complete pending storage share this recorded lifetime.
            unsafe {
                self.execute.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }
}

struct PublishedReader {
    device: TrackedCudaSlice<PublicationLease>,
    aliases: Arc<()>,
    consumer_streams: BTreeSet<u64>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PublicationStepInput {
    role: u64,
    index: u64,
    destination: u64,
    capacity_bytes: u64,
    backing: u64,
    backing_bytes: u64,
    layout: SemanticTensorLayout,
}

// SAFETY: the fixed CUDA ABI contains only u64 words and the checked tensor layout.
unsafe impl DeviceRepr for PublicationStepInput {}
const _: () = assert!(size_of::<PublicationStepInput>() == 160);

#[derive(Clone, Debug)]
struct StepInputBankPlan {
    storage_slot: usize,
    span: std::ops::Range<usize>,
}

#[derive(Clone)]
struct StepInputPlan {
    role: u64,
    index: u64,
    layout: SemanticTensorLayout,
    banks: [StepInputBankPlan; 2],
}

fn step_input_overlap(
    data: u64,
    bytes: usize,
    base: u64,
    capacity: usize,
) -> Result<bool, SemanticTransitionError> {
    let end = data
        .checked_add(bytes as u64)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let limit = base
        .checked_add(capacity as u64)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    Ok(if bytes == 0 {
        data >= base && data < limit
    } else {
        data < limit && base < end
    })
}

fn plan_step_inputs(
    banks: &[Vec<PublicationRange>; 2],
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    prefix_capacity: u64,
    allocation_bytes: &[usize],
) -> Result<Vec<StepInputPlan>, SemanticTransitionError> {
    let mut expected = BTreeSet::from([(1, 0), (3, 0), (15, 0), (16, 0), (17, 0), (44, 0)]);
    expected.extend(
        banks[0]
            .iter()
            .filter(|range| matches!(range.role, 16 | 17))
            .map(|range| (range.role, range.index)),
    );
    for (&key, layout) in layouts
        .iter()
        .filter(|(key, _)| matches!(key.0, 4..=13 | 18..=25))
    {
        let model = matches!(key.0, 18..=25);
        let bytes = tensor_layout_bytes(layout)?;
        if key != (layout.role, layout.index)
            || (!model && (canonical_tensor_layout(*layout)? != *layout || bytes == 0))
            || (model && layout.logical_axis != u64::MAX)
        {
            return Err(publication_input_error(
                "step input must retain its original tensor layout",
            ));
        }
        expected.insert(key);
    }
    for role in (4..=13).chain(15..=25) {
        for (index, key) in expected.iter().filter(|key| key.0 == role).enumerate() {
            if key.1 != index as u64 {
                return Err(publication_input_error(
                    "step input tensor indexes must be contiguous",
                ));
            }
        }
    }
    for bank in banks {
        let actual = bank
            .iter()
            .filter(|range| matches!(range.role, 1 | 3..=13 | 15..=25 | 44))
            .map(|range| (range.role, range.index))
            .collect::<Vec<_>>();
        if actual.len() != expected.len()
            || actual.iter().copied().collect::<BTreeSet<_>>() != expected
        {
            return Err(publication_input_error(
                "step input banks must have the exact original input roster",
            ));
        }
    }
    let mut plan = Vec::with_capacity(expected.len());
    let mut spans = Vec::new();
    for (role, index) in expected {
        let ranges = banks.each_ref().map(|bank| {
            *bank
                .iter()
                .find(|range| (range.role, range.index) == (role, index))
                .expect("validated input roster")
        });
        let layout = if role == 1 {
            committed_prefix_layout(&ranges[0], prefix_capacity)?
        } else if matches!(role, 3 | 15..=17 | 44) {
            SemanticTensorLayout {
                role,
                index,
                scalar_type: 1,
                element_bytes: 1,
                rank: 1,
                logical_axis: u64::MAX,
                dimensions: [ranges[0].length_bytes, 0, 0, 0],
                strides_bytes: [1, 0, 0, 0],
            }
        } else {
            layouts[&(role, index)]
        };
        let bytes = tensor_layout_bytes(&layout)?;
        if (bytes == 0 && !matches!(role, 18..=25))
            || (role == 3 && ranges.iter().any(|range| range.length_bytes != 32))
        {
            return Err(publication_input_error(
                "step input has no complete fixed physical capacity",
            ));
        }
        if matches!(role, 15..=17)
            && ranges.iter().any(|range| {
                range.logical_begin != 0
                    || if role == 15 {
                        range.length_bytes % size_of::<RawFeedbackRecord>() as u64 != 0
                            || range.logical_end
                                > range.length_bytes / size_of::<RawFeedbackRecord>() as u64
                    } else {
                        range.logical_end != 0
                    }
            })
        {
            return Err(publication_input_error(
                "raw feedback input differs from its original record extent",
            ));
        }
        let mut resolved = Vec::with_capacity(2);
        for (bank, range) in ranges.iter().enumerate() {
            if role == 1 {
                committed_prefix_layout(range, prefix_capacity)?;
            }
            let slot = usize::try_from(range.storage_slot)
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            let capacity = *allocation_bytes
                .get(slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let span = publication_export_span(range, &layout, capacity)?;
            if span.len() != bytes {
                return Err(publication_input_error(
                    "step input differs from its full typed capacity",
                ));
            }
            spans.push((bank, role, index, slot, span.clone()));
            resolved.push((slot, span, capacity));
        }
        if role == 1 && resolved[0] != resolved[1] {
            return Err(publication_input_error(
                "prefix step inputs must share their original capacity allocation",
            ));
        }
        if matches!(role, 4 | 5 | 15 | 18..=25 | 44)
            && (resolved[0].0 == resolved[1].0
                || resolved[0].1 != resolved[1].1
                || resolved[0].2 != resolved[1].2)
        {
            return Err(publication_input_error("bank-selected step inputs require disjoint banks with the same complete backing geometry"));
        }
        plan.push(StepInputPlan {
            role,
            index,
            layout,
            banks: resolved
                .into_iter()
                .map(
                    |(storage_slot, span, _allocation_bytes)| StepInputBankPlan {
                        storage_slot,
                        span,
                    },
                )
                .collect::<Vec<_>>()
                .try_into()
                .expect("two publication banks"),
        });
    }
    for (ordinal, (bank, role, index, slot, span)) in spans.iter().enumerate() {
        for (other_bank, other_role, other_index, other_slot, other_span) in &spans[..ordinal] {
            let same_shared = bank != other_bank
                && role == other_role
                && index == other_index
                && slot == other_slot
                && span == other_span
                && matches!(*role, 1 | 16 | 17);
            let same_model =
                bank == other_bank && matches!(*role, 18..=25) && matches!(*other_role, 18..=25);
            if slot == other_slot
                && span.start < other_span.end
                && other_span.start < span.end
                && !same_shared
                && !same_model
            {
                return Err(publication_input_error(
                    "step input caches and distinct tensor ranges must be isolated",
                ));
            }
        }
    }
    Ok(plan)
}

/// Fixed inputs are prepared without a host choice of publication bank. The
/// producer resolves the actual device lease and preserves its original seals.
struct PreparedStepInputs {
    storage: Arc<PublicationStorage>,
    reader: DeviceMemoryView<PublicationLease>,
    header: TrackedCudaSlice<PublicationHeader>,
    source: TrackedCudaSlice<SourceSlot>,
    ranges: TrackedCudaSlice<PublicationRange>,
    metadata_digests: TrackedCudaSlice<u64>,
    bindings: [TrackedCudaSlice<PublicationStepInput>; 2],
    binding_values: [Vec<PublicationStepInput>; 2],
    plans: Vec<StepInputPlan>,
    views: [BTreeMap<(u64, u64), DeviceMemoryView<u8>>; 2],
    private: Vec<TrackedCudaSlice<u8>>,
    execute: CudaFunction,
    guard: CudaFunction,
}

impl PreparedStepInputs {
    fn plan(storage: &PublicationStorage) -> Result<Vec<StepInputPlan>, SemanticTransitionError> {
        plan_step_inputs(
            &storage.bank_templates,
            &storage.layouts,
            storage.contract_value.prefix_capacity,
            &storage
                .allocations
                .iter()
                .map(TrackedCudaSlice::len)
                .collect::<Vec<_>>(),
        )
    }

    fn allocation_bytes(plans: &[StepInputPlan]) -> Result<usize, SemanticTransitionError> {
        let fixed =
            size_of::<PublicationHeader>() + 32 * size_of::<SourceSlot>() + u64::BITS as usize;
        let bytes = plans
            .len()
            .checked_mul(size_of::<PublicationRange>() + 2 * size_of::<PublicationStepInput>())
            .and_then(|n| n.checked_add(fixed))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        plans
            .iter()
            .filter(|input| !matches!(input.role, 1 | 4 | 5 | 18..=25))
            .try_fold(bytes, |sum, input| {
                sum.checked_add(input.banks[0].span.len())
                    .ok_or(SemanticTransitionError::GenerationExhausted)
            })
    }

    fn allocate(
        provider: &CudaKernelProvider,
        storage: Arc<PublicationStorage>,
        reader: DeviceMemoryView<PublicationLease>,
    ) -> Result<Self, SemanticTransitionError> {
        let plans = Self::plan(&storage)?;
        let mut reservation = provider
            .memory()
            .reserve_bytes(Self::allocation_bytes(&plans)? as u64)
            .map_err(|error| runtime_error("step input reservation", error))?;
        Self::allocate_reserved(provider, storage, reader, plans, &mut reservation)
    }

    fn allocate_reserved(
        provider: &CudaKernelProvider,
        storage: Arc<PublicationStorage>,
        reader: DeviceMemoryView<PublicationLease>,
        plans: Vec<StepInputPlan>,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<Self, SemanticTransitionError> {
        let contract = storage.contract_value;
        if contract.abi != 1
            || contract.model_generation == 0
            || contract.window_capacity != 32
            || reader.len() != 1
        {
            return Err(publication_input_error(
                "fixed step inputs require the original admitted model contract and device reader",
            ));
        }
        let execute = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_step_inputs",
            )
            .ok_or_else(|| {
                runtime_error(
                    "kernel lookup",
                    "publication step input producer unavailable",
                )
            })?;
        let guard = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_step_input_guard",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "publication step input guard unavailable")
            })?;
        let mut private = Vec::new();
        let mut views = [BTreeMap::new(), BTreeMap::new()];
        let mut binding_values = [
            Vec::with_capacity(plans.len()),
            Vec::with_capacity(plans.len()),
        ];
        for input in &plans {
            if matches!(input.role, 1 | 4 | 5 | 18..=25) {
                for bank in 0..2 {
                    let plan = &input.banks[bank];
                    let view = storage.allocations[plan.storage_slot]
                        .view()
                        .slice(plan.span.clone());
                    let backing = if matches!(input.role, 18..=25) {
                        view.allocation_view()
                            .ok_or(SemanticTransitionError::ObservationMismatch)?
                    } else {
                        view.clone()
                    };
                    binding_values[bank].push(PublicationStepInput {
                        role: input.role,
                        index: input.index,
                        destination: *view.device_ptr(),
                        capacity_bytes: view.len() as u64,
                        backing: *backing.device_ptr(),
                        backing_bytes: backing.len() as u64,
                        layout: input.layout,
                    });
                    views[bank].insert((input.role, input.index), view);
                }
            } else {
                private.push(
                    reservation
                        .alloc::<u8>(input.banks[0].span.len())
                        .map_err(|error| runtime_error("step private input allocation", error))?,
                );
                let view = private
                    .last()
                    .expect("retained private input allocation")
                    .view();
                for bank in 0..2 {
                    binding_values[bank].push(PublicationStepInput {
                        role: input.role,
                        index: input.index,
                        destination: *view.device_ptr(),
                        capacity_bytes: view.len() as u64,
                        backing: *view.device_ptr(),
                        backing_bytes: view.len() as u64,
                        layout: if matches!(input.role, 3 | 15..=17 | 44) {
                            SemanticTensorLayout::default()
                        } else {
                            input.layout
                        },
                    });
                    views[bank].insert((input.role, input.index), view.clone());
                }
            }
        }
        let owner = Self {
            header: reservation
                .alloc(1)
                .map_err(|error| runtime_error("step header allocation", error))?,
            source: reservation
                .alloc(32)
                .map_err(|error| runtime_error("step source allocation", error))?,
            ranges: reservation
                .alloc(plans.len())
                .map_err(|error| runtime_error("step range allocation", error))?,
            metadata_digests: reservation
                .alloc(8)
                .map_err(|error| runtime_error("step digest allocation", error))?,
            bindings: [
                reservation
                    .alloc(plans.len())
                    .map_err(|error| runtime_error("step binding allocation", error))?,
                reservation
                    .alloc(plans.len())
                    .map_err(|error| runtime_error("step binding allocation", error))?,
            ],
            storage,
            reader,
            plans,
            views,
            private,
            binding_values,
            execute,
            guard,
        };
        owner.validate_output_regions()?;
        Ok(owner)
    }

    fn validate_output_regions(&self) -> Result<(), SemanticTransitionError> {
        let mut output = vec![
            (
                self.header.device_ptr_value(),
                size_of::<PublicationHeader>(),
            ),
            (self.source.device_ptr_value(), 32 * size_of::<SourceSlot>()),
            (
                self.ranges.device_ptr_value(),
                self.ranges.len() * size_of::<PublicationRange>(),
            ),
            (self.metadata_digests.device_ptr_value(), u64::BITS as usize),
            (
                self.bindings[0].device_ptr_value(),
                self.bindings[0].len() * size_of::<PublicationStepInput>(),
            ),
            (
                self.bindings[1].device_ptr_value(),
                self.bindings[1].len() * size_of::<PublicationStepInput>(),
            ),
        ];
        output.extend(
            self.private
                .iter()
                .map(|allocation| (allocation.device_ptr_value(), allocation.len())),
        );
        let mut original = self
            .storage
            .allocations
            .iter()
            .map(|allocation| (allocation.device_ptr_value(), allocation.len()))
            .collect::<Vec<_>>();
        original.extend(
            self.storage
                .banks
                .iter()
                .map(|bank| (bank.device_ptr_value(), size_of::<PublicationBank>())),
        );
        original.extend(self.storage.directories.iter().map(|directory| {
            (
                directory.device_ptr_value(),
                directory.len() * size_of::<PublicationRange>(),
            )
        }));
        original.extend([
            (*self.reader.device_ptr(), size_of::<PublicationLease>()),
            (
                self.storage.control.device_ptr_value(),
                size_of::<PublicationControl>(),
            ),
            (
                self.storage.contract.device_ptr_value(),
                size_of::<PublicationContract>(),
            ),
            (
                self.storage.storage.device_ptr_value(),
                self.storage.storage.len() * size_of::<PublicationStorageEntry>(),
            ),
            (
                self.storage.role_counts.device_ptr_value(),
                self.storage.role_counts.len() * size_of::<PublicationRoleCount>(),
            ),
            (
                self.storage.terminals.device_ptr_value(),
                self.storage.terminals.len() * size_of::<u64>(),
            ),
        ]);
        for (ordinal, &(begin, bytes)) in output.iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            let end = begin
                .checked_add(bytes as u64)
                .filter(|&end| begin != 0 && end > begin)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            for &(other, other_bytes) in original.iter().chain(&output[..ordinal]) {
                let other_end = other
                    .checked_add(other_bytes as u64)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if begin < other_end && other < end {
                    return Err(publication_input_error(
                        "fixed step outputs overlap original storage or another output",
                    ));
                }
            }
        }
        Ok(())
    }

    // Session installs this owner before uploading fixed pointer/layout metadata.
    // No numerical values or expected digests travel through the host.
    fn initialize(&self, provider: &CudaKernelProvider) -> Result<(), SemanticTransitionError> {
        for bank in 0..2 {
            upload_publication(provider, &self.binding_values[bank], &self.bindings[bank])?;
        }
        Ok(())
    }

    fn enqueue(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.read(&self.reader);
        recorder.read(&self.storage.control);
        recorder.read(&self.storage.storage);
        recorder.read(&self.storage.contract);
        recorder.read(&self.storage.role_counts);
        recorder.read(&self.storage.terminals);
        for bank in &self.storage.banks {
            recorder.read(bank);
        }
        for directory in &self.storage.directories {
            recorder.read(directory);
        }
        for allocation in &self.storage.allocations {
            recorder.read(allocation);
        }
        for bindings in &self.bindings {
            recorder.read(bindings);
        }
        recorder.write(&self.header);
        recorder.write(&self.source);
        recorder.write(&self.ranges);
        recorder.write(&self.metadata_digests);
        for allocation in &self.private {
            recorder.write(allocation);
        }
        let arguments = (
            self.storage.control.device_ptr_value(),
            *self.reader.device_ptr(),
            self.bindings[0].device_ptr_value(),
            self.bindings[1].device_ptr_value(),
            self.bindings[0].len() as u64,
            self.header.device_ptr_value(),
            self.source.device_ptr_value(),
            self.ranges.device_ptr_value(),
            self.metadata_digests.device_ptr_value(),
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: every selectable bank and disjoint output is owned before
            // enqueue. The kernel validates all original seals before writing.
            unsafe {
                self.execute.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    fn verify(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.read(&self.header);
        recorder.read(&self.source);
        recorder.read(&self.ranges);
        recorder.read(&self.reader);
        for bindings in &self.bindings {
            recorder.read(bindings);
        }
        recorder.read(&self.metadata_digests);
        for bank in &self.views {
            for view in bank.values() {
                recorder.read(view);
            }
        }
        for allocation in &self.private {
            recorder.read(allocation);
        }
        let arguments = (
            *self.reader.device_ptr(),
            self.header.device_ptr_value(),
            self.source.device_ptr_value(),
            self.bindings[0].device_ptr_value(),
            self.bindings[1].device_ptr_value(),
            self.bindings[0].len() as u64,
            self.ranges.device_ptr_value(),
            self.metadata_digests.device_ptr_value(),
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: private producer snapshots and original shared aliases are
            // retained. No later bank or digest can replace this original basis.
            unsafe {
                self.guard.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    fn field_view(
        &self,
        field: PublicationBankField,
    ) -> Result<DeviceMemoryView<u8>, SemanticTransitionError> {
        let (span, _, _) = field.layout();
        // SAFETY: only initialized public Source/scalar fields of the checked
        // padding-free ABI are exposed, never the private header or hash words.
        unsafe {
            match field {
                PublicationBankField::Source => self
                    .source
                    .view()
                    .cast::<u8>()
                    .ok_or(SemanticTransitionError::ObservationMismatch),
                _ => self
                    .header
                    .view()
                    .cast::<u8>()
                    .map(|view| view.slice(span))
                    .ok_or(SemanticTransitionError::ObservationMismatch),
            }
        }
    }

    fn training_coordinates_view(&self) -> Result<DeviceMemoryView<u64>, SemanticTransitionError> {
        let begin = std::mem::offset_of!(PublicationHeader, training_cursor);
        let end = begin
            .checked_add(5 * size_of::<u64>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        unsafe {
            self.header
                .view()
                .cast::<u8>()
                .map(|view| view.slice(begin..end))
                .and_then(|view| view.cast::<u64>())
                .ok_or(SemanticTransitionError::ObservationMismatch)
        }
    }

    fn owns_tensor(
        &self,
        tensor: &PreparedSemanticTensor,
    ) -> Result<bool, SemanticTransitionError> {
        let bytes = tensor.source.as_ref().map_or(0, DeviceMemoryView::len);
        for field in [
            PublicationBankField::Source,
            PublicationBankField::PrefixExtent,
            PublicationBankField::RingHead,
            PublicationBankField::Terminal,
        ] {
            let view = self.field_view(field)?;
            let start = *view.device_ptr();
            if !step_input_overlap(tensor.data, bytes, start, view.len())? {
                continue;
            }
            let (_, shape, strides) = field.layout();
            let mut layout = SemanticTensorLayout {
                role: 0,
                index: tensor.layout.index,
                scalar_type: 3,
                element_bytes: 8,
                rank: shape.len() as u64,
                logical_axis: u64::MAX,
                ..SemanticTensorLayout::default()
            };
            for (axis, dimension) in shape.into_iter().enumerate() {
                layout.dimensions[axis] = dimension as u64;
            }
            for (axis, stride) in strides.into_iter().enumerate() {
                layout.strides_bytes[axis] = stride as u64 * 8;
            }
            if tensor.data != start
                || bytes != view.len()
                || tensor.layout != layout
                || (tensor.logical_begin, tensor.logical_end) != (0, 0)
            {
                return Err(publication_input_error("native source and scalar aliases must retain their exact exported type and extent"));
            }
            if !step_input_allocation_matches(tensor, &view) {
                return Err(publication_input_error(
                    "native source and scalar allocation origin does not match its producer",
                ));
            }
            return Ok(true);
        }
        for (pointer, capacity) in [
            (
                self.header.device_ptr_value(),
                size_of::<PublicationHeader>(),
            ),
            (
                self.ranges.device_ptr_value(),
                self.ranges.len() * size_of::<PublicationRange>(),
            ),
            (
                self.bindings[0].device_ptr_value(),
                self.bindings[0].len() * size_of::<PublicationStepInput>(),
            ),
            (
                self.bindings[1].device_ptr_value(),
                self.bindings[1].len() * size_of::<PublicationStepInput>(),
            ),
            (
                self.metadata_digests.device_ptr_value(),
                self.metadata_digests.len() * size_of::<u64>(),
            ),
        ] {
            if step_input_overlap(tensor.data, bytes, pointer, capacity)? {
                return Err(publication_input_error("native private header, range and digest storage cannot become a content operand"));
            }
        }
        // Select by the original typed coordinate before testing overlap: two
        // legitimate model views can share any subset of the same backing.
        if let Some(input) = self
            .plans
            .iter()
            .find(|input| (input.role, input.index) == (tensor.layout.role, tensor.layout.index))
        {
            if (18..=25).contains(&input.role) && bytes == 0 && tensor.native_allocation.is_none() {
                return Err(publication_input_error("empty prepared model inputs require their original native allocation provenance"));
            }
            for bank in 0..2 {
                let view = &self.views[bank][&(input.role, input.index)];
                let range = self.storage.bank_templates[bank]
                    .iter()
                    .find(|range| (range.role, range.index) == (input.role, input.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if step_input_alias_matches(
                    input,
                    range,
                    *view.device_ptr(),
                    view.len(),
                    tensor_content_identity(tensor),
                ) && step_input_allocation_matches(tensor, view)
                {
                    return Ok(true);
                }
            }
        }
        for bank in 0..2 {
            for input in &self.plans {
                let view = &self.views[bank][&(input.role, input.index)];
                let start = *view.device_ptr();
                if !(step_input_overlap(tensor.data, bytes, start, view.len())?
                    || input.role == 1 && bytes == 0 && tensor.data == start)
                {
                    continue;
                }
                let range = self.storage.bank_templates[bank]
                    .iter()
                    .find(|range| (range.role, range.index) == (input.role, input.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if input.role == 1
                    && step_input_alias_matches(
                        input,
                        range,
                        start,
                        view.len(),
                        tensor_content_identity(tensor),
                    )
                    && step_input_allocation_matches(tensor, view)
                {
                    return Ok(true);
                }
                return Err(publication_input_error("native step aliases must retain their original type, capacity, interval and coordinate"));
            }
        }
        for allocation in &self.private {
            let retained_origin = tensor.native_allocation.as_ref().is_some_and(|origin| {
                allocation
                    .view()
                    .allocation_provenance()
                    .is_some_and(|expected| origin.same_allocation(&expected))
            });
            if retained_origin
                || step_input_overlap(
                    tensor.data,
                    bytes,
                    allocation.device_ptr_value(),
                    allocation.len(),
                )?
            {
                return Err(publication_input_error(
                    "native step backing padding cannot become a separate content operand",
                ));
            }
        }
        Ok(false)
    }
}

fn step_input_allocation_matches(
    tensor: &PreparedSemanticTensor,
    view: &DeviceMemoryView<u8>,
) -> bool {
    match &tensor.native_allocation {
        Some(origin) => view
            .allocation_provenance()
            .is_some_and(|expected| origin.same_allocation(&expected)),
        // An address locates nonempty retained storage, but all independent
        // zero-byte native allocations may have the same null address.
        None => !view.is_empty(),
    }
}

fn step_input_alias_matches(
    input: &StepInputPlan,
    range: &PublicationRange,
    data: u64,
    capacity: usize,
    actual: (SemanticTensorLayout, u64, u64, u64, usize),
) -> bool {
    if actual.3 != data {
        return false;
    }
    let typed_capacity = actual.4 == capacity
        && actual.0 == input.layout
        && (actual.1, actual.2) == (range.logical_begin, range.logical_end);
    // The public raw prefix export is the exact committed byte interval of
    // this same original capacity allocation. It still uses the producer's
    // original raw prefix seal, including the legal empty prefix.
    let raw_prefix = input.role == 1
        && input.index == 0
        && range.role == 1
        && range.index == 0
        && actual.0 == publication_record_layout(range)
        && (actual.1, actual.2) == (0, 0)
        && u64::try_from(actual.4).ok() == Some(range.length_bytes)
        && actual.4 <= capacity;
    typed_capacity || raw_prefix
}

// Numerical producers and their original seals outlive the bank they consumed.
// A step is installed alongside acquisition, before any operation can enqueue.
// Its identity is set only after that actual device acquisition is validated.
struct StepContentStorage {
    identity: Option<SemanticPublishedIdentity>,
    content_witnesses: Arc<()>,
    aliases: Arc<()>,
    consumer_streams: BTreeSet<u64>,
    inputs: Option<Arc<PreparedStepInputs>>,
    // Deleters do not establish completion. Final step retirement joins every
    // recorded consumer before clearing any of these actual output owners.
    feedback: Vec<FeedbackBuffers>,
    content: Vec<TensorContentBuffers>,
    #[cfg_attr(
        not(feature = "semantic-policy"),
        expect(
            dead_code,
            reason = "adjoints are consumed by the semantic-policy VJP path"
        )
    )]
    adjoints: Vec<DeviceMemoryView<u8>>,
    #[cfg(feature = "semantic-policy")]
    policy_vjp_workspaces: Vec<Arc<PolicyVjpWorkspace>>,
    prepared: Option<PreparedStepStorage>,
}

impl StepContentStorage {
    fn new() -> Self {
        Self {
            identity: None,
            content_witnesses: Arc::new(()),
            aliases: Arc::new(()),
            consumer_streams: BTreeSet::new(),
            inputs: None,
            feedback: Vec::new(),
            content: Vec::new(),
            adjoints: Vec::new(),
            #[cfg(feature = "semantic-policy")]
            policy_vjp_workspaces: Vec::new(),
            prepared: None,
        }
    }
}

/// A fixed future step issued by its Session during cold segment construction.
/// It owns no host publication identity; its actual parent is acquired on device.
#[derive(Clone, Debug)]
pub struct SemanticPreparedStep {
    issuer: Arc<()>,
    scope: Arc<()>,
    token: u64,
}

/// Lexical native graph construction. The caller owns this outside any Session
/// mutex while invoking original model producers and instantiating the graph.
pub struct SemanticPreparedSegmentCapture {
    builder: crate::cuda_graph::ConditionalCudaGraphSequenceBuilder,
    scope: Arc<()>,
}

impl SemanticPreparedSegmentCapture {
    pub fn add_conditional_if<P, F, E>(
        &mut self,
        stream: &CudaStream,
        preflight: P,
        body: F,
    ) -> Result<u64, crate::cuda_graph::CudaConditionalGraphUnavailable>
    where
        P: FnOnce(u64) -> Result<(), E>,
        E: fmt::Display,
        F: FnOnce(
            &crate::cuda_graph::ConditionalCudaGraphBody,
        ) -> Result<(), crate::cuda_graph::CudaConditionalGraphUnavailable>,
    {
        self.builder.add_conditional_if(stream, preflight, body)
    }

    pub fn instantiate(self) -> Result<SemanticPreparedExecutable, SemanticTransitionError> {
        let graph = self
            .builder
            .instantiate()
            .map_err(|error| runtime_error("bounded segment instantiation", error))?;
        Ok(SemanticPreparedExecutable {
            scope: self.scope,
            graph,
        })
    }
}

/// A graph produced by one authentic Session construction scope. Only the
/// owning Session can take it; the caller retains it on installation failure.
pub struct SemanticPreparedExecutable {
    scope: Arc<()>,
    graph: CapturedCudaGraph,
}

struct PreparedSegmentState {
    issuer: Arc<()>,
    scope: Arc<()>,
    tokens: Vec<u64>,
    transitions: Vec<SemanticTransitionKind>,
    next: usize,
    active: bool,
    finished: bool,
    capturing: bool,
    #[cfg_attr(
        not(any(test, feature = "semantic-policy")),
        expect(
            dead_code,
            reason = "submission is owned by the semantic-policy graph path"
        )
    )]
    submitted: bool,
    completed: bool,
}

impl PreparedSegmentState {
    fn new(
        issuer: Arc<()>,
        tokens: Vec<u64>,
        transitions: impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone,
    ) -> Result<Self, SemanticTransitionError> {
        if tokens.is_empty()
            || tokens.contains(&0)
            || tokens.iter().copied().collect::<BTreeSet<_>>().len() != tokens.len()
            || tokens.len() != transitions.len()
            || transitions
                .clone()
                .any(|kind| kind == SemanticTransitionKind::Drain)
        {
            return Err(publication_input_error("a prepared segment requires distinct owned steps and one frozen proposal, recompute, or update mode per step"));
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(transitions.len())
            .map_err(|error| runtime_error("prepared schedule reservation", error))?;
        owned.extend(transitions);
        Ok(Self {
            issuer,
            scope: Arc::new(()),
            tokens,
            transitions: owned,
            next: 0,
            active: false,
            finished: false,
            capturing: false,
            submitted: false,
            completed: false,
        })
    }

    fn handles(&self) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(self.tokens.len())
            .map_err(|error| runtime_error("prepared handle reservation", error))?;
        handles.extend(self.tokens.iter().map(|&token| SemanticPreparedStep {
            issuer: Arc::clone(&self.issuer),
            scope: Arc::clone(&self.scope),
            token,
        }));
        Ok(handles)
    }

    fn check(
        &self,
        step: &SemanticPreparedStep,
        issuer: &Arc<()>,
        recording: bool,
    ) -> Result<(), SemanticTransitionError> {
        self.check_retained(step, issuer)?;
        if self.finished
            || (recording && (!self.active || self.tokens.get(self.next) != Some(&step.token)))
        {
            return Err(publication_input_error(
                "prepared step differs from its owning Session or active construction scope",
            ));
        }
        Ok(())
    }

    fn check_retained(
        &self,
        step: &SemanticPreparedStep,
        issuer: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(issuer, &self.issuer)
            || !Arc::ptr_eq(&step.issuer, issuer)
            || !Arc::ptr_eq(&step.scope, &self.scope)
            || !self.tokens.contains(&step.token)
        {
            return Err(publication_input_error(
                "retained prepared step differs from its original Session and construction scope",
            ));
        }
        Ok(())
    }

    fn requested_kind(
        &self,
        step: &SemanticPreparedStep,
        issuer: &Arc<()>,
    ) -> Result<SemanticTransitionKind, SemanticTransitionError> {
        self.check_retained(step, issuer)?;
        let index = self
            .tokens
            .iter()
            .position(|&token| token == step.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok(self.transitions[index])
    }

    fn enter(
        &mut self,
        step: &SemanticPreparedStep,
        issuer: &Arc<()>,
    ) -> Result<SemanticTransitionKind, SemanticTransitionError> {
        self.check(step, issuer, false)?;
        if self.active || self.tokens.get(self.next) != Some(&step.token) {
            return Err(publication_input_error(
                "prepared steps must record once in their original bounded order",
            ));
        }
        self.active = true;
        Ok(self.transitions[self.next])
    }

    fn leave(
        &mut self,
        step: &SemanticPreparedStep,
        issuer: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        self.check(step, issuer, true)?;
        self.active = false;
        self.next += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SemanticTransitionError> {
        if self.finished || self.active || self.next != self.tokens.len() {
            return Err(publication_input_error(
                "prepared segment has unrecorded or unfinished steps",
            ));
        }
        self.finished = true;
        Ok(())
    }

    #[cfg(any(test, feature = "semantic-policy"))]
    fn submit(&mut self) -> Result<(), SemanticTransitionError> {
        if !self.finished
            || self.active
            || self.next != self.tokens.len()
            || self.submitted
            || self.completed
        {
            return Err(publication_input_error(
                "prepared segment submission requires one complete unused construction",
            ));
        }
        self.submitted = true;
        Ok(())
    }
}

struct PreparedStepStorage {
    reader: TrackedCudaSlice<PublicationLease>,
    #[cfg(feature = "semantic-policy")]
    policy_buffers: Option<PolicyBuffers>,
    #[cfg(feature = "semantic-policy")]
    policy: Option<PolicyStorage>,
    #[cfg(feature = "semantic-policy")]
    support: Option<TrackedCudaSlice<u8>>,
    #[cfg(feature = "semantic-policy")]
    receipts: Option<TrackedCudaSlice<SemanticTransitionReceipt>>,
    #[cfg(feature = "semantic-policy")]
    state: TrackedCudaSlice<DeviceState>,
    #[cfg(feature = "semantic-policy")]
    policy_sources: Vec<PreparedSemanticTensor>,
    #[cfg(feature = "semantic-policy")]
    policy_witness: Option<SemanticTensorContentWitness>,
    digests: Option<Arc<TrackedCudaSlice<u64>>>,
    next_digest: usize,
    model_work: Option<PreparedModelWork>,
    model_update: Option<PreparedModelUpdate>,
    admit: CudaFunction,
    kind_gate: CudaFunction,
    kind_bank_gate: CudaFunction,
    active_gate: CudaFunction,
    #[cfg_attr(
        not(feature = "semantic-policy"),
        expect(
            dead_code,
            reason = "drain preparation is captured by the semantic-policy graph"
        )
    )]
    drain_prepare: CudaFunction,
    release: CudaFunction,
    witness: CudaFunction,
    content_guard: CudaFunction,
    inputs_recorded: bool,
    continuation: Option<PreparedContinuation>,
    transition_recorded: u8,
    drain_recorded: bool,
    observed: bool,
    result: TrackedCudaSlice<PreparedStepResult>,
    training_origin: Option<DeviceMemoryView<SemanticTrainingViewOriginRecord>>,
    training_view: Option<SemanticSelectedTrainingView>,
    result_kernel: CudaFunction,
}

struct PreparedModelUpdate {
    bindings: TrackedCudaSlice<ModelUpdateBinding>,
    admissibility: TrackedCudaSlice<u8>,
    output: Option<BoundModelUpdate>,
    #[cfg_attr(
        not(feature = "semantic-policy"),
        expect(
            dead_code,
            reason = "update admissibility is copied by the semantic-policy graph"
        )
    )]
    admissibility_copy: CudaFunction,
    #[cfg_attr(
        not(feature = "semantic-policy"),
        expect(
            dead_code,
            reason = "model updates are copied by the semantic-policy graph"
        )
    )]
    copy: CudaFunction,
}

struct BoundModelUpdate {
    values: Vec<ModelUpdateBinding>,
    allocations: Vec<PreparedSemanticTensor>,
    admissibility: PreparedSemanticTensor,
    _witness: SemanticTensorContentWitness,
}

impl PreparedModelUpdate {
    fn descriptor(&self) -> (u64, u64, u64) {
        (
            self.bindings.device_ptr_value(),
            self.bindings.len() as u64,
            self.admissibility.device_ptr_value(),
        )
    }

    fn record(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.bindings);
        recorder.read(&self.admissibility);
        if let Some(output) = &self.output {
            for allocation in &output.allocations {
                if let Some(source) = &allocation.source {
                    recorder.read(source);
                }
            }
            if let Some(source) = &output.admissibility.source {
                recorder.read(source);
            }
        }
    }

    #[cfg(feature = "semantic-policy")]
    fn enqueue_copy(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        storage: &PublicationStorage,
        reader: &TrackedCudaSlice<PublicationLease>,
    ) -> Result<(), SemanticTransitionError> {
        let output = self.output.as_ref().ok_or_else(|| {
            publication_input_error("prepared update has no original model output binding")
        })?;
        let allocation_count = u32::try_from(output.values.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        if allocation_count == 0 || allocation_count > 65_535 {
            return Err(publication_input_error(
                "prepared update model allocation roster exceeds the CUDA launch geometry",
            ));
        }
        let maximum_bytes = output
            .values
            .iter()
            .map(|binding| binding.bytes)
            .max()
            .unwrap_or(0);
        let blocks = maximum_bytes.div_ceil(256).clamp(1, 65_535) as u32;
        let mut recorder = domain.new_strict_recorder();
        storage.record(&mut recorder);
        recorder.read(reader);
        self.record(&mut recorder);
        recorder.write(&self.admissibility);
        let admissibility_source = output.admissibility.data;
        let admissibility_destination = self.admissibility.device_ptr_value();
        let arguments = (
            storage.control.device_ptr_value(),
            reader.device_ptr_value(),
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: the producer gate was checked by the same captured
            // content witness as the complete update backing. The native byte
            // is the stable predicate consumed by both copy and publication.
            unsafe {
                self.admissibility_copy.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (admissibility_source, admissibility_destination),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
            // SAFETY: the continuation kernel validated this immutable binding
            // roster first. Each CUDA grid row copies one disjoint allocation
            // into the inactive neural bank retained by PublicationStorage.
            unsafe {
                self.copy.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (blocks, allocation_count, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }
}

struct PreparedModelWork {
    recording: ModelWorkRecording,
    device: TrackedCudaSlice<ModelWorkEvent>,
    actual: TrackedCudaSlice<u64>,
    reset: CudaFunction,
    capture_bank: Option<usize>,
    replay_cursor: usize,
}

const PREPARED_TRANSITION_BANKS: u8 = 0b11;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModelWorkInput {
    events: u64,
    count: u64,
    bound: u64,
}

impl PreparedModelWork {
    #[cfg(feature = "semantic-policy")]
    fn begin_capture(&mut self, bank: usize, recorded: u8) -> Result<(), &'static str> {
        if bank > 1 || self.capture_bank.is_some() {
            return Err("model work capture requires one inactive bank recording");
        }
        if (bank == 0 && (recorded != 0 || self.recording.frozen_bound().is_some()))
            || (bank == 1 && (recorded != 1 || self.recording.frozen_bound().is_none()))
        {
            return Err("model work banks must record in zero then one order");
        }
        self.capture_bank = Some(bank);
        self.replay_cursor = 0;
        Ok(())
    }

    fn next_slot(&self) -> Result<usize, &'static str> {
        match self.capture_bank {
            Some(0) => Ok(self.recording.events().len()),
            Some(1) => Ok(self.replay_cursor),
            _ => Err("model work event requires an active prepared bank capture"),
        }
    }

    fn record_event(&mut self, event: ModelWorkEvent) -> Result<usize, &'static str> {
        let slot = self.next_slot()?;
        match self.capture_bank {
            Some(0) => self.recording.push(event)?,
            Some(1) => {
                if self.recording.events().get(slot) != Some(&event) {
                    return Err("second model branch differs from original work geometry");
                }
                self.replay_cursor += 1;
            }
            _ => return Err("model work event requires an active prepared bank capture"),
        }
        Ok(slot)
    }

    #[cfg(feature = "semantic-policy")]
    fn finish_capture(&mut self, bank: usize) -> Result<(), &'static str> {
        if self.capture_bank != Some(bank) {
            return Err("prepared transition differs from its active model work bank");
        }
        if bank == 0 {
            self.recording.freeze()?;
        } else if self.replay_cursor != self.recording.events().len() {
            return Err("second model branch omitted original work occurrences");
        }
        self.capture_bank = None;
        self.replay_cursor = 0;
        Ok(())
    }

    fn reset_slots(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        first: usize,
        count: usize,
    ) -> Result<(), SemanticTransitionError> {
        let end = first
            .checked_add(count)
            .filter(|end| *end <= self.actual.len() / 3)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut recorder = domain.new_strict_recorder();
        recorder.write(&self.actual);
        let arguments = (
            self.actual.device_ptr_value() + (first * 3 * size_of::<u64>()) as u64,
            ((end - first) * 3) as u64,
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: the checked range belongs to this original step's cold
            // allocation. Reset and producer kernels share the capture stream.
            unsafe {
                self.reset.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    fn descriptor(&self) -> ModelWorkInput {
        ModelWorkInput {
            events: self.device.device_ptr_value(),
            count: self.recording.events().len() as u64,
            bound: self
                .recording
                .frozen_bound()
                .expect("original work frozen before enqueue"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PreparedStepResult {
    abi: u64,
    word: u64,
    refusal: u64,
    header: PublicationHeader,
    advanced: u64,
}

// SAFETY: the C ABI contains initialized padding-free U64 fields and the
// already verified padding-free publication header representation.
unsafe impl DeviceRepr for PreparedStepResult {}

/// Private device content witness for the exact tensors of one retained parent.
/// Transient witnesses prove stability after capture; model witnesses compare
/// against the originally sealed model baseline. Neither proves computation.
/// The trusted model owner binds the witness to its actual original execution.
/// No digest, device pointer, status, or writable witness storage is exported.
#[derive(Clone)]
pub struct SemanticTensorContentWitness {
    issuer: Arc<()>,
    reader_token: u64,
    index: usize,
    _witness: Arc<()>,
}

struct TensorContentBuffers {
    tensors: Vec<PreparedSemanticTensor>,
    seals: TensorContentSeals,
    // A verification may hand off additional managed aliases. Keep them even
    // after enqueue failure; final step release joins consumer completion.
    verification_inputs: Vec<PreparedSemanticTensor>,
}

enum TensorContentSeals {
    Captured(Vec<CapturedTensorDigest>),
    // Original device seals and contract bytes, not a later bank directory or
    // a replacement baseline computed from the live numerical producer.
    Model(ModelContentSeals),
}

struct ModelContentSeals {
    ranges: TrackedCudaSlice<PublicationRange>,
    contract: TrackedCudaSlice<u8>,
    guard: CudaFunction,
    prepared: Option<PreparedModelContentSeals>,
}

struct PreparedModelContentSeals {
    storage: Arc<PublicationStorage>,
    reader: DeviceMemoryView<PublicationLease>,
    roster: TrackedCudaSlice<u64>,
    execute: CudaFunction,
}

struct ModelContentCopyPlan {
    directory_offsets: Vec<usize>,
    contract_storage_slot: usize,
    contract_span: std::ops::Range<usize>,
}

fn model_content_copy_plan(
    directory: &[PublicationRange],
    tensor_ranges: &[usize],
    allocation_bytes: &[usize],
) -> Result<ModelContentCopyPlan, SemanticTransitionError> {
    let mut contracts = directory
        .iter()
        .enumerate()
        .filter(|(_, range)| range.role == 44);
    let (contract_index, contract) = contracts
        .next()
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let mut seen = BTreeSet::new();
    if contracts.next().is_some()
        || contract.index != 0
        || contract.generation != 1
        || contract.length_bytes == 0
        || tensor_ranges.is_empty()
        || tensor_ranges.iter().any(|&index| {
            !seen.insert(index)
                || directory
                    .get(index)
                    .is_none_or(|range| !matches!(range.role, 18..=20))
        })
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let contract_storage_slot = usize::try_from(contract.storage_slot)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    let allocation = *allocation_bytes
        .get(contract_storage_slot)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let begin = usize::try_from(contract.offset_bytes)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    let bytes = usize::try_from(contract.length_bytes)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    let end = begin
        .checked_add(bytes)
        .filter(|&end| end <= allocation)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let directory_offsets = tensor_ranges
        .iter()
        .copied()
        .chain(std::iter::once(contract_index))
        .map(|index| {
            index
                .checked_mul(size_of::<PublicationRange>())
                .ok_or(SemanticTransitionError::ObservationMismatch)
        })
        .collect::<Result<Vec<_>, _>>()?;
    directory_offsets
        .len()
        .checked_mul(size_of::<PublicationRange>())
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    Ok(ModelContentCopyPlan {
        directory_offsets,
        contract_storage_slot,
        contract_span: begin..end,
    })
}

impl ModelContentSeals {
    fn allocate(
        provider: &CudaKernelProvider,
        plan: &ModelContentCopyPlan,
    ) -> Result<Self, SemanticTransitionError> {
        let guard = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_retained_model_contract_guard",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "retained model contract guard unavailable")
            })?;
        Ok(Self {
            ranges: allocate_publication(provider, plan.directory_offsets.len())?,
            contract: allocate_publication(provider, plan.contract_span.len())?,
            guard,
            prepared: None,
        })
    }

    fn snapshot_prepared(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let source = self
            .prepared
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut recorder = domain.new_strict_recorder();
        source.storage.record(&mut recorder);
        recorder.read(&source.reader);
        recorder.read(&source.roster);
        recorder.write(&self.ranges);
        recorder.write(&self.contract);
        let arguments = (
            source.storage.control.device_ptr_value(),
            *source.reader.device_ptr(),
            source.roster.device_ptr_value(),
            (source.roster.len() / 2) as u64,
            self.ranges.device_ptr_value(),
            self.contract.device_ptr_value(),
            self.contract.len() as u64,
        );
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: fixed roster and output capacities are cold-owned; the
            // kernel selects and validates the actual device lease and copies
            // original model seals and contract bytes without resealing them.
            unsafe {
                source.execute.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    fn snapshot(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        directory: &TrackedCudaSlice<PublicationRange>,
        source: &TrackedCudaSlice<u8>,
        plan: &ModelContentCopyPlan,
    ) -> Result<(), SemanticTransitionError> {
        let row_bytes = size_of::<PublicationRange>();
        let directory_bytes = directory
            .len()
            .checked_mul(row_bytes)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if self.ranges.len() != plan.directory_offsets.len()
            || self.contract.len() != plan.contract_span.len()
            || plan.contract_span.end > source.len()
            || plan.directory_offsets.iter().any(|&offset| {
                offset
                    .checked_add(row_bytes)
                    .is_none_or(|end| end > directory_bytes)
            })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut recorder = domain.new_strict_recorder();
        recorder.read(directory);
        recorder.read(source);
        recorder.write(&self.ranges);
        recorder.write(&self.contract);
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: every complete original row and the raw contract interval
            // are bounded above by their actual tracked allocations. Private
            // destinations are retained by Session before this first enqueue.
            unsafe {
                for (index, &offset) in plan.directory_offsets.iter().enumerate() {
                    sys::cuMemcpyDtoDAsync_v2(
                        self.ranges.device_ptr_value() + (index * row_bytes) as u64,
                        directory.device_ptr_value() + offset as u64,
                        row_bytes,
                        enqueue.stream().cu_stream(),
                    )
                    .result()
                    .map_err(|e| XlogError::Kernel(format!("original model seal snapshot: {e}")))?;
                }
                sys::cuMemcpyDtoDAsync_v2(
                    self.contract.device_ptr_value(),
                    source.device_ptr_value() + plan.contract_span.start as u64,
                    self.contract.len(),
                    enqueue.stream().cu_stream(),
                )
                .result()
                .map_err(|e| XlogError::Kernel(format!("original model contract snapshot: {e}")))?;
            }
            Ok::<(), XlogError>(())
        })
    }

    fn verify_contract(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.read(&self.ranges);
        recorder.read(&self.contract);
        let arguments = (
            self.contract.device_ptr_value(),
            self.contract.len() as u64,
            self.ranges.device_ptr_value()
                + ((self.ranges.len() - 1) * size_of::<PublicationRange>()) as u64,
        );
        let execute = self.guard.clone();
        enqueue_recorded(domain, poisoned, recorder, |enqueue| {
            // SAFETY: the last retained row is the original nonempty model
            // contract range; payload and expected seal remain private device
            // copies. No current directory or replacement model digest is read.
            unsafe {
                execute.launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|e| XlogError::Kernel(e.to_string()))
        })
    }
}

enum CapturedTensorDigest {
    Tensor {
        cells: Arc<TrackedCudaSlice<u64>>,
        offset: usize,
        // Native outputs are sealed by their producer before export.
        producer_sealed: bool,
    },
    // Range inputs retain the producer's raw/typed seals. Source and scalar
    // views retain the producer's private complete metadata snapshot.
    Publication(Arc<PreparedStepInputs>),
}

fn tensor_content_identity(
    tensor: &PreparedSemanticTensor,
) -> (SemanticTensorLayout, u64, u64, u64, usize) {
    (
        tensor.layout,
        tensor.logical_begin,
        tensor.logical_end,
        tensor.data,
        tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
    )
}

fn same_tensor_content_owner(
    expected: &PreparedSemanticTensor,
    actual: &PreparedSemanticTensor,
) -> bool {
    tensor_content_identity(expected) == tensor_content_identity(actual)
        && match (&expected.native_allocation, &actual.native_allocation) {
            (Some(expected), Some(actual)) => expected.same_allocation(actual),
            (None, None) => true,
            _ => false,
        }
}

impl StepContentStorage {
    fn feedback_origin_digest(
        &self,
        tensor: &PreparedSemanticTensor,
    ) -> Result<Option<CapturedTensorDigest>, SemanticTransitionError> {
        let bytes = tensor.source.as_ref().map_or(0, DeviceMemoryView::len) as u64;
        let end = tensor
            .data
            .checked_add(bytes)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        for feedback in &self.feedback {
            for (index, original) in feedback.content.tensors.iter().enumerate() {
                let source = original
                    .source
                    .as_ref()
                    .expect("native feedback output owner");
                let original_end = original
                    .data
                    .checked_add(source.len() as u64)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if bytes == 0 || tensor.data >= original_end || original.data >= end {
                    continue;
                }
                if tensor_content_identity(tensor) != tensor_content_identity(original) {
                    return Err(publication_input_error("native feedback aliases must retain their original type, shape, interval and roster coordinate"));
                }
                let TensorContentSeals::Captured(digests) = &feedback.content.seals else {
                    return Err(SemanticTransitionError::ObservationMismatch);
                };
                let CapturedTensorDigest::Tensor { cells, offset, .. } = &digests[index] else {
                    return Err(SemanticTransitionError::ObservationMismatch);
                };
                return Ok(Some(CapturedTensorDigest::Tensor {
                    cells: Arc::clone(cells),
                    offset: *offset,
                    producer_sealed: true,
                }));
            }
        }
        Ok(None)
    }
}

impl TensorContentBuffers {
    fn enqueue(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        execute: &CudaFunction,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        if let TensorContentSeals::Model(model) = &self.seals {
            if !verify || model.ranges.len() != self.tensors.len() + 1 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            model.verify_contract(domain, poisoned)?;
        }
        for (ordinal, tensor) in self.tensors.iter().enumerate() {
            if let TensorContentSeals::Captured(digests) = &self.seals {
                if let CapturedTensorDigest::Publication(inputs) = &digests[ordinal] {
                    if !digests[..ordinal].iter().any(|digest| matches!(digest,
                        CapturedTensorDigest::Publication(previous) if Arc::ptr_eq(inputs, previous))) {
                        inputs.verify(domain, poisoned)?;
                    }
                    // The full-capacity alias may have a shorter original
                    // logical interval. Its original device range is the basis.
                    continue;
                }
            }
            let (pointer, bytes) = tensor
                .source
                .as_ref()
                .map_or((0, 0), |source| (*source.device_ptr(), source.len()));
            let range = tensor_content_range(
                &tensor.layout,
                tensor.logical_begin,
                tensor.logical_end,
                bytes,
            )?;
            let mut recorder = domain.new_strict_recorder();
            if let Some(source) = &tensor.source {
                recorder.read(source);
            }
            let (expected, verify) = match &self.seals {
                TensorContentSeals::Captured(digests) => {
                    let CapturedTensorDigest::Tensor {
                        cells,
                        offset,
                        producer_sealed,
                    } = &digests[ordinal]
                    else {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    };
                    let verify = verify || *producer_sealed;
                    if verify {
                        recorder.read(cells.as_ref());
                    } else {
                        recorder.write(cells.as_ref());
                    }
                    if offset.checked_add(4).is_none_or(|end| end > cells.len()) {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    (
                        cells.device_ptr_value() + (*offset * size_of::<u64>()) as u64,
                        verify,
                    )
                }
                TensorContentSeals::Model(model) => {
                    recorder.read(&model.ranges);
                    (
                        model.ranges.device_ptr_value()
                            + (ordinal * size_of::<PublicationRange>()
                                + std::mem::offset_of!(PublicationRange, digest))
                                as u64,
                        true,
                    )
                }
            };
            let arguments = (
                pointer,
                bytes as u64,
                range,
                tensor.layout,
                expected,
                u64::from(verify),
            );
            let execute = execute.clone();
            enqueue_recorded(domain, poisoned, recorder, |enqueue| {
                // SAFETY: the original managed source and exact layout retain
                // their allocation; all four private digest cells or the
                // original copied publication rows are retained and recorded.
                unsafe {
                    execute.launch_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        arguments,
                    )
                }
                .map_err(|e| XlogError::Kernel(e.to_string()))
            })?;
        }
        Ok(())
    }
}

fn model_content_ranges(
    directory: &[PublicationRange],
    layouts: &BTreeMap<(u64, u64), SemanticTensorLayout>,
    tensors: impl Iterator<Item = (SemanticTensorLayout, u64, u64, usize)>,
) -> Result<Vec<usize>, SemanticTransitionError> {
    // This cold plan fixes geometry only. The original device reader selects
    // and authenticates the model record before its expected seals are copied.
    if directory.iter().filter(|range| range.role == 44).count() != 1
        || !directory.iter().any(|range| {
            range.role == 44 && range.index == 0 && range.generation == 1 && range.length_bytes != 0
        })
        || !directory.iter().any(|range| range.role == 18)
    {
        return Err(publication_input_error(
            "model content requires the complete original roster and sealed model contract",
        ));
    }
    let expected: BTreeSet<_> = directory
        .iter()
        .filter(|range| matches!(range.role, 18..=20))
        .map(|range| (range.role, range.index))
        .collect();
    let mut seen = BTreeSet::new();
    let mut ranges = Vec::new();
    for (actual, begin, end, bytes) in tensors {
        let key = (actual.role, actual.index);
        if !expected.contains(&key) || !seen.insert(key) {
            return Err(publication_input_error(
                "model content must name each original model tensor exactly once",
            ));
        }
        tensor_content_range(&actual, begin, end, bytes)?;
        let (ordinal, range) = directory
            .iter()
            .enumerate()
            .find(|(_, range)| (range.role, range.index) == key)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut original = *layouts
            .get(&key)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.logical_axis != u64::MAX {
            let axis = usize::try_from(original.logical_axis)
                .ok()
                .filter(|&axis| axis < 4)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            original.dimensions[axis] = range
                .logical_end
                .checked_sub(range.logical_begin)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
        }
        // Logical content includes type, shape and interval, but neither the
        // producer address nor its valid, independently checked physical strides.
        original.strides_bytes = actual.strides_bytes;
        if original != actual
            || range.generation != 1
            || (range.logical_begin, range.logical_end) != (begin, end)
        {
            return Err(publication_input_error(
                "live model tensor differs from the original logical type, shape or interval",
            ));
        }
        ranges.push(ordinal);
    }
    if seen != expected {
        return Err(publication_input_error(
            "model content omits an original model tensor",
        ));
    }
    Ok(ranges)
}

fn tensor_content_range(
    layout: &SemanticTensorLayout,
    begin: u64,
    end: u64,
    bytes: usize,
) -> Result<PublicationRange, SemanticTransitionError> {
    if (layout.role != 0 && SemanticStateRole::from_code(layout.role).is_none())
        || end < begin
        || tensor_layout_bytes(layout)? != bytes
        || if layout.logical_axis == u64::MAX {
            begin != 0 || end != 0
        } else {
            end - begin != layout.dimensions[layout.logical_axis as usize]
        }
    {
        return Err(publication_input_error(
            "content tensor must retain its exact type, shape, interval and byte extent",
        ));
    }
    Ok(PublicationRange {
        role: layout.role,
        index: layout.index,
        length_bytes: bytes as u64,
        logical_begin: begin,
        logical_end: end,
        ..PublicationRange::default()
    })
}

fn validate_content_coordinates(
    coordinates: impl Iterator<Item = (u64, u64)>,
) -> Result<(), SemanticTransitionError> {
    let mut distinct = BTreeSet::new();
    for (ordinal, (role, index)) in coordinates.enumerate() {
        if (role == 0 && index != ordinal as u64)
            || (role != 0 && SemanticStateRole::from_code(role).is_none())
            || !distinct.insert((role, index))
        {
            return Err(publication_input_error(
                "content coordinates require unique state roles or exact private capture positions",
            ));
        }
    }
    Ok(())
}

/// An actual device-acquired reader. Only its originating Session can inspect,
/// export or release it; it is neither cloneable nor externally constructible.
pub struct SemanticPublishedLease {
    issuer: Arc<()>,
    token: u64,
    identity: SemanticPublishedIdentity,
    header: PublicationHeader,
    source: [SemanticTextSlot; 32],
    directory: Vec<PublicationRange>,
    active: bool,
}

fn committed_prefix_layout(
    range: &PublicationRange,
    prefix_capacity: u64,
) -> Result<SemanticTensorLayout, SemanticTransitionError> {
    let row_bytes = size_of::<SemanticTextSlot>() as u64;
    if range.role != SemanticStateRole::PrefixSource as u64
        || range.index != 0
        || range.logical_begin != 0
        || range.logical_end > prefix_capacity
        || range.logical_end.checked_mul(row_bytes) != Some(range.length_bytes)
        || i64::try_from(prefix_capacity).is_err()
        || !range
            .offset_bytes
            .is_multiple_of(std::mem::align_of::<SemanticTextSlot>() as u64)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let layout = SemanticTensorLayout {
        role: SemanticStateRole::PrefixSource as u64,
        index: 0,
        scalar_type: 3,
        element_bytes: 8,
        rank: 2,
        logical_axis: 0,
        dimensions: [prefix_capacity, 8, 0, 0],
        strides_bytes: [row_bytes, 8, 0, 0],
    };
    tensor_layout_bytes(&layout)?;
    Ok(layout)
}

/// Bound a view using its cold geometry without widening the sealed logical
/// range. Only prefix/cache owners expose reserved row capacity. Other records
/// and model tensors remain bounded by their complete original sealed extent.
fn publication_export_span(
    range: &PublicationRange,
    layout: &SemanticTensorLayout,
    allocation_bytes: usize,
) -> Result<std::ops::Range<usize>, SemanticTransitionError> {
    let bytes = tensor_layout_bytes(layout)?;
    let begin = usize::try_from(range.offset_bytes)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    let sealed_bytes = usize::try_from(range.length_bytes)
        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
    let capacity = matches!(range.role, 1 | 4 | 5 | 51..=54);
    if range.generation != 1
        || (range.role, range.index) != (layout.role, layout.index)
        || !range.offset_bytes.is_multiple_of(layout.element_bytes)
        || begin
            .checked_add(sealed_bytes)
            .is_none_or(|end| end > allocation_bytes)
        || range.logical_end < range.logical_begin
        || (layout.logical_axis != u64::MAX
            && (range.logical_begin != 0
                || range.logical_end > layout.dimensions[layout.logical_axis as usize]))
        || (!capacity && bytes > sealed_bytes)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let end = begin
        .checked_add(bytes.max(sealed_bytes))
        .filter(|&end| end <= allocation_bytes)
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    Ok(begin..end)
}

fn publication_record_layout(range: &PublicationRange) -> SemanticTensorLayout {
    SemanticTensorLayout {
        role: range.role,
        index: range.index,
        element_bytes: 1,
        scalar_type: 1,
        rank: 1,
        logical_axis: u64::MAX,
        dimensions: [range.length_bytes, 0, 0, 0],
        strides_bytes: [1, 0, 0, 0],
    }
}

impl SemanticPublishedLease {
    fn model_context_header(&self) -> Result<PublicationHeader, SemanticTransitionError> {
        let header = self.header;
        if header.abi != 1
            || header.instance != self.identity.instance
            || header.publication_word != self.identity.word
            || header.sealed_epoch != self.identity.word >> 1
            || header.logical_digest != self.identity.logical_digest
            || header.state_digest != self.identity.state_digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(header)
    }
}

/// Value-independent flat coordinates emitted by the native feedback producer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticFeedbackSchema {
    pub identity: [u8; 32],
    pub statement_bytes: usize,
    pub feature_width: usize,
}

impl SemanticFeedbackSchema {
    pub fn new(statements: [&[u8]; 3]) -> Result<Self, SemanticTransitionError> {
        if statements.iter().any(|statement| statement.is_empty()) {
            return Err(publication_input_error(
                "native feedback requires three nonempty typed statements",
            ));
        }
        let statement_bytes = statements
            .iter()
            .map(|statement| statement.len())
            .max()
            .unwrap();
        let feature_width = statement_bytes
            .checked_mul(9)
            .and_then(|n| n.checked_add(4))
            .filter(|&n| i64::try_from(n).is_ok())
            .ok_or_else(|| publication_input_error("native feedback schema width overflows"))?;
        let mut hash = Sha256::new();
        hash.update(b"xlog.feedback.flat-bit-presence.v1\0");
        hash.update((statement_bytes as u64).to_le_bytes());
        hash.update(b"byte[i]:bits[0..8],presence;pro,1,contra,1;invalid:all-zero");
        Ok(Self {
            identity: hash.finalize().into(),
            statement_bytes,
            feature_width,
        })
    }
}

/// Named columns of each current feedback support row. These service values do
/// not enter the learned feature schema. Rows follow newest-to-oldest ancestry.
pub const SEMANTIC_FEEDBACK_SUPPORT_FIELDS: [&str; 12] = [
    "original_statement",
    "original_support",
    "support_slot",
    "support_generation",
    "version_slot",
    "version_generation",
    "insertion_ordinal",
    "polarity",
    "support_digest_0",
    "support_digest_1",
    "support_digest_2",
    "support_digest_3",
];

/// Features, validity and current causal provenance from one acquired bank.
/// Original outputs retain their step independently of later bank retirement.
pub struct SemanticFeedback {
    pub parent: SemanticPublishedIdentity,
    pub schema: SemanticFeedbackSchema,
    features: DlpackManagedTensor,
    device_validity: DlpackManagedTensor,
    query_receipts: DlpackManagedTensor,
    original_statements: DlpackManagedTensor,
    support_offsets: DlpackManagedTensor,
    supports: DlpackManagedTensor,
}

impl SemanticFeedback {
    /// Features, validity, opaque native current receipts, original statement
    /// indices, support offsets, and named support rows, respectively. All
    /// service tensors are U64. Invalid slots have no receipt/contributors and
    /// original index U64::MAX; valid Neither has an original index and no rows.
    pub fn into_dlpack(self) -> [DlpackManagedTensor; 6] {
        [
            self.features,
            self.device_validity,
            self.query_receipts,
            self.original_statements,
            self.support_offsets,
            self.supports,
        ]
    }
}

struct FeedbackBuffers {
    schema: SemanticFeedbackSchema,
    slots: usize,
    support_capacity: usize,
    encode: CudaFunction,
    project: CudaFunction,
    witness: CudaFunction,
    content: TensorContentBuffers,
    features: TrackedCudaSlice<f32>,
    validity: TrackedCudaSlice<u8>,
    status: TrackedCudaSlice<u64>,
    query_receipts: TrackedCudaSlice<u64>,
    original_statements: TrackedCudaSlice<u64>,
    support_offsets: TrackedCudaSlice<u64>,
    supports: TrackedCudaSlice<u64>,
}

#[derive(Clone, Copy)]
struct FeedbackAllocationPlan {
    slots: usize,
    support_capacity: usize,
    cells: usize,
    query_cells: usize,
    support_cells: usize,
    offset_cells: usize,
    bytes: usize,
}

impl FeedbackAllocationPlan {
    fn new(
        schema: &SemanticFeedbackSchema,
        slots: u64,
        support_capacity: u64,
    ) -> Result<Self, SemanticTransitionError> {
        let slots = usize::try_from(slots)
            .ok()
            .filter(|&n| n >= 2 && i64::try_from(n).is_ok())
            .ok_or_else(|| {
                publication_input_error("feedback slot capacity is not representable")
            })?;
        let support_capacity = usize::try_from(support_capacity)
            .ok()
            .filter(|&n| n != 0 && i64::try_from(n).is_ok())
            .ok_or_else(|| {
                publication_input_error("feedback support capacity is not representable")
            })?;
        let cells = slots
            .checked_mul(schema.feature_width)
            .ok_or_else(|| publication_input_error("feedback feature extent overflows"))?;
        let query_cells = slots
            .checked_mul(42)
            .ok_or_else(|| publication_input_error("feedback receipt extent overflows"))?;
        let support_cells = support_capacity
            .checked_mul(SEMANTIC_FEEDBACK_SUPPORT_FIELDS.len())
            .ok_or_else(|| publication_input_error("feedback support extent overflows"))?;
        let offset_cells = slots
            .checked_add(1)
            .ok_or_else(|| publication_input_error("feedback offset extent overflows"))?;
        let service_bytes = query_cells
            .checked_add(slots)
            .and_then(|n| n.checked_add(offset_cells))
            .and_then(|n| n.checked_add(support_cells))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| publication_input_error("feedback service extent overflows"))?;
        let bytes = cells
            .checked_mul(size_of::<f32>())
            .and_then(|n| {
                slots
                    .checked_mul(9)
                    .and_then(|metadata| n.checked_add(metadata))
            })
            .and_then(|n| n.checked_add(service_bytes))
            .and_then(|n| n.checked_add(6 * 4 * size_of::<u64>()))
            .ok_or_else(|| publication_input_error("feedback allocation extent overflows"))?;
        Ok(Self {
            slots,
            support_capacity,
            cells,
            query_cells,
            support_cells,
            offset_cells,
            bytes,
        })
    }
}

impl FeedbackBuffers {
    // Cold preparation only. The same fixed outputs and kernels are used by
    // enqueue; no acquired parent, device values or digest is cached here.
    fn allocate(
        provider: &CudaKernelProvider,
        schema: SemanticFeedbackSchema,
        slots: u64,
        support_capacity: u64,
    ) -> Result<Self, SemanticTransitionError> {
        let plan = FeedbackAllocationPlan::new(&schema, slots, support_capacity)?;
        let mut reservation = provider
            .memory()
            .reserve_bytes(plan.bytes as u64)
            .map_err(|error| runtime_error("feedback reservation", error))?;
        Self::allocate_reserved(provider, schema, plan, &mut reservation)
    }

    fn allocate_reserved(
        provider: &CudaKernelProvider,
        schema: SemanticFeedbackSchema,
        plan: FeedbackAllocationPlan,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<Self, SemanticTransitionError> {
        let FeedbackAllocationPlan {
            slots,
            support_capacity,
            cells,
            query_cells,
            support_cells,
            offset_cells,
            ..
        } = plan;
        let encode = provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_feedback_encode")
            .ok_or_else(|| {
                runtime_error("kernel lookup", "semantic feedback kernel unavailable")
            })?;
        let project = provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_feedback_project")
            .ok_or_else(|| {
                runtime_error("kernel lookup", "semantic feedback projection unavailable")
            })?;
        let witness = provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_tensor_content_witness",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "tensor content witness unavailable"))?;
        let mut buffers = Self {
            schema,
            slots,
            support_capacity,
            encode,
            project,
            witness,
            content: TensorContentBuffers {
                tensors: Vec::new(),
                seals: TensorContentSeals::Captured(Vec::new()),
                verification_inputs: Vec::new(),
            },
            features: reservation
                .alloc::<f32>(cells)
                .map_err(|error| runtime_error("feedback features allocation", error))?,
            validity: reservation
                .alloc::<u8>(slots)
                .map_err(|error| runtime_error("feedback validity allocation", error))?,
            status: reservation
                .alloc::<u64>(slots)
                .map_err(|error| runtime_error("feedback status allocation", error))?,
            query_receipts: reservation
                .alloc::<u64>(query_cells)
                .map_err(|error| runtime_error("feedback receipt allocation", error))?,
            original_statements: reservation
                .alloc::<u64>(slots)
                .map_err(|error| runtime_error("feedback origin allocation", error))?,
            support_offsets: reservation
                .alloc::<u64>(offset_cells)
                .map_err(|error| runtime_error("feedback offset allocation", error))?,
            supports: reservation
                .alloc::<u64>(support_cells)
                .map_err(|error| runtime_error("feedback support allocation", error))?,
        };
        let words = |view: DeviceMemoryView<u64>| {
            // SAFETY: u64 has no padding. The byte view retains the original
            // tracked allocation, without a managed alias that owns its reader.
            unsafe { view.cast::<u8>() }.ok_or(SemanticTransitionError::ObservationMismatch)
        };
        // SAFETY: f32 has no padding and the byte view retains its real owner.
        let features = unsafe { buffers.features.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let originals = [
            features,
            buffers.validity.view(),
            words(buffers.query_receipts.view())?,
            words(buffers.original_statements.view())?,
            words(buffers.support_offsets.view())?,
            words(buffers.supports.view())?,
        ];
        let shapes = [
            (
                6,
                4,
                2,
                [slots as u64, buffers.schema.feature_width as u64, 0, 0],
            ),
            (8, 1, 1, [slots as u64, 0, 0, 0]),
            (3, 8, 2, [slots as u64, 42, 0, 0]),
            (3, 8, 1, [slots as u64, 0, 0, 0]),
            (3, 8, 1, [offset_cells as u64, 0, 0, 0]),
            (3, 8, 2, [support_capacity as u64, 12, 0, 0]),
        ];
        let mut digests = Vec::with_capacity(originals.len());
        for (index, (source, (scalar_type, element_bytes, rank, dimensions))) in
            originals.into_iter().zip(shapes).enumerate()
        {
            let layout = canonical_tensor_layout(SemanticTensorLayout {
                role: 0,
                index: index as u64,
                scalar_type,
                element_bytes,
                rank,
                dimensions,
                logical_axis: u64::MAX,
                strides_bytes: [0; 4],
            })?;
            tensor_content_range(&layout, 0, 0, source.len())?;
            buffers.content.tensors.push(PreparedSemanticTensor {
                layout,
                logical_begin: 0,
                logical_end: 0,
                data: *source.device_ptr(),
                source: Some(source),
                native_allocation: None,
                _empty_owner: None,
            });
            digests.push(CapturedTensorDigest::Tensor {
                cells: Arc::new(
                    reservation
                        .alloc::<u64>(4)
                        .map_err(|error| runtime_error("feedback witness allocation", error))?,
                ),
                offset: 0,
                producer_sealed: false,
            });
        }
        buffers.content.seals = TensorContentSeals::Captured(digests);
        Ok(buffers)
    }

    // The caller retains these prepared outputs before the first may-enqueue
    // boundary, including failures and unwinding. This path does not allocate
    // device storage, export tensors, synchronize or insert stream joins.
    #[expect(
        clippy::too_many_arguments,
        reason = "feedback execution records every retained native owner explicitly"
    )]
    fn enqueue(
        &self,
        domain: &ResidentExecutionDomain,
        poisoned: &mut bool,
        graph: &SemanticHypergraph,
        storage: &PublicationStorage,
        task: &TrackedCudaSlice<u64>,
        lease: &TrackedCudaSlice<PublicationLease>,
        descriptor: Descriptor,
    ) -> Result<(), SemanticTransitionError> {
        let mut recorder = domain.new_strict_recorder();
        recorder.read(lease);
        recorder.read(&storage.control);
        recorder.read(&storage.storage);
        recorder.read(&storage.contract);
        for bank in &storage.banks {
            recorder.read(bank);
        }
        for directory in &storage.directories {
            recorder.read(directory);
        }
        for allocation in &storage.allocations {
            recorder.read(allocation);
        }
        recorder.write(&self.features);
        recorder.write(&self.validity);
        recorder.write(&self.status);
        let arguments = (
            storage.control.device_ptr_value(),
            lease.device_ptr_value(),
            self.schema.statement_bytes as u64,
            self.slots as u64,
            self.features.device_ptr_value(),
            self.validity.device_ptr_value(),
            self.status.device_ptr_value(),
        );
        enqueue_recorded(domain, poisoned, recorder, |stream| {
            // SAFETY: both directories and every selectable allocation are
            // retained. The kernel resolves and validates the acquired lease's
            // original ranges; output extents were checked during preparation.
            unsafe {
                self.encode.clone().launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (self.slots.min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        let mut recorder = domain.new_strict_recorder();
        graph.record_transition(&mut recorder);
        storage.record(&mut recorder);
        recorder.read(lease);
        recorder.read(task);
        recorder.read_write(&self.status);
        recorder.write(&self.query_receipts);
        recorder.write(&self.original_statements);
        recorder.write(&self.support_offsets);
        recorder.write(&self.supports);
        let arguments = (
            descriptor,
            lease.device_ptr_value(),
            self.slots as u64,
            self.query_receipts.device_ptr_value(),
            self.original_statements.device_ptr_value(),
            self.support_offsets.device_ptr_value(),
            self.supports.device_ptr_value(),
            self.support_capacity as u64,
            self.status.device_ptr_value(),
        );
        enqueue_recorded(domain, poisoned, recorder, |stream| {
            // SAFETY: the original task, actual arena and publication owners
            // are recorded. Lineage is queried from this device-acquired parent,
            // never a captured host selection or detached feedback record.
            unsafe {
                self.project.clone().launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        // The first seal follows the genuine encoder and lineage producer on
        // the same stream, before any dependent consumer receives an alias.
        self.content.enqueue(domain, poisoned, &self.witness, false)
    }
}

impl SemanticPublishedLease {
    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "DLPack export retains independent shape, type, alias, and publication owners"
)]
fn export_owned_allocation(
    view: DeviceMemoryView<u8>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: (u8, u8),
    device_id: i32,
    alias_guard: Arc<()>,
    publication: Arc<PublicationStorage>,
    continuation: Option<Arc<TextBindingStorage>>,
    retained_owner: Option<Arc<dyn Send + Sync>>,
) -> DlpackManagedTensor {
    let mut export = Box::new(PublishedTensorExport {
        managed: crate::dlpack::DLManagedTensor {
            dl_tensor: crate::dlpack::DLTensor {
                data: *view.device_ptr() as usize as *mut std::ffi::c_void,
                device: crate::dlpack::DLDevice {
                    device_type: 2,
                    device_id,
                },
                ndim: shape.len() as i32,
                dtype: crate::dlpack::DLDataType {
                    code: dtype.0,
                    bits: dtype.1,
                    lanes: 1,
                },
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(delete_published_tensor),
        },
        shape,
        strides,
        _view: view,
        _alias_guard: alias_guard,
        _publication: publication,
        _continuation: continuation,
        _retained_owner: retained_owner,
    });
    export.managed.dl_tensor.shape = export.shape.as_mut_ptr();
    export.managed.dl_tensor.strides = export.strides.as_mut_ptr();
    let owner = Box::into_raw(export);
    // SAFETY: the stable owner contains the genuine native allocation and its
    // exact metadata. Its bank or step keeps storage through final completion.
    unsafe {
        (*owner).managed.manager_ctx = owner.cast();
        DlpackManagedTensor::from_raw(&mut (*owner).managed)
    }
}

struct PublishedTensorExport {
    managed: crate::dlpack::DLManagedTensor,
    shape: Vec<i64>,
    strides: Vec<i64>,
    _view: DeviceMemoryView<u8>,
    _alias_guard: Arc<()>,
    _publication: Arc<PublicationStorage>,
    _continuation: Option<Arc<TextBindingStorage>>,
    _retained_owner: Option<Arc<dyn Send + Sync>>,
}

unsafe extern "C" fn delete_published_tensor(managed: *mut crate::dlpack::DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: this callback owns exactly the Box installed below. DLPack
    // transfers its managed record once; no unrelated pointer is accepted.
    let owner = unsafe { (*managed).manager_ctx.cast::<PublishedTensorExport>() };
    if !owner.is_null() {
        drop(unsafe { Box::from_raw(owner) });
    }
}

fn dlpack_consumer_stream(stream: u64) -> Result<sys::CUstream, SemanticTransitionError> {
    if matches!(stream, 0 | 2) {
        return Err(publication_input_error("DLPack consumer requires legacy default stream 1 or an explicit live stream; zero and per-thread default are unsupported"));
    }
    Ok(if stream == 1 {
        std::ptr::null_mut()
    } else {
        stream as usize as sys::CUstream
    })
}

// Every call below uses the fixed integral, padding-free publication ABI checked
// above, or SourceSlot's checked eight-u64 layout. This is cold control metadata.
fn publication_abi_bytes<T: DeviceRepr>(records: &[T]) -> Vec<u8> {
    // SAFETY: callers use initialized padding-free C ABI records only. The byte
    // slice is bounded by the live input allocation and copied before return.
    unsafe { std::slice::from_raw_parts(records.as_ptr().cast(), std::mem::size_of_val(records)) }
        .to_vec()
}

fn initial_intent_records(
    effect: &[u8],
    entry_capacity: u64,
    payload_capacity: usize,
    final_payload_bytes: u64,
) -> Result<[SemanticStateRecord; 2], SemanticTransitionError> {
    let entries_bytes = usize::try_from(entry_capacity)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<IntentEntry>()))
        .and_then(|bytes| bytes.checked_add(size_of::<IntentQueueHeader>()))
        .ok_or_else(|| {
            publication_input_error("intent entry capacity overflows its native allocation")
        })?;
    let required_payload = u64::try_from(effect.len())
        .ok()
        .and_then(|bytes| bytes.checked_add(final_payload_bytes));
    if effect.is_empty()
        || entry_capacity == 0
        || final_payload_bytes == 0
        || required_payload.is_none_or(|bytes| bytes > payload_capacity as u64)
    {
        return Err(publication_input_error(
            "initial intent requires its actual effect and reserved final-output capacity",
        ));
    }
    let header = IntentQueueHeader {
        abi: 1,
        count: 0,
        capacity: entry_capacity,
        payload_used_bytes: effect.len() as u64,
        payload_capacity_bytes: payload_capacity as u64,
        effect_offset_bytes: 0,
        effect_length_bytes: effect.len() as u64,
        chain_head: Identity256::default(),
    };
    Ok([
        SemanticStateRecord {
            role: SemanticStateRole::IntentEntries,
            index: 0,
            bytes: publication_abi_bytes(&[header]),
            capacity_bytes: entries_bytes,
        },
        SemanticStateRecord {
            role: SemanticStateRole::IntentPayload,
            index: 0,
            bytes: effect.to_vec(),
            capacity_bytes: payload_capacity,
        },
    ])
}

fn publication_mutable_role(role: u64) -> bool {
    matches!(role, 3..=15 | 18..=25 | 30 | 33 | 39 | 44 | 51..=55)
}

fn continuation_role(role: u64) -> bool {
    matches!(role, 1 | 4..=13 | 39 | 51..=55)
}

fn validate_parent_records(
    parent: &SemanticParentBinding,
    tensors: &[PreparedSemanticTensor],
) -> Result<(), SemanticTransitionError> {
    validate_publication_counts(&parent.role_counts)?;
    validate_text_parent(
        &parent.source,
        parent.prefix.len() as u64,
        parent.prefix_capacity,
        parent.feedback_capacity,
        parent.max_position,
        parent.ring_head,
        parent.provenance_records,
    )?;
    if parent.feedback_capacity < 3
        || parent.rng.stream_serial >= 1 << 56
        || u64::from(parent.rng.model_generation) != parent.model_generation
        || parent.pad_token >= TEXT_CARDINALITY as u64
        || parent.terminal_tokens.is_empty()
        || parent
            .terminal_tokens
            .iter()
            .any(|&token| token >= TEXT_CARDINALITY as u64)
        || parent
            .terminal_tokens
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != parent.terminal_tokens.len()
    {
        return Err(publication_input_error(
            "parent generations, feedback extent, or token contract mismatch",
        ));
    }
    for (position, slot) in parent.prefix.iter().enumerate() {
        if slot.kind != 1
            || slot.valid != 1
            || slot.committed != 1
            || slot.logical_position != position as u64
            || slot.token >= TEXT_CARDINALITY as u64
            || !matches!(slot.provenance, 1 | 2)
            || slot.provenance_record >= parent.provenance_records
            || slot.recomputed > 1
        {
            return Err(publication_input_error(
                "committed prefix source is not the exact ordered valid prefix",
            ));
        }
    }
    let mut keys = BTreeSet::new();
    for record in &parent.records {
        let role = record.role as u64;
        if is_tensor_role(role)
            || matches!(role, 1 | 3 | 14..=16 | 33 | 43 | 47 | 48 | 55)
            || record.index >= parent.role_counts[role as usize - 1]
            || (record.bytes.is_empty() && role != 2)
            || record.bytes.len() > record.capacity_bytes
            || !keys.insert((role, record.index))
        {
            return Err(publication_input_error(
                "duplicate, generated, tensor, empty, or out-of-capacity control record",
            ));
        }
        if role == 2
            && (parent
                .provenance_records
                .checked_mul(size_of::<TokenProvenance>() as u64)
                != Some(record.bytes.len() as u64)
                || (parent.provenance_records + 32)
                    .checked_mul(size_of::<TokenProvenance>() as u64)
                    .is_none_or(|n| n > record.capacity_bytes as u64))
        {
            return Err(publication_input_error(
                "token provenance records or append capacity differ from the actual source ledger",
            ));
        }
        if role == 44 {
            parent
                .model_contract_layout
                .validate(record.bytes.len() as u64)?;
        }
        if role == 30 {
            if record.bytes.len() < size_of::<IntentQueueHeader>() {
                return Err(publication_input_error(
                    "intent queue has no complete owner header",
                ));
            }
            // SAFETY: complete checked header bytes, integral padding-free ABI.
            let header = unsafe {
                record
                    .bytes
                    .as_ptr()
                    .cast::<IntentQueueHeader>()
                    .read_unaligned()
            };
            if header.abi != 1
                || header.count > header.capacity
                || header
                    .count
                    .checked_mul(size_of::<IntentEntry>() as u64)
                    .and_then(|n| n.checked_add(size_of::<IntentQueueHeader>() as u64))
                    != Some(record.bytes.len() as u64)
                || header
                    .capacity
                    .checked_mul(size_of::<IntentEntry>() as u64)
                    .and_then(|n| n.checked_add(size_of::<IntentQueueHeader>() as u64))
                    != Some(record.capacity_bytes as u64)
                || header.payload_used_bytes > header.payload_capacity_bytes
                || header.effect_length_bytes == 0
                || header
                    .effect_offset_bytes
                    .checked_add(header.effect_length_bytes)
                    .is_none_or(|n| n > header.payload_used_bytes)
            {
                return Err(publication_input_error(
                    "intent queue count, capacity, or actual effect payload is inconsistent",
                ));
            }
            let payload = parent
                .records
                .iter()
                .find(|record| record.role == SemanticStateRole::IntentPayload)
                .ok_or_else(|| publication_input_error("intent queue payload owner is absent"))?;
            if payload.bytes.len() as u64 != header.payload_used_bytes
                || payload.capacity_bytes as u64 != header.payload_capacity_bytes
            {
                return Err(publication_input_error(
                    "intent queue does not resolve to its actual payload allocation",
                ));
            }
        }
    }
    for tensor in tensors {
        let layout = &tensor.layout;
        if !is_tensor_role(layout.role)
            || layout.role >= 51
            || layout.index >= parent.role_counts[layout.role as usize - 1]
            || !keys.insert((layout.role, layout.index))
        {
            return Err(publication_input_error("initial parent tensor role is duplicate, uncomputed-active, or outside its model roster"));
        }
        tensor_layout_bytes(layout)?;
        if matches!(layout.role, 4 | 5) {
            if tensor.logical_begin != 0
                || tensor.logical_end != parent.prefix.len() as u64
                || layout.dimensions[2] != parent.prefix.len() as u64
            {
                return Err(publication_input_error(
                    "initial attention tensor does not cover exactly the acquired prefix",
                ));
            }
            prefix_capacity_layout(*layout, parent.prefix_capacity)?;
        } else if layout.logical_axis != u64::MAX
            || (matches!(layout.role, 18..=25)
                && (tensor.logical_begin != 0 || tensor.logical_end != 0))
        {
            return Err(publication_input_error(
                "non-positional model tensors require a non-sliced exact layout",
            ));
        }
        if matches!(layout.role, 8..=13) && (layout.scalar_type != 6 || layout.element_bytes != 4) {
            return Err(publication_input_error(
                "value and edit accumulators require original FP32 storage",
            ));
        }
    }
    for layout in &parent.active_layouts {
        if !matches!(layout.role, 51..=54)
            || layout.index >= parent.role_counts[layout.role as usize - 1]
            || !keys.insert((layout.role, layout.index))
            || tensor_layout_bytes(layout)? == 0
        {
            return Err(publication_input_error("active output capacity layout is absent, duplicate, or outside the cold model roster"));
        }
        validate_active_capacity(
            layout,
            32u64
                .checked_add(parent.feedback_capacity)
                .ok_or_else(|| publication_input_error("active row capacity overflow"))?,
        )?;
    }
    for role in 1..=55u64 {
        if matches!(role, 1 | 3 | 14..=16 | 33 | 43 | 47 | 48 | 55) {
            continue;
        }
        for index in 0..parent.role_counts[role as usize - 1] {
            if !keys.contains(&(role, index)) {
                return Err(publication_input_error(
                    "publication role lacks its actual cold owner",
                ));
            }
        }
    }
    Ok(())
}

/// Validate the initial cold text import and derive its structural interval.
/// This establishes source metadata, not model completion or cache coverage.
fn validate_text_parent(
    slots: &[SourceSlot; 32],
    prefix_extent: u64,
    prefix_capacity: u64,
    feedback_capacity: u64,
    max_position: u64,
    ring_head: u64,
    provenance_records: u64,
) -> Result<u64, SemanticTransitionError> {
    let invalid = |detail: &str| SemanticTransitionError::InvalidInput {
        detail: detail.into(),
    };
    let text_limit = prefix_capacity
        .checked_add(32)
        .ok_or_else(|| invalid("text capacity overflow"))?;
    if prefix_extent > prefix_capacity
        || ring_head >= 32
        || text_limit
            .checked_add(feedback_capacity)
            .is_none_or(|limit| limit > max_position)
        || max_position != 262144
    {
        return Err(invalid("text prefix, ring or position capacity mismatch"));
    }
    let mut positions = BTreeSet::new();
    let mut filled = BTreeSet::new();
    for slot in slots {
        if slot.kind > 2
            || (if slot.kind == 2 {
                slot.token != MASK_TOKEN as u64
            } else {
                slot.token >= TEXT_CARDINALITY as u64
            })
        {
            return Err(invalid(
                "text token or row kind is outside its cold contract",
            ));
        }
        if slot.kind == 0 {
            if slot.logical_position != 0
                || slot.provenance != 0
                || slot.valid != 0
                || slot.committed != 0
                || slot.recomputed != 0
                || slot.provenance_record != 0
            {
                return Err(invalid("padding source slot carries active metadata"));
            }
            continue;
        }
        if slot.valid != 1
            || slot.committed != 0
            || slot.recomputed > 1
            || slot.logical_position < prefix_extent
            || slot.logical_position >= text_limit
            || !positions.insert(slot.logical_position)
        {
            return Err(invalid(
                "active source slot has inconsistent position or provenance",
            ));
        }
        if slot.kind == 1 {
            if !matches!(slot.provenance, 1 | 2) || slot.provenance_record >= provenance_records {
                return Err(invalid(
                    "filled source slot requires source or generated provenance",
                ));
            }
            filled.insert(slot.logical_position);
        } else if slot.provenance != 0 || slot.provenance_record != 0 || slot.recomputed != 0 {
            return Err(invalid(
                "mask source slot cannot carry filled-token coverage",
            ));
        }
    }
    let mut structural_end = prefix_extent;
    while filled.contains(&structural_end) {
        if structural_end == prefix_capacity {
            return Err(invalid(
                "contiguous source completion exceeds prefix capacity",
            ));
        }
        structural_end += 1;
    }
    Ok(structural_end)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Descriptor {
    logits: u64,
    support: u64,
    scratch: u64,
    receipts: u64,
    state: u64,
    components: u64,
    codebooks: u64,
    arena: [u64; 7],
    policy: PolicyDescriptor,
    backward: PolicyBackward,
    task: u64,
    publication: PublicationCommand,
    text: TextBinding,
    model_work: ModelWorkInput,
}

#[derive(Clone, Copy)]
struct TransitionKernelPointers {
    logits: u64,
    support: u64,
    receipts: u64,
    state: u64,
    lease: u64,
    text: TextBinding,
    policy: PolicyDescriptor,
    model_work: ModelWorkInput,
}

impl Descriptor {
    fn with_transition(mut self, original: TransitionKernelPointers) -> Self {
        self.logits = original.logits;
        self.support = original.support;
        self.receipts = original.receipts;
        self.state = original.state;
        self.publication.lease = original.lease;
        self.text = original.text;
        self.policy = original.policy;
        self.model_work = original.model_work;
        self
    }
}

struct TransitionKernelIo<'a> {
    logits: &'a TrackedCudaSlice<f32>,
    support: &'a TrackedCudaSlice<u8>,
    receipts: &'a TrackedCudaSlice<SemanticTransitionReceipt>,
    state: &'a TrackedCudaSlice<DeviceState>,
    text: Option<&'a TextBindingStorage>,
    lease: Option<&'a TrackedCudaSlice<PublicationLease>>,
    model_work: Option<&'a PreparedModelWork>,
    model_update: Option<&'a PreparedModelUpdate>,
    training_selection: Option<&'a DeviceMemoryView<SemanticTrainingViewSelection>>,
    #[cfg(feature = "semantic-policy")]
    policy: Option<&'a PolicyStorage>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PolicyField {
    embeddings: u64,
    biases: u64,
    cardinality: u32,
    null_category: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PolicyDescriptor {
    z: u64,
    recurrence: u64,
    positions: u64,
    hidden: u64,
    scores: u64,
    recurrent: u64,
    fields: [PolicyField; 18],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PolicyBackward {
    cotangents: u64,
    parameters: u64,
    text: u64,
    recurrent: u64,
    scores: u64,
    status: u64,
    parameter_cells: u64,
    text_cells: u64,
}

#[cfg(feature = "semantic-policy")]
fn record_policy_vjp(
    domain: &ResidentExecutionDomain,
    poisoned: &mut bool,
    recorder: LaunchRecorder,
    execute: CudaFunction,
    descriptor: Descriptor,
    coefficients: &TrackedCudaSlice<f64>,
    score_cotangents: &DeviceMemoryView<f64>,
) -> Result<(), SemanticTransitionError> {
    enqueue_recorded(domain, poisoned, recorder, |enqueue| {
        // SAFETY: the recorder retains the original cotangent producer and
        // every forward/output allocation. The destination is the fixed F64
        // bank owned by this exact invocation. This routine deliberately makes
        // no assumption about capture state: its caller supplies the resident
        // execution domain whose stream owns both operations.
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                coefficients.device_ptr_value(),
                *score_cotangents.device_ptr(),
                COMPONENT_COUNT * 8,
                enqueue.stream().cu_stream(),
            )
            .result()
            .map_err(|error| XlogError::Kernel(format!("policy cotangent snapshot: {error}")))?;
            execute
                .launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (descriptor,),
                )
                .map_err(|error| XlogError::Kernel(format!("policy backward launch: {error}")))?;
        }
        Ok::<(), XlogError>(())
    })
}

/// Dedicated device-owned adjoints for one exact policy invocation. Their tracked
/// allocations can be consumed on another recorded stream without host copies.
#[cfg(feature = "semantic-policy")]
pub struct SemanticPolicyGradients {
    pub binding: SemanticCatalogueBinding,
    pub invocation: SemanticRngBinding,
    pub layout: SemanticPolicyLayout,
    pub parameters: TrackedCudaSlice<f32>,
    pub text_logits: TrackedCudaSlice<f32>,
    provider: Arc<CudaKernelProvider>,
    step_aliases: Arc<()>,
    publication: Arc<PublicationStorage>,
    continuation: Arc<TextBindingStorage>,
    vjp_workspace: Arc<PolicyVjpWorkspace>,
    support_cells: usize,
}

#[cfg(feature = "semantic-policy")]
impl SemanticPolicyGradients {
    /// Export the original parameter and full MASK adjoints, in that order.
    /// Backward already ordered the caller's explicit consumer stream after the
    /// device error guard. Final step release joins that consumer's last use.
    pub fn into_dlpack(self) -> Result<[DlpackManagedTensor; 2], SemanticTransitionError> {
        let layouts = policy_producer_layouts(self.support_cells, self.layout.parameter_cells)?;
        let retained_owner: Arc<dyn Send + Sync> = self.vjp_workspace;
        let export = |values: TrackedCudaSlice<f32>, layout: SemanticTensorLayout| {
            let rank = layout.rank as usize;
            let shape = layout.dimensions[..rank]
                .iter()
                .map(|&value| value as i64)
                .collect();
            let strides = layout.strides_bytes[..rank]
                .iter()
                .map(|&value| (value / layout.element_bytes) as i64)
                .collect();
            export_owned_allocation(
                values.into_bytes().view(),
                shape,
                strides,
                (2, 32),
                self.provider.device().ordinal() as i32,
                Arc::clone(&self.step_aliases),
                Arc::clone(&self.publication),
                Some(Arc::clone(&self.continuation)),
                Some(Arc::clone(&retained_owner)),
            )
        };
        Ok([
            export(self.parameters, layouts[2]),
            export(self.text_logits, layouts[0]),
        ])
    }
}

#[cfg(feature = "semantic-policy")]
fn policy_producer_layouts(
    support_cells: usize,
    parameter_cells: usize,
) -> Result<[SemanticTensorLayout; 4], SemanticTransitionError> {
    let layout = |index, scalar_type, element_bytes, rank, dimensions| {
        canonical_tensor_layout(SemanticTensorLayout {
            role: 0,
            index,
            scalar_type,
            element_bytes,
            rank,
            dimensions,
            logical_axis: u64::MAX,
            strides_bytes: [0; 4],
        })
    };
    Ok([
        layout(0, 6, 4, 2, [32, TEXT_CARDINALITY as u64, 0, 0])?,
        layout(1, 8, 1, 1, [support_cells as u64, 0, 0, 0])?,
        layout(2, 6, 4, 1, [parameter_cells as u64, 0, 0, 0])?,
        layout(3, 6, 4, 2, [1, COMPONENT_COUNT as u64, 0, 0])?,
    ])
}

#[cfg(feature = "semantic-policy")]
struct PolicyBuffers {
    layout: SemanticPolicyLayout,
    parameters: TrackedCudaSlice<f32>,
    hidden: TrackedCudaSlice<f32>,
    scores: TrackedCudaSlice<f32>,
    recurrent: TrackedCudaSlice<f32>,
    text_logits: TrackedCudaSlice<f32>,
    component_baselines: TrackedCudaSlice<f32>,
    adjoints: Option<PolicyAdjointBuffers>,
}

/// Reserved with the original forward tape, then consumed by its sole backward.
/// Output storage is never recycled into a later invocation's working banks.
#[cfg(feature = "semantic-policy")]
struct PolicyAdjointBuffers {
    parameters: TrackedCudaSlice<f32>,
    text_logits: TrackedCudaSlice<f32>,
    recurrent: TrackedCudaSlice<f32>,
    scores: TrackedCudaSlice<f32>,
    coefficients: TrackedCudaSlice<f64>,
    status: TrackedCudaSlice<u64>,
}

/// Every allocation referenced by a recorded policy VJP but not exported as an
/// adjoint. Exported gradient owners retain this workspace so deferred CUDA work
/// cannot observe reclaimed scratch or producer memory.
#[cfg(feature = "semantic-policy")]
struct PolicyVjpWorkspace {
    _recurrent: TrackedCudaSlice<f32>,
    _scores: TrackedCudaSlice<f32>,
    coefficients: TrackedCudaSlice<f64>,
    _status: TrackedCudaSlice<u64>,
    score_cotangents: DeviceMemoryView<f64>,
}

#[cfg(feature = "semantic-policy")]
struct PolicyStorage {
    buffers: PolicyBuffers,
    text_binding: Arc<TextBindingStorage>,
    replacement: Option<PolicyReplacement>,
    tape_live: bool,
}

#[cfg(feature = "semantic-policy")]
impl std::ops::Deref for PolicyStorage {
    type Target = PolicyBuffers;
    fn deref(&self) -> &Self::Target {
        &self.buffers
    }
}

#[cfg(feature = "semantic-policy")]
struct PolicyProducerViews {
    text_logits: DeviceMemoryView<f32>,
    product_support: DeviceMemoryView<u8>,
    parameters: DeviceMemoryView<f32>,
    component_baselines: DeviceMemoryView<f32>,
}

#[cfg(feature = "semantic-policy")]
fn policy_source_views(
    originals: &[PreparedSemanticTensor],
) -> Result<PolicyProducerViews, SemanticTransitionError> {
    if originals.len() != 4 {
        return Err(publication_input_error(
            "policy requires its complete original producer roster",
        ));
    }
    let source = |index: usize| {
        originals[index]
            .source
            .clone()
            .ok_or_else(|| publication_input_error("policy producer storage is empty"))
    };
    // SAFETY: policy_producer_layouts and the original witness established
    // contiguous F32 shape, byte extents and alignment before this conversion.
    let text_logits = unsafe { source(0)?.cast::<f32>() }.ok_or_else(|| {
        publication_input_error("original text logits are not aligned F32 storage")
    })?;
    // SAFETY: the same typed original admission holds for model parameters.
    let parameters = unsafe { source(2)?.cast::<f32>() }.ok_or_else(|| {
        publication_input_error("original policy parameters are not aligned F32 storage")
    })?;
    // SAFETY: the fourth admitted producer is the exact rank-two FP32
    // component-baseline row issued by the original model invocation.
    let component_baselines = unsafe { source(3)?.cast::<f32>() }.ok_or_else(|| {
        publication_input_error("original component baselines are not aligned F32 storage")
    })?;
    Ok(PolicyProducerViews {
        text_logits,
        product_support: source(1)?,
        parameters,
        component_baselines,
    })
}

#[cfg(feature = "semantic-policy")]
fn enqueue_policy_snapshots(
    domain: &ResidentExecutionDomain,
    poisoned: &mut bool,
    buffers: &PolicyBuffers,
    support: &TrackedCudaSlice<u8>,
    original: &PolicyProducerViews,
) -> Result<(), SemanticTransitionError> {
    if original.text_logits.len() != buffers.text_logits.len()
        || original.product_support.len() != support.len()
        || original.parameters.len() != buffers.parameters.len()
        || original.component_baselines.len() != buffers.component_baselines.len()
    {
        return Err(publication_input_error(
            "original policy producers differ from their fixed native capacities",
        ));
    }
    let mut recorder = domain.new_strict_recorder();
    recorder.read(&original.text_logits);
    recorder.read(&original.product_support);
    recorder.read(&original.parameters);
    recorder.read(&original.component_baselines);
    recorder.write(&buffers.text_logits);
    recorder.write(support);
    recorder.write(&buffers.parameters);
    recorder.write(&buffers.component_baselines);
    enqueue_recorded(domain, poisoned, recorder, |enqueue| {
        for (destination, source, bytes) in [
            (
                buffers.text_logits.device_ptr_value(),
                *original.text_logits.device_ptr(),
                original.text_logits.len() * 4,
            ),
            (
                support.device_ptr_value(),
                *original.product_support.device_ptr(),
                original.product_support.len(),
            ),
            (
                buffers.parameters.device_ptr_value(),
                *original.parameters.device_ptr(),
                original.parameters.len() * 4,
            ),
            (
                buffers.component_baselines.device_ptr_value(),
                *original.component_baselines.device_ptr(),
                original.component_baselines.len() * 4,
            ),
        ] {
            // SAFETY: fixed typed capacities bound each original source and
            // distinct retained destination; this performs only device copies.
            unsafe {
                sys::cuMemcpyDtoDAsync_v2(destination, source, bytes, enqueue.stream().cu_stream())
            }
            .result()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        }
        Ok::<(), XlogError>(())
    })
}

#[cfg(feature = "semantic-policy")]
struct PolicyReplacement {
    support: TrackedCudaSlice<u8>,
    receipts: TrackedCudaSlice<SemanticTransitionReceipt>,
}

/// Original numerical inputs and receipts survive later publication and rebinding.
/// Backward consumes this exact invocation once, independently of the current bank.
#[cfg(feature = "semantic-policy")]
struct PolicyTape {
    invocation: SemanticRngBinding,
    refusal: Option<SemanticTransitionRefusal>,
    binding: SemanticCatalogueBinding,
    policy: PolicyStorage,
    support: TrackedCudaSlice<u8>,
    receipts: TrackedCudaSlice<SemanticTransitionReceipt>,
    // Immutable cold catalogue allocations remain shared by their original
    // invocation tapes; no later binding overwrites their bytes.
    components: DeviceMemoryView<SemanticComponent>,
    codebooks: DeviceMemoryView<u64>,
}

#[cfg(feature = "semantic-policy")]
struct PolicyVjpRecording {
    descriptor: Descriptor,
    recorder: LaunchRecorder,
}

impl crate::cuda_compat::KernelParamStorage for Descriptor {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (self as *const Self).cast_mut().cast()
    }
}
impl crate::cuda_compat::IntoKernelParamStorage for Descriptor {
    type Storage = Self;
    fn into_kernel_param_storage(self) -> Self {
        self
    }
}

macro_rules! content_kernel_parameter {
    ($ty:ty) => {
        impl crate::cuda_compat::KernelParamStorage for $ty {
            fn as_kernel_param(&self) -> *mut std::ffi::c_void {
                (self as *const Self).cast_mut().cast()
            }
        }
        impl crate::cuda_compat::IntoKernelParamStorage for $ty {
            type Storage = Self;
            fn into_kernel_param_storage(self) -> Self {
                self
            }
        }
    };
}
content_kernel_parameter!(PublicationRange);
content_kernel_parameter!(SemanticTensorLayout);
content_kernel_parameter!(ContinuationInputs);

const _: () = assert!(size_of::<SemanticTransitionReceipt>() == 264);
const _: () = assert!(size_of::<SemanticTaskFacts>() == 64);
const _: () = assert!(size_of::<DeviceTaskEvaluation>() == 3256);
const _: () = assert!(size_of::<DeviceState>() == 6952);
const _: () = assert!(size_of::<PolicyField>() == 24);
const _: () = assert!(size_of::<PolicyDescriptor>() == 480);
const _: () = assert!(size_of::<PolicyBackward>() == 64);
const _: () = assert!(size_of::<Descriptor>() == 736);
const OBSERVATION_BYTES: usize =
    size_of::<DeviceState>() + COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>();

#[cfg(test)]
mod task_state_contract {
    use super::*;

    #[test]
    fn text_null_requires_the_complete_singleton_receipt() {
        let mut receipt = SemanticTransitionReceipt {
            kind: COMPONENT_KIND_TEXT as u32,
            choice: TEXT_NULL as u32,
            legal_count: 1,
            active_count: 1,
            p: [1, 0, 0],
            q: [1, 0, 0],
            factor_denominator: [1, 0, 0],
            cdf_end: 1 << 63,
            mass: 1 << 63,
            ..SemanticTransitionReceipt::default()
        };
        assert!(canonical_text_null(&receipt));
        let exact = receipt;
        receipt.choice = 0;
        assert!(!canonical_text_null(&receipt));
        receipt = exact;
        receipt.p[1] = 1;
        assert!(!canonical_text_null(&receipt));
        receipt = exact;
        receipt.cdf_start = 1;
        assert!(!canonical_text_null(&receipt));
        receipt = exact;
        receipt.active_count = 2;
        assert!(!canonical_text_null(&receipt));
        receipt = exact;
        receipt.mass -= 1;
        assert!(!canonical_text_null(&receipt));
    }

    #[test]
    fn task_state_bank_includes_actual_query_receipts() {
        assert_eq!(size_of::<DeviceState>(), 6952);
        assert_eq!(size_of::<Descriptor>(), 736);
    }

    #[test]
    fn ordinary_refusal_requires_original_coordinates_and_completed_cleanup() {
        let rng = SemanticRngBinding {
            model_generation: 3,
            family_id: 4,
            stream_serial: 5,
            proposal: 6,
        };
        let binding = SemanticActionCatalogue::current().binding();
        let original = DeviceState {
            model_generation: rng.model_generation,
            family_id: u32::from(rng.family_id),
            stream_serial: rng.stream_serial,
            proposal: 6,
            next_proposal: 6,
            catalogue_generation: binding.generation,
            catalogue_digest: binding.digest,
            binding_digest: binding.digest,
            status: 5,
            ..DeviceState::default()
        };
        assert_eq!(
            refused_terminal(&original, rng, binding, binding.digest).unwrap(),
            SemanticTransitionRefusal::NonFinitePolicyInput { completed_draws: 0 }
        );
        for mutation in 0..8 {
            let mut state = original;
            match mutation {
                0 => state.model_generation += 1,
                1 => state.family_id += 1,
                2 => state.stream_serial += 1,
                3 => state.next_proposal += 1,
                4 => state.blocks = COMPONENT_COUNT as u64,
                5 => state.importance_weight = 1.0,
                6 => state.cleanup_receipts[0][0] = 1,
                _ => state.retired_roots_mask = 4,
            }
            assert!(refused_terminal(&state, rng, binding, binding.digest).is_err());
        }
        for status in [0, 1, 3, 4, 6, 7, 8, 9, 10] {
            let mut state = original;
            state.status = status;
            assert!(refused_terminal(&state, rng, binding, binding.digest).is_err());
        }
        let mut state = original;
        state.status = 2;
        state.blocks = 50;
        assert_eq!(
            refused_terminal(&state, rng, binding, binding.digest).unwrap(),
            SemanticTransitionRefusal::InvalidFinalSupport {
                completed_draws: 50
            }
        );
        state.status = 11;
        assert!(refused_terminal(&state, rng, binding, binding.digest).is_err());
        state.execution_work.overflow = 1;
        assert_eq!(
            refused_terminal(&state, rng, binding, binding.digest).unwrap(),
            SemanticTransitionRefusal::WorkCounterOverflow {
                completed_draws: 50
            }
        );
        state.next_proposal += 1;
        assert!(refused_terminal(&state, rng, binding, binding.digest).is_err());
    }

    #[test]
    fn refused_seal_cleanup_distinguishes_aliases_from_retired_private_roots() {
        let mut state = DeviceState::default();
        assert!(refused_root_cleanup(&state, 17, 2, 3).is_ok());
        state.semantic_receipts[3][33] = 1;
        state.semantic_receipts[3][39] = 17;
        state.semantic_receipts[3][1] = 2;
        state.semantic_receipts[3][34] = 2;
        state.semantic_receipts[3][35] = 3;
        assert!(refused_root_cleanup(&state, 17, 2, 3).is_err());
        state.retired_roots_mask = 1;
        assert!(refused_root_cleanup(&state, 17, 2, 3).is_ok());
        state.cleanup_receipts[1][39] = 17;
        assert!(
            refused_root_cleanup(&state, 17, 2, 3).is_err(),
            "base aliases are never retired"
        );
        state.semantic_receipts[3][1] = 1;
        state.semantic_receipts[3][34] = 7;
        state.cleanup_receipts[1][3] = 7;
        state.cleanup_receipts[1][4] = 4;
        assert!(refused_root_cleanup(&state, 17, 2, 3).is_ok());
        state.cleanup_receipts[1][4] = 3;
        assert!(refused_root_cleanup(&state, 17, 2, 3).is_err());
        state.cleanup_receipts[1][4] = 4;
        state.retired_roots_mask = 3;
        assert!(
            refused_root_cleanup(&state, 17, 2, 3).is_err(),
            "no seal exists in the other lane"
        );
    }
}

/// Completed device outcome. A refusal carries no successor, selected-score
/// receipts, or derivative authority; it is not a successful observation.
#[derive(Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "published outcomes retain the original invocation owner without another allocation"
)]
pub enum SemanticTransitionOutcome {
    Published(SemanticTransitionObservation),
    Refused(SemanticTransitionRefusal),
}

impl SemanticTransitionOutcome {
    /// Inspect the original result without relinquishing its invocation owner.
    /// A refusal preserves the same typed error as consuming the outcome.
    pub fn as_published(&self) -> Result<&SemanticTransitionObservation, SemanticTransitionError> {
        match self {
            Self::Published(observation) => Ok(observation),
            Self::Refused(refusal) => Err(refusal.into_error()),
        }
    }

    pub fn into_published(self) -> Result<SemanticTransitionObservation, SemanticTransitionError> {
        match self {
            Self::Published(observation) => Ok(observation),
            Self::Refused(refusal) => Err(refusal.into_error()),
        }
    }
}

/// Ordinary device refusal after authentic parent and cleanup reconciliation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTransitionRefusal {
    InvalidFinalSupport { completed_draws: u64 },
    NonFinitePolicyInput { completed_draws: u64 },
    WorkCounterOverflow { completed_draws: u64 },
    InvalidTrainingView { status: u64 },
}

impl SemanticTransitionRefusal {
    fn into_error(self) -> SemanticTransitionError {
        match self {
            Self::InvalidFinalSupport { completed_draws } => {
                SemanticTransitionError::InvalidFinalSupport { completed_draws }
            }
            Self::NonFinitePolicyInput { completed_draws } => {
                SemanticTransitionError::NonFinitePolicyInput { completed_draws }
            }
            Self::WorkCounterOverflow { completed_draws } => {
                SemanticTransitionError::WorkCounterOverflow { completed_draws }
            }
            Self::InvalidTrainingView { status } => {
                SemanticTransitionError::InvalidTrainingView { status }
            }
        }
    }
}

fn refused_terminal(
    state: &DeviceState,
    rng: SemanticRngBinding,
    catalogue: SemanticCatalogueBinding,
    binding_digest: Identity256,
) -> Result<SemanticTransitionRefusal, SemanticTransitionError> {
    if state.model_generation != rng.model_generation
        || state.family_id != u32::from(rng.family_id)
        || state.stream_serial != rng.stream_serial
        || state.proposal != u64::from(rng.proposal)
        || state.next_proposal != state.proposal
        || state.blocks >= COMPONENT_COUNT as u64
        || state.catalogue_generation != catalogue.generation
        || state.catalogue_digest != catalogue.digest
        || state.binding_digest != binding_digest
        || state.importance_weight != 0.0
        || state.retired_roots_mask & !3 != 0
        || state.cleanup_receipts.iter().any(|receipt| receipt[0] != 0)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    match state.status {
        2 => Ok(SemanticTransitionRefusal::InvalidFinalSupport {
            completed_draws: state.blocks,
        }),
        5 => Ok(SemanticTransitionRefusal::NonFinitePolicyInput {
            completed_draws: state.blocks,
        }),
        11 if state.execution_work.overflow == 1 => {
            Ok(SemanticTransitionRefusal::WorkCounterOverflow {
                completed_draws: state.blocks,
            })
        }
        12..=14 if state.execution_work.overflow == 0 && state.blocks == 0 => {
            Ok(SemanticTransitionRefusal::InvalidTrainingView {
                status: state.status - 11,
            })
        }
        _ => Err(SemanticTransitionError::ObservationMismatch),
    }
}

fn refused_root_cleanup(
    state: &DeviceState,
    owner: u64,
    base_slot: u64,
    base_generation: u64,
) -> Result<(), SemanticTransitionError> {
    let mismatch = || SemanticTransitionError::ObservationMismatch;
    // This slot may describe either the inactive root retired by publication
    // admission or a later candidate discard. The completed kernel checks the
    // corresponding owner operation and changes status to failure on error.
    let cleanup = &state.cleanup_receipts[0];
    if cleanup.iter().any(|&word| word != 0) && (cleanup[0] != 0 || cleanup[39] != owner) {
        return Err(mismatch());
    }
    for lane in 0..2 {
        let sealed = &state.semantic_receipts[3 + lane * 3];
        let cleanup = &state.cleanup_receipts[1 + lane];
        let consumed = state.retired_roots_mask & (1 << lane) != 0;
        // Resident root-kind is 1; a cleared or failed seal owns no sealed root.
        if sealed[0] != 0 || sealed[33] != 1 {
            if consumed || cleanup.iter().any(|&word| word != 0) {
                return Err(mismatch());
            }
            continue;
        }
        if !consumed || sealed[39] != owner {
            return Err(mismatch());
        }
        match sealed[1] {
            // An unchanged seal only aliases the protected base, so there is
            // deliberately no retirement receipt for it.
            2 if sealed[34] == base_slot
                && sealed[35] == base_generation
                && cleanup.iter().all(|&word| word == 0) => {}
            1 if cleanup[0] == 0
                && cleanup[1] == 0
                && cleanup[39] == owner
                && cleanup[3] == sealed[34]
                && sealed[35].checked_add(1) == Some(cleanup[4]) => {}
            _ => return Err(mismatch()),
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
pub struct SemanticTransitionObservation {
    pub components: Vec<SemanticTransitionReceipt>,
    pub root_handle: SemanticRootHandle,
    pub root: SemanticRootSnapshot,
    pub next_proposal: u64,
    pub lanes: [SemanticTransitionLane; 2],
    /// Full device-computed FP64 importance weight in original component order.
    /// The receipts retain the exact rational factors; this rounded reduction
    /// neither replaces them nor certifies the downstream parameter gradient.
    pub importance_weight: f64,
    /// Present only for the separately cold-bound finite evaluator.
    pub task_evaluation: Option<SemanticTaskEvaluation>,
}

/// A task-local refusal is not a semantic infrastructure failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTaskRefusal {
    Scope,
    HardConstraint,
}

/// One native evaluation of the fixed base/learned roster. The selector runs
/// only on device; these are the resulting facts, not caller-supplied scores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTaskEvaluation {
    pub binding: Identity256,
    pub query_count: u64,
    pub winner: u64,
    pub return_value: i64,
    pub facts: [SemanticTaskFacts; 3],
}

/// Device terminal and both ordered edit results. A refused lane has no root;
/// successful edit receipts preceding rollback are not published graph state.
#[derive(Debug, PartialEq, Eq)]
pub struct SemanticTransitionLane {
    /// No sealed root is produced for a task-local refusal. An owner refusal
    /// remains an actual owner error, distinct from an evaluator decision.
    pub root: Option<Result<(SemanticRootHandle, SemanticRootSnapshot), SemanticHypergraphError>>,
    pub task_refusal: Option<SemanticTaskRefusal>,
    pub edits: [Result<Option<SemanticInsertOutcome>, SemanticHypergraphError>; 2],
    /// Actual owner work, retained even when the candidate is discarded.
    pub work: SemanticTransitionWork,
}

/// Work performed by the two semantic edits of one learned candidate. These are
/// device counters, not estimates from the sampled actions or final truth alone.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticTransitionWork {
    /// Calls to the modifying semantic owner, including a call that refuses.
    /// A NO_EDIT or a suppressed edit does not call that owner.
    pub edit_commands: u64,
    /// New reachable support attachments made before any candidate discard.
    /// Re-inserting the same support event does not add an attachment.
    pub added_supports: u64,
    /// Successful edits that change a truth other than NEITHER. Adding another
    /// reason for the same truth and repeating an event do not increment this.
    pub defined_truth_changes: u64,
}

impl SemanticTransitionWork {
    fn is_valid(self) -> bool {
        self.edit_commands <= 2
            && self.added_supports <= self.edit_commands
            && self.defined_truth_changes <= self.added_supports
    }
}

/// Provider counters and explicit session stream waits. Compare snapshots around
/// launch only: cold binding/capture and terminal observation are outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticTransitionHostIoStats {
    pub htod_bytes: u64,
    pub dtoh_bytes: u64,
    pub htod_calls: u64,
    pub dtoh_calls: u64,
    pub launch_metadata_bytes: u64,
    pub launch_metadata_calls: u64,
    pub metadata_dtoh_calls: u64,
    pub observation_bytes: u64,
    pub observation_calls: u64,
    pub session_stream_waits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticTransitionError {
    CatalogueMismatch,
    InvalidInput {
        detail: String,
    },
    InvalidFinalSupport {
        completed_draws: u64,
    },
    NonFinitePolicyInput {
        completed_draws: u64,
    },
    WorkCounterOverflow {
        completed_draws: u64,
    },
    InvalidPolicyGradient {
        status: u64,
    },
    InvalidTrainingView {
        status: u64,
    },
    InvalidTaskBank,
    InvalidTaskBase,
    TaskOwnerFailure,
    PublicationRefused {
        status: u64,
    },
    OverlappingInputs,
    OverlappingLaunch,
    UnconsumedPolicyTape,
    UnpublishedTask,
    NotBound,
    NotCaptured,
    NoPendingLaunch,
    GenerationExhausted,
    ObservationMismatch,
    Poisoned,
    Semantic(SemanticHypergraphError),
    Runtime {
        operation: &'static str,
        detail: String,
    },
}

impl fmt::Display for SemanticTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SemanticTransitionError {}

fn runtime_error(operation: &'static str, error: impl fmt::Display) -> SemanticTransitionError {
    SemanticTransitionError::Runtime {
        operation,
        detail: error.to_string(),
    }
}

/// Pinned terminal bank, allocated before capture, never used for hot-loop metadata.
struct PinnedObservation {
    buffer: crate::device::PinnedHostBuffer,
}

impl PinnedObservation {
    fn new(stream: Arc<CudaStream>) -> Result<Self, SemanticTransitionError> {
        Ok(Self {
            buffer: crate::device::PinnedHostBuffer::new(&stream, OBSERVATION_BYTES)
                .map_err(|error| runtime_error("pinned observation allocation", error))?,
        })
    }

    fn enqueue_copy(&mut self, state: u64, receipts: u64, stream: &CudaStream) -> XlogResult<()> {
        // SAFETY: the recorder admits both fixed-size device ranges. The shared
        // pinned owner retains this whole batch, including a partially failed copy.
        unsafe {
            self.buffer.enqueue(stream, |ptr| {
                for (offset, source, len) in [
                    (0, state, size_of::<DeviceState>()),
                    (
                        size_of::<DeviceState>(),
                        receipts,
                        OBSERVATION_BYTES - size_of::<DeviceState>(),
                    ),
                ] {
                    // SAFETY: fixed-size ranges belong to this pinned bank and live device
                    // owners. Session waits before reading or freeing either allocation.
                    let code = {
                        sys::cuMemcpyDtoHAsync_v2(
                            ptr.add(offset).cast(),
                            source,
                            len,
                            stream.cu_stream(),
                        )
                    };
                    code.result()?;
                }
                Ok(())
            })
        }
        .map_err(|error| XlogError::Kernel(format!("semantic final observation copy: {error}")))
    }

    /// Caller must have completed the stream wait after both copies.
    unsafe fn read(
        &mut self,
    ) -> Result<(DeviceState, Vec<SemanticTransitionReceipt>), SemanticTransitionError> {
        // SAFETY: both fixed-size copies initialized the complete byte bank.
        let bytes = unsafe { self.buffer.read_vec::<u8>(OBSERVATION_BYTES) }
            .map_err(|error| runtime_error("pinned observation read", error))?;
        // SAFETY: caller establishes completion; allocations and integral ABI sizes
        // are checked, and unaligned reads do not require additional host alignment.
        let state = unsafe { bytes.as_ptr().cast::<DeviceState>().read_unaligned() };
        let components = (0..COMPONENT_COUNT)
            .map(|i| unsafe {
                bytes
                    .as_ptr()
                    .add(size_of::<DeviceState>() + i * size_of::<SemanticTransitionReceipt>())
                    .cast::<SemanticTransitionReceipt>()
                    .read_unaligned()
            })
            .collect();
        Ok((state, components))
    }
}

/// Single owner of one serial transition stream, graph, scratch, and receipt bank.
/// Retains the provider independently of the caller's handle.
pub struct SemanticTransitionSession {
    // Drop waits before destroying the exec, then retires its device storage.
    captured: Option<CapturedCudaGraph>,
    domain: ResidentExecutionDomain,
    stream: Arc<CudaStream>,
    graph: std::mem::ManuallyDrop<SemanticHypergraph>,
    root: SemanticRootHandle,
    base_snapshot: SemanticRootSnapshot,
    codebooks: ActionCodebooks,
    device_codebooks: TrackedCudaSlice<u64>,
    device_components: TrackedCudaSlice<SemanticComponent>,
    // Private observer/checker results never enter the proposal codebooks.
    task: Option<(TaskEvaluationBinding, TrackedCudaSlice<u64>)>,
    training_views: Option<Arc<SemanticTrainingViewArena>>,
    training_origins: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    task_epoch: u64,
    // Terminal observation does not publish a world. Keep the chosen roots and
    // raw banks immutable until the joint publisher consumes this pending tuple.
    task_observed: bool,
    publication: Option<Arc<PublicationStorage>>,
    publication_uploads: Vec<(usize, PublicationPayload)>,
    publication_issuer: Arc<()>,
    readers: BTreeMap<u64, PublishedReader>,
    steps: BTreeMap<u64, StepContentStorage>,
    prepared_segment: Option<PreparedSegmentState>,
    prepared_resources: Vec<Arc<dyn Send + Sync>>,
    next_reader: u64,
    continuation_base: Option<u64>,
    text_binding: Option<Arc<TextBindingStorage>>,
    admitted_transition: Option<(u64, u64, SemanticTransitionKind)>,
    release_events: Vec<cudarc::driver::CudaEvent>,
    logits: TrackedCudaSlice<f32>,
    support: TrackedCudaSlice<u8>,
    scratch: TrackedCudaSlice<u64>,
    receipts: TrackedCudaSlice<SemanticTransitionReceipt>,
    state: TrackedCudaSlice<DeviceState>,
    pinned: PinnedObservation,
    execute: CudaFunction,
    #[cfg(feature = "semantic-policy")]
    policy: Option<PolicyStorage>,
    #[cfg(feature = "semantic-policy")]
    policy_tapes: Vec<PolicyTape>,
    rng: Option<SemanticRngBinding>,
    next_proposal: u64,
    pending: bool,
    poisoned: bool,
    stream_waits: u64,
    // All captured work and storage retire before their driver/allocation owner.
    provider: Arc<CudaKernelProvider>,
}

/// Results retain each actual device invocation. An early admission refusal
/// owns no numerical invocation, selected receipts, or derivative authority.
/// Intermediate root records are historical receipts; a later step may retire
/// their slots. Acquire the canonical final reader for the current publication.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "completed outcomes retain their original invocation and step owners"
)]
pub enum SemanticPreparedStepOutcome {
    Skipped {
        step: SemanticPreparedStep,
        status: u64,
    },
    Completed {
        step: SemanticPreparedStep,
        invocation: SemanticRngBinding,
        transition: SemanticTransitionKind,
        outcome: SemanticTransitionOutcome,
    },
}

const _: () = assert!(size_of::<PreparedStepResult>() == size_of::<PublicationHeader>() + 32);

fn prepared_step_allocation_bytes(
    input_bytes: usize,
    feedback_bytes: usize,
    policy_bytes: usize,
) -> Result<u64, SemanticTransitionError> {
    [
        size_of::<PublicationLease>(),
        size_of::<PreparedStepResult>(),
        input_bytes,
        feedback_bytes,
        policy_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |sum, bytes| {
        sum.checked_add(bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })
    .and_then(|bytes| {
        u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)
    })
}

fn prepared_segment_allocation_bytes(
    step_bytes: u64,
    transition_bound: usize,
    training_origin_bytes: usize,
) -> Result<u64, SemanticTransitionError> {
    if transition_bound == 0 {
        return Err(publication_input_error(
            "a prepared segment requires a positive transition bound",
        ));
    }
    u64::try_from(transition_bound)
        .ok()
        .and_then(|bound| step_bytes.checked_mul(bound))
        .and_then(|bytes| {
            u64::try_from(training_origin_bytes)
                .ok()
                .and_then(|origin_bytes| bytes.checked_add(origin_bytes))
        })
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

#[cfg(feature = "semantic-policy")]
fn policy_buffer_bytes(layout: &SemanticPolicyLayout) -> Result<usize, SemanticTransitionError> {
    [
        layout.parameter_cells,
        layout.score_cells(),
        layout.recurrent_cells(),
        32 * TEXT_CARDINALITY,
    ]
    .into_iter()
    .try_fold(0usize, |sum, cells| {
        sum.checked_add(cells)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })?
    // Each primal bank has a dedicated adjoint bank. Only the forward hidden
    // vector is shared scratch. Baselines are immutable model outputs without
    // an adjoint bank; coefficients and status keep their own types.
    .checked_mul(2)
    .and_then(|cells| cells.checked_add(128))
    .and_then(|cells| cells.checked_add(COMPONENT_COUNT))
    .and_then(|cells| cells.checked_mul(size_of::<f32>()))
    .and_then(|bytes| bytes.checked_add(COMPONENT_COUNT * size_of::<f64>()))
    .and_then(|bytes| bytes.checked_add(size_of::<u64>()))
    .ok_or(SemanticTransitionError::GenerationExhausted)
}

fn validate_rebinding_ownership(
    prepared_segment: Option<&PreparedSegmentState>,
    unreconciled_acquisition: bool,
) -> Result<(), SemanticTransitionError> {
    if prepared_segment.is_some() {
        return Err(publication_input_error(
            "a prepared segment still owns its executable and native resources",
        ));
    }
    if unreconciled_acquisition {
        return Err(publication_input_error(
            "an unreconciled acquisition still owns its native step",
        ));
    }
    Ok(())
}

fn validate_prepared_completion(
    lease: &PublicationLease,
    parent: &PublicationHeader,
    result: &PreparedStepResult,
    instance: Identity256,
) -> Result<SemanticTransitionKind, SemanticTransitionError> {
    if lease.abi != 1
        || lease.status != 0
        || lease.active != 0
        || lease.instance != instance
        || lease.bank > 1
        || lease.bank != lease.word & 1
        || lease.epoch != lease.word >> 1
        || parent.abi != 1
        || parent.instance != instance
        || parent.publication_word != lease.word
        || parent.sealed_epoch != lease.epoch
        || result.abi != 1
        || result.header.abi != 1
        || result.header.instance != instance
        || result.header.publication_word != result.word
        || result.header.sealed_epoch != result.word >> 1
        || result.advanced > 1
        || result.advanced != u64::from(result.word != lease.word && result.refusal == 0)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    if result.word != lease.word {
        let epoch = lease
            .epoch
            .checked_add(1)
            .filter(|&epoch| epoch <= u64::MAX >> 1)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if result.word != (epoch << 1) | ((lease.word ^ 1) & 1)
            || result.header.base_word != lease.word
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
    }
    match lease.transition_kind {
        1 => Ok(SemanticTransitionKind::Proposal),
        2 => Ok(SemanticTransitionKind::Recompute),
        3 => Ok(SemanticTransitionKind::Drain),
        4 => Ok(SemanticTransitionKind::Update),
        _ => Err(SemanticTransitionError::ObservationMismatch),
    }
}

#[cfg(feature = "semantic-policy")]
impl SemanticTransitionSession {
    /// Submit this Session's complete graph once with the freshly validated
    /// canonical authority snapshot. The metadata upload precedes graph launch.
    pub fn launch_prepared_segment(
        &mut self,
        authority_decisions: Vec<u8>,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotCaptured)?;
        if !build.finished
            || build.submitted
            || self.captured.is_none()
            || authority_decisions.is_empty()
        {
            return Err(publication_input_error(
                "prepared submission requires one unused graph and fresh canonical authority bytes",
            ));
        }
        for token in &build.tokens {
            let prepared = self
                .steps
                .get(token)
                .and_then(|step| step.prepared.as_ref())
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let continuation = prepared
                .continuation
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            if prepared.transition_recorded != PREPARED_TRANSITION_BANKS
                || continuation.inputs.authority_bytes != authority_decisions.len() as u64
            {
                return Err(publication_input_error(
                    "fresh authority snapshot differs from the captured continuation byte extent",
                ));
            }
        }
        let authority = storage
            .continuation_templates
            .iter()
            .find(|range| range.role == 39 && range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let slot = usize::try_from(authority.storage_slot)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let destination = storage
            .allocations
            .get(slot)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if authority.offset_bytes != 0 || authority_decisions.len() > destination.len() {
            return Err(publication_input_error(
                "fresh authority snapshot exceeds its original fixed allocation",
            ));
        }
        let recorder = self.prepared_segment_recorder()?;
        self.prepared_segment
            .as_mut()
            .expect("checked segment")
            .submit()?;
        self.publication_uploads = vec![(slot, PublicationPayload::Metadata(authority_decisions))];
        let result = (|| {
            let PublicationPayload::Metadata(bytes) = &self.publication_uploads[0].1 else {
                unreachable!("owned authority metadata")
            };
            let mut destination = storage.allocations[slot].view().slice(..bytes.len());
            self.provider
                .htod_launch_metadata_sync_copy_into(bytes, &mut destination)
                .map_err(|error| runtime_error("prepared fresh authority upload", error))?;
            let graph = self
                .captured
                .as_ref()
                .expect("checked original captured graph");
            enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
                graph.launch_in(enqueue)
            })
        })();
        if result.is_err() {
            self.poisoned = true;
            return result;
        }
        self.pending = true;
        Ok(())
    }

    /// True only after the actual graph stream completed successfully. This
    /// remains available when a later observation or integrity check fails.
    pub fn prepared_segment_completion_known(&self) -> bool {
        self.prepared_segment
            .as_ref()
            .is_some_and(|build| build.completed)
    }

    pub fn complete_prepared_segment(
        &mut self,
    ) -> Result<Vec<SemanticPreparedStepOutcome>, SemanticTransitionError> {
        if self.is_poisoned() {
            return Err(SemanticTransitionError::Poisoned);
        }
        if !self.pending
            || !self
                .prepared_segment
                .as_ref()
                .is_some_and(|build| build.submitted && !build.completed)
        {
            return Err(SemanticTransitionError::NoPendingLaunch);
        }
        let result = self.complete_prepared_segment_inner();
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.pending = false;
            self.publication_uploads.clear();
            self.continuation_base = None;
            self.admitted_transition = None;
            self.text_binding = None;
        }
        result
    }

    fn complete_prepared_segment_inner(
        &mut self,
    ) -> Result<Vec<SemanticPreparedStepOutcome>, SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("prepared completion stream admission", error))?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "prepared graph terminal wait",
            CudaStream::synchronize,
        )?;
        self.prepared_segment
            .as_mut()
            .expect("submitted segment")
            .completed = true;
        let handles = self
            .prepared_segment
            .as_ref()
            .expect("submitted segment")
            .handles()?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let mut outcomes = Vec::new();
        outcomes
            .try_reserve_exact(handles.len())
            .map_err(|error| runtime_error("prepared outcome reservation", error))?;
        let mut previous = None;
        for step in handles {
            let prepared = self.steps[&step.token]
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let reader = prepared.reader.view();
            let result_view = prepared.result.view();
            let lease = self.publication_read(reader)?[0];
            let result = self.publication_read(result_view)?[0];
            if result.abi == 0 {
                if lease.active != 0
                    || lease.abi != 0
                    || lease.transition_kind != 0
                    || !matches!(lease.status, 1..=5 | 7)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                outcomes.push(SemanticPreparedStepOutcome::Skipped {
                    step,
                    status: lease.status,
                });
                continue;
            }
            let owner = &self.steps[&step.token];
            let prepared = owner.prepared.as_ref().expect("original prepared owner");
            let inputs = Arc::clone(
                owner
                    .inputs
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
            );
            let state_view = prepared.state.view();
            let receipts = prepared
                .receipts
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .view();
            let parent = self.publication_read(inputs.header.view())?[0];
            if previous.is_some_and(|header| header != parent) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let transition =
                validate_prepared_completion(&lease, &parent, &result, storage.instance)?;
            let state = self.publication_read(state_view)?[0];
            if transition != SemanticTransitionKind::Drain {
                let work = self.steps[&step.token]
                    .prepared
                    .as_ref()
                    .expect("original prepared owner")
                    .model_work
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if !state
                    .execution_work
                    .validate_model(&work.recording, state.status == 11)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
            let components = if transition == SemanticTransitionKind::Proposal {
                self.publication_read(receipts)?
            } else {
                Vec::new()
            };
            let invocation = parent.rng_binding()?;
            let outcome = self.reconcile_prepared_transition(
                &parent, &result, state, components, invocation, transition,
            )?;
            previous = Some(result.header);
            #[cfg(feature = "semantic-policy")]
            if transition == SemanticTransitionKind::Proposal {
                self.take_prepared_policy_tape(
                    &step,
                    invocation,
                    match &outcome {
                        SemanticTransitionOutcome::Refused(refusal) => Some(*refusal),
                        _ => None,
                    },
                )?;
            }
            self.steps
                .get_mut(&step.token)
                .expect("retained original step")
                .prepared
                .as_mut()
                .expect("retained original prepared owner")
                .observed = true;
            outcomes.push(SemanticPreparedStepOutcome::Completed {
                step,
                invocation,
                transition,
                outcome,
            });
        }
        let control = self.publication_read(storage.control.view())?[0];
        if control.abi != 1
            || control.instance != storage.instance
            || control.reader_gate != 0
            || previous.is_some_and(|header| control.word != header.publication_word)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(outcomes)
    }

    fn reconcile_prepared_transition(
        &mut self,
        parent: &PublicationHeader,
        result: &PreparedStepResult,
        state: DeviceState,
        components: Vec<SemanticTransitionReceipt>,
        invocation: SemanticRngBinding,
        transition: SemanticTransitionKind,
    ) -> Result<SemanticTransitionOutcome, SemanticTransitionError> {
        if result.refusal != 0 || state.status == 10 {
            return Err(SemanticTransitionError::PublicationRefused {
                status: result.refusal,
            });
        }
        if matches!(state.status, 2 | 5 | 11 | 12..=14) {
            let refusal = refused_terminal(
                &state,
                invocation,
                SemanticActionCatalogue::current().binding(),
                self.binding().digest,
            )?;
            if result.word != parent.publication_word || result.header != *parent {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let (root, snapshot) = self.observe_publication_root(&state, parent, 0)?;
            refused_root_cleanup(
                &state,
                self.graph.transition_arena()[1],
                u64::from(root.slot()),
                root.generation(),
            )?;
            self.root = root;
            self.base_snapshot = snapshot;
            self.next_proposal = parent.proposal;
            return Ok(SemanticTransitionOutcome::Refused(refusal));
        }
        match state.status {
            7 => return Err(SemanticTransitionError::InvalidTaskBank),
            8 => return Err(SemanticTransitionError::InvalidTaskBase),
            9 => return Err(SemanticTransitionError::TaskOwnerFailure),
            _ => {}
        }
        let next = parent
            .proposal
            .checked_add(u64::from(transition == SemanticTransitionKind::Proposal))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if state.status != 0
            || result.word == parent.publication_word
            || result.header.proposal != next
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let winner = if transition == SemanticTransitionKind::Proposal {
            state.task_evaluation.winner
        } else {
            0
        };
        let index = match winner {
            0 => 0,
            1 => 3,
            2 => 6,
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        if winner > 0 && state.retired_roots_mask & (1 << (winner - 1)) != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let root = self.observe_publication_root(&state, &result.header, index)?;
        self.finish_observed_transition(
            state,
            components,
            invocation,
            Some((result.header, root, transition)),
            true,
        )
    }
}

#[cfg(test)]
mod prepared_completion_tests {
    use super::*;

    #[test]
    fn prepared_allocation_plan_counts_original_banks_before_expanding_bound() {
        let schema = SemanticFeedbackSchema::new([&[1u8; 64], &[2u8; 64], &[3u8; 64]]).unwrap();
        let feedback = FeedbackAllocationPlan::new(&schema, 2, 3).unwrap();
        let expected_feedback = 2 * schema.feature_width * size_of::<f32>()
            + 2 * 9
            + (2 * 42 + 2 + 3 + 3 * SEMANTIC_FEEDBACK_SUPPORT_FIELDS.len()) * size_of::<u64>()
            + 6 * 4 * size_of::<u64>();
        assert_eq!(feedback.bytes, expected_feedback);
        let input_bytes = PreparedStepInputs::allocation_bytes(&[]).unwrap();
        assert_eq!(
            input_bytes,
            size_of::<PublicationHeader>() + 32 * size_of::<SourceSlot>() + u64::BITS as usize
        );
        let plans = [
            StepInputPlan {
                role: 1,
                index: 0,
                layout: SemanticTensorLayout::default(),
                banks: [
                    StepInputBankPlan {
                        storage_slot: 0,
                        span: 0..1024,
                    },
                    StepInputBankPlan {
                        storage_slot: 0,
                        span: 0..1024,
                    },
                ],
            },
            StepInputPlan {
                role: 2,
                index: 0,
                layout: SemanticTensorLayout::default(),
                banks: [
                    StepInputBankPlan {
                        storage_slot: 1,
                        span: 0..32,
                    },
                    StepInputBankPlan {
                        storage_slot: 2,
                        span: 0..32,
                    },
                ],
            },
        ];
        assert_eq!(
            PreparedStepInputs::allocation_bytes(&plans).unwrap(),
            input_bytes
                + 2 * (size_of::<PublicationRange>() + 2 * size_of::<PublicationStepInput>())
                + 32
        );
        assert!(PreparedStepInputs::allocation_bytes(&[StepInputPlan {
            banks: [
                StepInputBankPlan {
                    storage_slot: 1,
                    span: 0..usize::MAX,
                },
                StepInputBankPlan {
                    storage_slot: 2,
                    span: 0..usize::MAX,
                },
            ],
            ..plans[1].clone()
        }])
        .is_err());
        let step_bytes = prepared_step_allocation_bytes(input_bytes, feedback.bytes, 0).unwrap();
        assert_eq!(
            step_bytes,
            (input_bytes
                + feedback.bytes
                + size_of::<PublicationLease>()
                + size_of::<PreparedStepResult>()) as u64
        );
        assert_eq!(
            prepared_segment_allocation_bytes(step_bytes, 3, 0).unwrap(),
            3 * step_bytes
        );
        assert!(prepared_segment_allocation_bytes(step_bytes, 0, 0).is_err());
        assert!(prepared_segment_allocation_bytes(step_bytes, usize::MAX, 0).is_err());
    }

    #[test]
    fn prepared_scope_blocks_ordinary_rebinding_until_retirement() {
        let issuer = Arc::new(());
        let mut build = PreparedSegmentState::new(
            Arc::clone(&issuer),
            vec![1],
            std::iter::once(SemanticTransitionKind::Proposal),
        )
        .unwrap();
        assert!(validate_rebinding_ownership(None, false).is_ok());
        assert!(validate_rebinding_ownership(None, true).is_err());
        assert!(validate_rebinding_ownership(Some(&build), false).is_err());
        let step = build.handles().unwrap().remove(0);
        build.enter(&step, &issuer).unwrap();
        build.leave(&step, &issuer).unwrap();
        build.finish().unwrap();
        build.submit().unwrap();
        build.completed = true;
        assert!(validate_rebinding_ownership(Some(&build), false).is_err());
    }

    #[test]
    fn prepared_submission_is_once_only_after_complete_construction() {
        let issuer = Arc::new(());
        let mut build = PreparedSegmentState::new(
            Arc::clone(&issuer),
            vec![1],
            std::iter::once(SemanticTransitionKind::Proposal),
        )
        .unwrap();
        assert!(build.submit().is_err());
        let step = build.handles().unwrap().remove(0);
        build.enter(&step, &issuer).unwrap();
        build.leave(&step, &issuer).unwrap();
        build.finish().unwrap();
        build.submit().unwrap();
        assert!(build.submit().is_err());
        build.completed = true;
        assert!(build.submit().is_err());
    }

    #[test]
    fn completion_uses_original_parent_and_requires_device_release() {
        let parent = PublicationHeader {
            abi: 1,
            publication_word: 2,
            sealed_epoch: 1,
            proposal: 37,
            model_generation: 9,
            family_id: 3,
            stream_serial: 91,
            ..PublicationHeader::default()
        };
        let mut lease = PublicationLease {
            abi: 1,
            word: 2,
            epoch: 1,
            transition_kind: 1,
            ..PublicationLease::default()
        };
        let mut result = PreparedStepResult {
            abi: 1,
            word: 5,
            advanced: 1,
            header: PublicationHeader {
                abi: 1,
                base_word: 2,
                publication_word: 5,
                sealed_epoch: 2,
                proposal: 38,
                ..parent
            },
            ..PreparedStepResult::default()
        };
        assert_eq!(
            validate_prepared_completion(&lease, &parent, &result, parent.instance).unwrap(),
            SemanticTransitionKind::Proposal
        );
        assert_eq!(parent.rng_binding().unwrap().proposal, 37);
        lease.active = 1;
        assert!(validate_prepared_completion(&lease, &parent, &result, parent.instance).is_err());
        lease.active = 0;
        result.word = 8;
        assert!(validate_prepared_completion(&lease, &parent, &result, parent.instance).is_err());
        result.word = 5;
        lease.transition_kind = 0;
        assert!(validate_prepared_completion(&lease, &parent, &result, parent.instance).is_err());
    }
}

struct UnreleasedPublicationOwners {
    _graph: SemanticHypergraph,
    _publication: Arc<PublicationStorage>,
    _training_views: Option<Arc<SemanticTrainingViewArena>>,
    _training_origins: Option<Arc<TrackedCudaSlice<SemanticTrainingViewOriginRecord>>>,
    _readers: BTreeMap<u64, PublishedReader>,
    _steps: BTreeMap<u64, StepContentStorage>,
    _prepared_resources: Vec<Arc<dyn Send + Sync>>,
    _text_binding: Option<Arc<TextBindingStorage>>,
    #[cfg(feature = "semantic-policy")]
    _policy: Option<PolicyStorage>,
    #[cfg(feature = "semantic-policy")]
    _policy_tapes: Vec<PolicyTape>,
    _events: Vec<cudarc::driver::CudaEvent>,
    _uploads: Vec<(usize, PublicationPayload)>,
    _stream: Arc<CudaStream>,
}

thread_local! {
    // Driver/function/semantic owners are constructing-thread confined. A
    // violated explicit-release contract must retain these actual owners even
    // when that thread exits; no unsafe Send or foreign-thread destructor runs.
    static UNRELEASED_PUBLICATIONS: std::cell::RefCell<Vec<std::mem::ManuallyDrop<UnreleasedPublicationOwners>>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for SemanticTransitionSession {
    fn drop(&mut self) {
        // Captured work retires before either the semantic owner or its banks.
        drop(self.captured.take());
        if !self.readers.is_empty() || !self.steps.is_empty() {
            if let Some(publication) = self.publication.take() {
                // SAFETY: graph is taken exactly once here and its normal
                // destructor is disabled by ManuallyDrop in Session storage.
                let graph = unsafe { std::mem::ManuallyDrop::take(&mut self.graph) };
                let owners = UnreleasedPublicationOwners {
                    _graph: graph,
                    _publication: publication,
                    _training_views: self.training_views.take(),
                    _training_origins: self.training_origins.take(),
                    _readers: std::mem::take(&mut self.readers),
                    _events: std::mem::take(&mut self.release_events),
                    _steps: std::mem::take(&mut self.steps),
                    _text_binding: self.text_binding.take(),
                    _prepared_resources: std::mem::take(&mut self.prepared_resources),
                    #[cfg(feature = "semantic-policy")]
                    _policy: self.policy.take(),
                    #[cfg(feature = "semantic-policy")]
                    _policy_tapes: std::mem::take(&mut self.policy_tapes),
                    _uploads: std::mem::take(&mut self.publication_uploads),
                    _stream: Arc::clone(&self.stream),
                };
                eprintln!("semantic publication owners retained: Session dropped with unreleased readers or retained steps");
                let owners = std::mem::ManuallyDrop::new(owners);
                let _ = UNRELEASED_PUBLICATIONS
                    .try_with(|quarantine| quarantine.borrow_mut().push(owners));
                return;
            }
        }
        // SAFETY: no active reader can reach the graph, and the captured exec
        // retired above. This is the graph's unique normal destructor call.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.graph) };
    }
}

impl SemanticTransitionSession {
    /// Allocate every native step owner before the first capture begins.
    /// Freeze requested modes and reserve the complete segment before creating
    /// per-step storage. Device admission alone selects terminal drain.
    pub fn prepare_segment_steps(
        &mut self,
        transitions: impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone,
    ) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        self.ensure_rebindable()?;
        let transitions = transitions.collect::<Vec<_>>();
        if transitions.is_empty() || transitions.contains(&SemanticTransitionKind::Drain) {
            return Err(publication_input_error(
                "a prepared segment requires frozen proposal, recompute, or update steps",
            ));
        }
        let update_count = transitions
            .iter()
            .filter(|kind| **kind == SemanticTransitionKind::Update)
            .count();
        let training_view = if update_count == 0 {
            None
        } else {
            Some(Arc::clone(self.training_views.as_ref().ok_or_else(
                || {
                    publication_input_error(
                        "prepared updates require the native training-view arena",
                    )
                },
            )?))
        };
        if self.prepared_segment.is_some()
            || !self.readers.is_empty()
            || self.captured.is_some()
            || self.admitted_transition.is_some()
            || self.text_binding.is_some()
        {
            return Err(publication_input_error("segment construction requires an idle Session with no acquired reader or prior executable"));
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let transition_bound = transitions.len();
        let bound = u64::try_from(transition_bound)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let end = self
            .next_reader
            .checked_add(bound)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let schema = SemanticFeedbackSchema::new(self.feedback_statement_bytes()?)?;
        let descriptor = self.descriptor();
        let input_plans = PreparedStepInputs::plan(&storage)?;
        let feedback_plan = FeedbackAllocationPlan::new(
            &schema,
            storage.contract_value.feedback_capacity,
            descriptor.arena[4],
        )?;
        #[cfg(not(feature = "semantic-policy"))]
        let policy_bytes = 0;
        #[cfg(feature = "semantic-policy")]
        let policy_bytes = policy_buffer_bytes(&self.policy_layout()?)?
            .checked_add(self.codebooks.input_cells)
            .and_then(|n| n.checked_add(COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>()))
            .and_then(|n| n.checked_add(size_of::<DeviceState>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let step_bytes = prepared_step_allocation_bytes(
            PreparedStepInputs::allocation_bytes(&input_plans)?,
            feedback_plan.bytes,
            policy_bytes,
        )?;
        let training_origin_bytes = if self.training_views.is_some() {
            transition_bound
                .checked_mul(size_of::<SemanticTrainingViewOriginRecord>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        } else {
            0
        };
        let training_view_bytes = training_view.as_ref().map_or(Ok(0usize), |arena| {
            arena
                .selection_bytes()?
                .checked_mul(update_count)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        let update_binding_bytes = update_count
            .checked_mul(storage.model_slots.len())
            .and_then(|count| count.checked_mul(size_of::<ModelUpdateBinding>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let update_binding_bytes = u64::try_from(update_binding_bytes)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let bytes =
            prepared_segment_allocation_bytes(step_bytes, transition_bound, training_origin_bytes)?
                .checked_add(
                    u64::try_from(training_view_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .and_then(|bytes| bytes.checked_add(update_binding_bytes))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        // Claim the actual local and runtime budgets before any T-sized host
        // collection or native allocation. Every fixed bank consumes this claim.
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes)
            .map_err(|error| runtime_error("prepared segment reservation", error))?;
        let training_origins = if training_origin_bytes == 0 {
            None
        } else {
            let origins = reservation
                .alloc::<SemanticTrainingViewOriginRecord>(transition_bound)
                .map_err(|error| runtime_error("training origin roster allocation", error))?;
            upload_publication(
                &self.provider,
                &vec![SemanticTrainingViewOriginRecord::default(); transition_bound],
                &origins,
            )?;
            Some(Arc::new(origins))
        };
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(transition_bound)
            .map_err(|error| runtime_error("prepared token reservation", error))?;
        tokens.extend((self.next_reader..end).map(|token| token + 1));
        let build = PreparedSegmentState::new(
            Arc::clone(&self.publication_issuer),
            tokens,
            transitions.iter().copied(),
        )?;
        let handles = build.handles()?;
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("segment preparation stream admission", error))?;
        let kernel = |name| {
            self.provider
                .device()
                .inner()
                .get_func("xlog_semantic_transition", name)
                .ok_or_else(|| runtime_error("kernel lookup", format!("{name} unavailable")))
        };
        let admit = kernel("semantic_publication_step_admit")?;
        let kind_gate = kernel("semantic_publication_step_kind_gate")?;
        let kind_bank_gate = kernel("semantic_publication_step_kind_bank_gate")?;
        let active_gate = kernel("semantic_publication_step_active_gate")?;
        let drain_prepare = kernel("semantic_publication_prepare_drain")?;
        let release = kernel("semantic_publication_step_release")?;
        let witness = kernel("semantic_tensor_content_witness")?;
        let content_guard = kernel("semantic_publication_content_guard")?;
        let result_kernel = kernel("semantic_publication_step_result")?;
        let model_update_admissibility_copy =
            kernel("semantic_publication_prepare_model_update_admissibility")?;
        let model_update_copy = kernel("semantic_publication_apply_model_update")?;
        self.prepared_segment = Some(build);
        self.next_reader = end;
        let result = (|| {
            for (ordinal, (handle, kind)) in handles.iter().zip(&transitions).enumerate() {
                let reader = reservation
                    .alloc(1)
                    .map_err(|error| runtime_error("prepared reader allocation", error))?;
                let result = reservation
                    .alloc(1)
                    .map_err(|error| runtime_error("prepared result allocation", error))?;
                let model_update = if *kind == SemanticTransitionKind::Update {
                    let bindings = reservation
                        .alloc::<ModelUpdateBinding>(storage.model_slots.len())
                        .map_err(|error| runtime_error("model update binding allocation", error))?;
                    upload_publication(
                        &self.provider,
                        &vec![ModelUpdateBinding::default(); storage.model_slots.len()],
                        &bindings,
                    )?;
                    let admissibility = reservation.alloc::<u8>(1).map_err(|error| {
                        runtime_error("model update admissibility allocation", error)
                    })?;
                    upload_publication(&self.provider, &[0u8], &admissibility)?;
                    Some(PreparedModelUpdate {
                        bindings,
                        admissibility,
                        output: None,
                        admissibility_copy: model_update_admissibility_copy.clone(),
                        copy: model_update_copy.clone(),
                    })
                } else {
                    None
                };
                let inputs = Arc::new(PreparedStepInputs::allocate_reserved(
                    &self.provider,
                    Arc::clone(&storage),
                    reader.view(),
                    input_plans.clone(),
                    &mut reservation,
                )?);
                let feedback = FeedbackBuffers::allocate_reserved(
                    &self.provider,
                    schema.clone(),
                    feedback_plan,
                    &mut reservation,
                )?;
                #[cfg(feature = "semantic-policy")]
                let (policy_buffers, support, receipts, state) =
                    self.prepare_step_policy_storage(&mut reservation)?;
                let mut step = StepContentStorage::new();
                step.inputs = Some(Arc::clone(&inputs));
                step.feedback.push(feedback);
                step.prepared = Some(PreparedStepStorage {
                    reader,
                    digests: None,
                    next_digest: 0,
                    model_work: None,
                    model_update,
                    admit: admit.clone(),
                    kind_gate: kind_gate.clone(),
                    kind_bank_gate: kind_bank_gate.clone(),
                    active_gate: active_gate.clone(),
                    drain_prepare: drain_prepare.clone(),
                    release: release.clone(),
                    witness: witness.clone(),
                    content_guard: content_guard.clone(),
                    inputs_recorded: false,
                    continuation: None,
                    transition_recorded: 0,
                    drain_recorded: false,
                    observed: false,
                    result,
                    training_origin: training_origins
                        .as_ref()
                        .and_then(|origins| origins.view().try_slice(ordinal..ordinal + 1)),
                    training_view: if *kind == SemanticTransitionKind::Update {
                        Some(
                            training_view
                                .as_ref()
                                .expect("validated update training-view owner")
                                .allocate_selection_reserved(
                                    &mut reservation,
                                    training_origins.clone(),
                                )?,
                        )
                    } else {
                        None
                    },
                    result_kernel: result_kernel.clone(),
                    #[cfg(feature = "semantic-policy")]
                    policy_buffers: Some(policy_buffers),
                    #[cfg(feature = "semantic-policy")]
                    policy: None,
                    #[cfg(feature = "semantic-policy")]
                    support: Some(support),
                    #[cfg(feature = "semantic-policy")]
                    receipts: Some(receipts),
                    #[cfg(feature = "semantic-policy")]
                    state,
                    #[cfg(feature = "semantic-policy")]
                    policy_sources: Vec::new(),
                    #[cfg(feature = "semantic-policy")]
                    policy_witness: None,
                });
                self.steps.insert(handle.token, step);
                let prepared = self.steps[&handle.token]
                    .prepared
                    .as_ref()
                    .expect("installed fixed step");
                // Only immutable catalogue identity is initialized on the host;
                // actual invocation coordinates come from device admission.
                #[cfg(feature = "semantic-policy")]
                upload_publication(
                    &self.provider,
                    &[DeviceState {
                        catalogue_generation: self.binding().generation,
                        catalogue_digest: Identity256::from_bytes(CATALOGUE_DIGEST),
                        binding_digest: self.binding().digest,
                        ..DeviceState::default()
                    }],
                    &prepared.state,
                )?;
                upload_publication(
                    &self.provider,
                    &[PublicationLease::default()],
                    &prepared.reader,
                )?;
                upload_publication(
                    &self.provider,
                    &[PreparedStepResult::default()],
                    &prepared.result,
                )?;
                inputs.initialize(&self.provider)?;
            }
            if reservation.remaining_bytes() != 0 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            self.training_origins = training_origins;
            Ok(handles)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn checked_prepared_step(
        &self,
        step: &SemanticPreparedStep,
        recording: bool,
    ) -> Result<&StepContentStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or_else(|| publication_input_error("Session has no prepared segment"))?;
        if recording {
            build.check(step, &self.publication_issuer, true)?;
        } else {
            build.check_retained(step, &self.publication_issuer)?;
        }
        self.steps
            .get(&step.token)
            .filter(|owner| owner.prepared.is_some() && owner.identity.is_none())
            .ok_or_else(|| {
                publication_input_error("prepared step has no original native storage owner")
            })
    }

    /// Original nominal schedule mode, available only before construction ends.
    /// It does not predict the effective mode selected by device admission.
    pub fn prepared_transition_kind(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<SemanticTransitionKind, SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        let build = self
            .prepared_segment
            .as_ref()
            .expect("checked prepared scope");
        build.check(step, &self.publication_issuer, false)?;
        build.requested_kind(step, &self.publication_issuer)
    }

    fn check_prepared_cold(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        if self
            .prepared_segment
            .as_ref()
            .expect("checked build")
            .capturing
        {
            return Err(publication_input_error(
                "fixed resources must be prepared before capture begins",
            ));
        }
        Ok(())
    }

    pub fn prepared_stream(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<Arc<CudaStream>, SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        Ok(Arc::clone(&self.stream))
    }

    /// Reserve private digest storage for original tensors born while recording.
    /// The producer derives this bound from its retained allocation capacity.
    pub fn reserve_tensor_content(
        &mut self,
        step: &SemanticPreparedStep,
        tensor_capacity: usize,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if tensor_capacity == 0 {
            return Err(publication_input_error(
                "content reservation requires a positive original tensor capacity",
            ));
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("content reservation stream admission", error))?;
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .digests
            .is_some()
        {
            return Err(publication_input_error(
                "original content capacity is reserved once per fixed step",
            ));
        }
        let cells = tensor_capacity
            .checked_mul(4)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let digest = Arc::new(allocate_publication(&self.provider, cells)?);
        self.steps
            .get_mut(&step.token)
            .expect("checked fixed step")
            .prepared
            .as_mut()
            .expect("checked prepared owner")
            .digests = Some(digest);
        Ok(())
    }

    /// Reserve and allocate the original producer's event roster before capture.
    /// This does not execute the model or upload any recording-time event data.
    pub fn reserve_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        event_capacity: usize,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .model_work
            .is_some()
        {
            return Err(publication_input_error(
                "original model work capacity is reserved once per step",
            ));
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("model work reservation stream admission", error))?;
        let bytes = event_capacity
            .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes =
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes)
            .map_err(|error| runtime_error("original model work reservation", error))?;
        let recording = ModelWorkRecording::new(event_capacity).map_err(publication_input_error)?;
        let reset = self
            .provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_model_work_reset")
            .ok_or_else(|| runtime_error("kernel lookup", "model work reset unavailable"))?;
        let device = reservation
            .alloc(event_capacity)
            .map_err(|error| runtime_error("original model work allocation", error))?;
        let actual = reservation
            .alloc(event_capacity * 3)
            .map_err(|error| runtime_error("original device work allocation", error))?;
        let work = PreparedModelWork {
            recording,
            device,
            actual,
            reset,
            capture_bank: None,
            replay_cursor: 0,
        };
        // Initialize the complete exported scratch allocation cold. Each real
        // producer occurrence additionally records its own reset before use.
        work.reset_slots(&self.domain, &mut self.poisoned, 0, event_capacity)?;
        self.steps
            .get_mut(&step.token)
            .expect("checked original step")
            .prepared
            .as_mut()
            .expect("prepared owner")
            .model_work = Some(work);
        Ok(())
    }

    /// Export the original work scratch as U64[capacity,3] before capture.
    /// Each row is [actual_units, sticky_overflow, producer_writes]. It is not a
    /// canonical receipt or a writable alias of the immutable work descriptors.
    pub fn model_work_buffer(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if consumer_stream != self.stream.cu_stream() as u64 {
            return Err(publication_input_error(
                "model work producers must use the original native recording stream",
            ));
        }
        let work = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .model_work
            .as_ref()
            .ok_or_else(|| publication_input_error("model work requires its cold reservation"))?;
        let capacity = i64::try_from(work.actual.len() / 3)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        // SAFETY: the entire padding-free U64 scratch bank was initialized by
        // the cold reset; the exported view retains its actual allocation owner.
        let view = unsafe { work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.export_prepared_view(
            step,
            view,
            vec![capacity, 3],
            vec![3, 1],
            (1, 64),
            consumer_stream,
        )
    }

    /// Reserve an occurrence in the current original recording and enqueue its
    /// reset immediately before the producer. A composite operator records one
    /// occurrence per actual unit category, rather than charging a dense mask.
    pub fn record_model_device_work(
        &mut self,
        step: &SemanticPreparedStep,
        kind: ModelWorkKind,
        upper_dimensions: &[u64],
    ) -> Result<usize, SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        let result = (|| {
            let work = self
                .steps
                .get_mut(&step.token)
                .expect("checked original step")
                .prepared
                .as_mut()
                .expect("prepared owner")
                .model_work
                .as_mut()
                .ok_or_else(|| {
                    publication_input_error("model work requires its cold reservation")
                })?;
            let slot = work.next_slot().map_err(publication_input_error)?;
            if slot >= work.actual.len() / 3 {
                return Err(publication_input_error(
                    "model work exceeds its cold event capacity",
                ));
            }
            let actual = work
                .actual
                .device_ptr_value()
                .checked_add((slot * 3 * size_of::<u64>()) as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let event = ModelWorkEvent::device_operation(kind, upper_dimensions, actual)
                .map_err(publication_input_error)?;
            let recorded = work.record_event(event).map_err(publication_input_error)?;
            debug_assert_eq!(recorded, slot);
            work.reset_slots(&self.domain, &mut self.poisoned, slot, 1)?;
            Ok(slot)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Append an actual operator occurrence during the original first recording.
    /// Geometry is expressed in the agreed unit (coordinates, summands or bytes),
    /// not FLOPs, elapsed time, or a caller-computed aggregate cost.
    pub fn record_model_work(
        &mut self,
        step: &SemanticPreparedStep,
        kind: ModelWorkKind,
        dimensions: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        let result = (|| {
            let event =
                ModelWorkEvent::operation(kind, dimensions).map_err(publication_input_error)?;
            let work = self
                .steps
                .get_mut(&step.token)
                .expect("checked original step")
                .prepared
                .as_mut()
                .expect("prepared owner")
                .model_work
                .as_mut()
                .ok_or_else(|| {
                    publication_input_error("model work requires its original cold reservation")
                })?;
            work.record_event(event)
                .map(|_| ())
                .map_err(publication_input_error)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Charge each original saved clone from its retained native tensor layout.
    /// A repeated storage still costs another copy when a new witness records
    /// another save occurrence. Reusing the same occurrence is rejected.
    pub fn record_saved_tensor_work(
        &mut self,
        step: &SemanticPreparedStep,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        let result = (|| {
            let owner = self
                .steps
                .get_mut(&step.token)
                .expect("checked original step");
            if !Arc::ptr_eq(&witness.issuer, &self.publication_issuer)
                || witness.reader_token != step.token
                || !Arc::ptr_eq(&witness._witness, &owner.content_witnesses)
            {
                return Err(publication_input_error(
                    "saved work requires this original step and tensor witness",
                ));
            }
            let content = owner.content.get(witness.index).ok_or_else(|| {
                publication_input_error("saved work has no retained original tensor occurrence")
            })?;
            let work = owner
                .prepared
                .as_mut()
                .expect("prepared owner")
                .model_work
                .as_mut()
                .ok_or_else(|| {
                    publication_input_error("saved work requires its original cold reservation")
                })?;
            for (index, tensor) in content.tensors.iter().enumerate() {
                let layout = &tensor.layout;
                let dimensions = layout
                    .dimensions
                    .get(..layout.rank as usize)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let event = ModelWorkEvent::saved(
                    witness.index as u64,
                    index as u64,
                    dimensions,
                    layout.element_bytes,
                )
                .map_err(publication_input_error)?;
                work.record_event(event).map_err(publication_input_error)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn export_prepared_view(
        &mut self,
        step: &SemanticPreparedStep,
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        let guard = Arc::clone(&self.steps[&step.token].aliases);
        self.steps
            .get_mut(&step.token)
            .expect("checked fixed step")
            .consumer_streams
            .insert(consumer_stream);
        self.export_owned_view(view, shape, strides, dtype, guard, consumer_stream)
    }

    pub fn prepared_source(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        let (_, shape, strides) = PublicationBankField::Source.layout();
        let view = self.steps[&step.token]
            .inputs
            .as_ref()
            .expect("fixed input owner")
            .field_view(PublicationBankField::Source)?;
        self.export_prepared_view(step, view, shape, strides, (1, 64), consumer_stream)
    }

    pub fn prepared_prefix(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.prepared_tensor(
            step,
            SemanticStateRole::PrefixSource,
            0,
            bank,
            consumer_stream,
        )
    }

    pub fn prepared_record(
        &mut self,
        step: &SemanticPreparedStep,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if is_tensor_role(role as u64) {
            return Err(publication_input_error(
                "tensor roles require their original model layout",
            ));
        }
        if role == SemanticStateRole::PrefixSource {
            return Err(publication_input_error(
                "prepared prefixes require the fixed typed capacity and original device extent",
            ));
        }
        let storage = self.publication.as_ref().expect("prepared publication");
        let key = (role as u64, index);
        let inputs = self.steps[&step.token]
            .inputs
            .as_ref()
            .expect("fixed inputs");
        let view = if let Some(view) = inputs.views[0].get(&key) {
            view.clone()
        } else {
            let original = storage.bank_templates[0]
                .iter()
                .find(|range| (range.role, range.index) == key)
                .ok_or_else(|| {
                    publication_input_error("record has no original fixed declaration")
                })?;
            let other = storage.bank_templates[1]
                .iter()
                .find(|range| (range.role, range.index) == key)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if publication_mutable_role(role as u64)
                || original.storage_slot != other.storage_slot
                || original.offset_bytes != other.offset_bytes
                || original.length_bytes != other.length_bytes
            {
                return Err(publication_input_error(
                    "bank-varying records require a private prepared input producer",
                ));
            }
            let begin = original.offset_bytes as usize;
            let end = begin
                .checked_add(original.length_bytes as usize)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            storage.allocations[original.storage_slot as usize]
                .view()
                .slice(begin..end)
        };
        let bytes =
            i64::try_from(view.len()).map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        self.export_prepared_view(step, view, vec![bytes], vec![1], (1, 8), consumer_stream)
    }

    pub fn prepared_range_keys(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<Vec<(SemanticStateRole, u64)>, SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        self.publication
            .as_ref()
            .expect("prepared publication")
            .bank_templates[0]
            .iter()
            .map(|range| {
                SemanticStateRole::from_code(range.role)
                    .map(|role| (role, range.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect()
    }

    /// Export the fixed original capacity; live extents remain device inputs.
    pub fn prepared_tensor(
        &mut self,
        step: &SemanticPreparedStep,
        role: SemanticStateRole,
        index: u64,
        bank: usize,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        let (view, layout) = self.prepared_tensor_view(step, role, index, bank)?;
        let shape = layout.dimensions[..layout.rank as usize]
            .iter()
            .map(|&value| {
                i64::try_from(value).map_err(|_| SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let strides = layout.strides_bytes[..layout.rank as usize]
            .iter()
            .map(|&value| {
                i64::try_from(value / layout.element_bytes)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = match layout.scalar_type {
            1 => (1, 8),
            2 => (1, 32),
            3 => (1, 64),
            4 => (2, 16),
            5 => (4, 16),
            6 => (2, 32),
            7 => (0, 64),
            8 => (6, 8),
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        self.export_prepared_view(step, view, shape, strides, dtype, consumer_stream)
    }

    /// Export one fixed device port produced by this Update step's native
    /// selection. Its status word remains device-resident and gates consumers.
    pub fn prepared_training_view_port(
        &mut self,
        step: &SemanticPreparedStep,
        port: SemanticTrainingViewPort,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if self.prepared_transition_kind(step)? != SemanticTransitionKind::Update {
            return Err(publication_input_error(
                "native selected training views belong only to prepared updates",
            ));
        }
        let (view, shape, strides, dtype) = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .training_view
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .port(port)?;
        self.export_prepared_view(step, view, shape, strides, dtype, consumer_stream)
    }

    /// Actual allocation origin and complete backing bytes of the fixed view
    /// exported by `prepared_tensor`. The byte export retains the original step
    /// grant and stream ordering. Live extents and version counters are separate.
    pub fn prepared_tensor_allocation(
        &mut self,
        step: &SemanticPreparedStep,
        role: SemanticStateRole,
        index: u64,
        bank: usize,
        consumer_stream: u64,
    ) -> Result<(DeviceAllocationProvenance, DlpackManagedTensor), SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let (view, _) = self.prepared_tensor_view(step, role, index, bank)?;
        let provenance = view.allocation_provenance().ok_or_else(|| {
            publication_input_error("prepared tensor has no complete native allocation owner")
        })?;
        let allocation = view
            .allocation_view()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bytes = i64::try_from(allocation.len())
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let tensor = self.export_prepared_view(
            step,
            allocation,
            vec![bytes],
            vec![1],
            (1, 8),
            consumer_stream,
        )?;
        Ok((provenance, tensor))
    }

    fn prepared_tensor_view(
        &self,
        step: &SemanticPreparedStep,
        role: SemanticStateRole,
        index: u64,
        bank: usize,
    ) -> Result<(DeviceMemoryView<u8>, SemanticTensorLayout), SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if bank > 1 {
            return Err(publication_input_error(
                "prepared tensor bank must be zero or one",
            ));
        }
        let inputs = self.steps[&step.token]
            .inputs
            .as_ref()
            .expect("fixed input owner");
        let plan = inputs
            .plans
            .iter()
            .find(|plan| (plan.role, plan.index) == (role as u64, index))
            .ok_or_else(|| {
                publication_input_error("prepared tensor has no fixed original input owner")
            })?;
        Ok((
            inputs.views[bank][&(role as u64, index)].clone(),
            plan.layout,
        ))
    }

    pub fn prepared_terminal(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.prepared_bank_field(step, PublicationBankField::Terminal, consumer_stream)
    }

    pub fn prepared_prefix_extent(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.prepared_bank_field(step, PublicationBankField::PrefixExtent, consumer_stream)
    }

    pub fn prepared_ring_head(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.prepared_bank_field(step, PublicationBankField::RingHead, consumer_stream)
    }

    fn prepared_bank_field(
        &mut self,
        step: &SemanticPreparedStep,
        field: PublicationBankField,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        let inputs = self.steps[&step.token]
            .inputs
            .as_ref()
            .expect("prepared input owner");
        let (_, shape, strides) = field.layout();
        self.export_prepared_view(
            step,
            inputs.field_view(field)?,
            shape,
            strides,
            (1, 64),
            consumer_stream,
        )
    }

    /// Immutable admitted model geometry; no future publication values occur.
    pub fn prepared_model_geometry(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<(Identity256, Identity256, u64, u64, u64, u64), SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        let contract = self
            .publication
            .as_ref()
            .expect("prepared publication")
            .contract_value;
        Ok((
            contract.topology_identity,
            contract.table_identity,
            contract.model_generation,
            contract.prefix_capacity,
            contract.feedback_capacity,
            contract.max_position,
        ))
    }

    /// Authenticate the retained model inputs of an actually completed step.
    /// This cold observation joins every original consumer before checking the
    /// private input seals and reading that step's actual acquisition/result.
    /// It never acquires a current bank or constructs a historical live reader.
    ///
    /// Returns (predecessor, successor-or-None, model generation, model geometry
    /// digest, model numerical digest). A numerical refusal has no successor;
    /// skipped steps and unknown completion have no observable input binding.
    /// Model record 44 and complete model backing allocations are the existing
    /// prepared record/tensor exports, retained until original step retirement.
    /// Recheck this binding after serializing those read-only exports. This is
    /// not complete semantic replay material or authority for another forward.
    pub fn prepared_model_binding(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_streams: &[u64],
    ) -> Result<
        (
            SemanticPublishedIdentity,
            Option<SemanticPublishedIdentity>,
            u64,
            Identity256,
            Identity256,
        ),
        SemanticTransitionError,
    > {
        let owner = self.checked_prepared_step(step, false)?;
        let prepared = owner.prepared.as_ref().expect("checked prepared owner");
        if !self
            .prepared_segment
            .as_ref()
            .expect("checked prepared scope")
            .completed
            || !prepared.observed
        {
            return Err(publication_input_error(
                "model input observation requires this step's known completed execution",
            ));
        }
        let inputs = Arc::clone(
            owner
                .inputs
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        let reader = prepared.reader.view();
        let result = prepared.result.view();
        let _ordinary =
            crate::cuda_graph::reserve_uncaptured_stream(&self.stream).map_err(|error| {
                runtime_error("completed model observation stream admission", error)
            })?;
        self.complete_step_consumers_by_token(step.token, consumer_streams)?;
        inputs.verify(&self.domain, &mut self.poisoned)?;
        let parent = self.publication_read(inputs.header.view())?[0];
        let reader = self.publication_read(reader)?[0];
        let result = self.publication_read(result)?[0];
        if validate_prepared_completion(&reader, &parent, &result, inputs.storage.instance).is_err()
            || result.refusal != 0
            || (result.advanced == 0 && result.header != parent)
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let identity = |header: PublicationHeader| SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        };
        Ok((
            identity(parent),
            (result.advanced == 1).then(|| identity(result.header)),
            parent.model_generation,
            parent.model_geometry_digest,
            parent.model_numerical_digest,
        ))
    }

    pub fn prepared_feedback(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<(SemanticFeedbackSchema, [DlpackManagedTensor; 6]), SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        let feedback = &self.steps[&step.token].feedback[0];
        let schema = feedback.schema.clone();
        let layouts = feedback
            .content
            .tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.source.as_ref().expect("feedback owner").clone(),
                    tensor.layout,
                )
            })
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(6);
        for (view, layout) in layouts {
            let shape = layout.dimensions[..layout.rank as usize]
                .iter()
                .map(|&value| value as i64)
                .collect();
            let strides = layout.strides_bytes[..layout.rank as usize]
                .iter()
                .map(|&value| (value / layout.element_bytes) as i64)
                .collect();
            let dtype = match layout.scalar_type {
                3 => (1, 64),
                6 => (2, 32),
                8 => (6, 8),
                _ => return Err(SemanticTransitionError::ObservationMismatch),
            };
            outputs.push(self.export_prepared_view(
                step,
                view,
                shape,
                strides,
                dtype,
                consumer_stream,
            )?);
        }
        Ok((
            schema,
            outputs
                .try_into()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
        ))
    }

    /// Cold readback of each original raw feedback, statement and provenance
    /// record, after known completion and before this prepared step retires.
    /// The original input guard brackets readback; neither a current-bank
    /// reader nor a fresh encoder/semantic query is constructed. These bytes
    /// are inputs to the canonical carrier, not an execution or replay grant.
    pub fn prepared_feedback_records(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_streams: &[u64],
    ) -> Result<
        (
            SemanticPublishedIdentity,
            Option<SemanticPublishedIdentity>,
            Vec<SemanticFeedbackRecordMaterial>,
        ),
        SemanticTransitionError,
    > {
        let binding = self.prepared_model_binding(step, consumer_streams)?;
        let inputs = Arc::clone(
            self.checked_prepared_step(step, false)?
                .inputs
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        let ranges = self.publication_read(inputs.ranges.view())?;
        let mut records = Vec::new();
        for range in ranges
            .into_iter()
            .filter(|range| matches!(range.role, 15..=17))
        {
            let original = inputs
                .storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let capacity = usize::try_from(range.offset_bytes)
                .ok()
                .and_then(|begin| original.len().checked_sub(begin))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let view = inputs
                .views
                .first()
                .expect("two publication input banks")
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let bytes = self.publication_read(view.clone())?;
            match (PublicationMaterialRange {
                range,
                capacity,
                bytes,
            })
            .into_feedback_record()
            {
                Ok(record) => records.push(record),
                Err(error) => {
                    self.poisoned = true;
                    return Err(error);
                }
            }
        }
        if self.prepared_model_binding(step, consumer_streams)? != binding {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok((binding.0, binding.1, records))
    }

    /// The caller keeps the builder outside its Session lock while recording
    /// original producers. Retained resources own their preallocated pools.
    pub fn begin_prepared_segment(
        &mut self,
        resources: Vec<Arc<dyn Send + Sync>>,
    ) -> Result<(SemanticPreparedSegmentCapture, Arc<CudaStream>), SemanticTransitionError> {
        self.ensure_quiescent()?;
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if build.capturing || build.finished {
            return Err(publication_input_error(
                "prepared segment construction already began",
            ));
        }
        // Keep the actual pool/model allocations beyond graph retirement for
        // original late consumers. Unknown completion quarantines this same set.
        self.prepared_resources = resources.clone();
        let builder = crate::cuda_graph::ConditionalCudaGraphSequenceBuilder::new_retaining(
            &self.stream,
            resources,
        )
        .map_err(|error| runtime_error("bounded segment capture", error))?;
        let scope = Arc::clone(&build.scope);
        self.prepared_segment
            .as_mut()
            .expect("checked build")
            .capturing = true;
        self.graph.enter_transition();
        Ok((
            SemanticPreparedSegmentCapture { builder, scope },
            Arc::clone(&self.stream),
        ))
    }

    fn check_prepared_content_stream(
        &self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if dlpack_consumer_stream(consumer_stream)? != self.stream.cu_stream() {
            return Err(publication_input_error(
                "recorded content must use its original native capture stream",
            ));
        }
        if !self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .inputs_recorded
        {
            return Err(publication_input_error(
                "original step inputs must precede recorded content",
            ));
        }
        Ok(())
    }

    /// Bind the original live model roster cold. Each execution later copies
    /// its expected seals from the actual device-acquired publication.
    pub fn bind_prepared_model_content(
        &mut self,
        step: &SemanticPreparedStep,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<SemanticTensorContentWitness, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        dlpack_consumer_stream(consumer_stream)?;
        let storage = Arc::clone(self.publication.as_ref().expect("prepared publication"));
        let original = prepare_semantic_tensors(&self.provider, tensors)?;
        let directory = &storage.bank_templates[0];
        let indices = model_content_ranges(
            directory,
            &storage.layouts,
            original.iter().map(|tensor| {
                (
                    tensor.layout,
                    tensor.logical_begin,
                    tensor.logical_end,
                    tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                )
            }),
        )?;
        let plan = model_content_copy_plan(
            directory,
            &indices,
            &storage
                .allocations
                .iter()
                .map(TrackedCudaSlice::len)
                .collect::<Vec<_>>(),
        )?;
        let mut seals = ModelContentSeals::allocate(&self.provider, &plan)?;
        let values = original
            .iter()
            .flat_map(|tensor| [tensor.layout.role, tensor.layout.index])
            .collect::<Vec<_>>();
        let roster = allocate_publication(&self.provider, values.len())?;
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_model_seals",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "original model seal snapshot unavailable")
            })?;
        let reader = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease")
            .reader
            .view();
        seals.prepared = Some(PreparedModelContentSeals {
            storage,
            reader,
            roster,
            execute,
        });
        let owner = self.steps.get_mut(&step.token).expect("checked fixed step");
        owner.consumer_streams.insert(consumer_stream);
        let index = owner.content.len();
        owner.content.push(TensorContentBuffers {
            tensors: original,
            seals: TensorContentSeals::Model(seals),
            verification_inputs: Vec::new(),
        });
        let handle = SemanticTensorContentWitness {
            issuer: Arc::clone(&self.publication_issuer),
            reader_token: step.token,
            index,
            _witness: Arc::clone(&owner.content_witnesses),
        };
        let TensorContentSeals::Model(seals) = &owner.content[index].seals else {
            unreachable!("installed original model seals")
        };
        if let Err(error) = upload_publication(
            &self.provider,
            &values,
            &seals
                .prepared
                .as_ref()
                .expect("installed original roster")
                .roster,
        ) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(handle)
    }

    /// Attach actual original output owners as they are born during recording.
    /// Every private digest cell already exists; this operation cannot allocate
    /// device storage or establish a second baseline for native producer aliases.
    pub fn capture_prepared_tensor_content(
        &mut self,
        step: &SemanticPreparedStep,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<SemanticTensorContentWitness, SemanticTransitionError> {
        self.check_prepared_content_stream(step, consumer_stream)?;
        if tensors.is_empty() {
            return Err(publication_input_error(
                "original content requires an actual tensor roster",
            ));
        }
        let original = prepare_semantic_tensors(&self.provider, tensors)?;
        // Retain actual producers before validating coordinates or consuming
        // a reserved digest. Recording failure poisons this construction.
        let owner = self.steps.get_mut(&step.token).expect("checked fixed step");
        let index = owner.content.len();
        owner.content.push(TensorContentBuffers {
            tensors: original,
            seals: TensorContentSeals::Captured(Vec::new()),
            verification_inputs: Vec::new(),
        });
        let result = (|| {
            let owner = &self.steps[&step.token];
            let original = &owner.content[index].tensors;
            validate_content_coordinates(
                original
                    .iter()
                    .map(|tensor| (tensor.layout.role, tensor.layout.index)),
            )?;
            let mut digests = Vec::with_capacity(original.len());
            let prepared = owner.prepared.as_ref().expect("prepared content owner");
            let mut next = prepared.next_digest;
            for tensor in original {
                let mut native = owner.feedback_origin_digest(tensor)?;
                if native.is_none() {
                    let inputs = owner.inputs.as_ref().expect("fixed step inputs");
                    if inputs.owns_tensor(tensor)? {
                        native = Some(CapturedTensorDigest::Publication(Arc::clone(inputs)));
                    }
                }
                if let Some(native) = native {
                    digests.push(native);
                    continue;
                }
                for (&token, other) in &self.steps {
                    if token == step.token {
                        continue;
                    }
                    let other_input = match &other.inputs {
                        Some(inputs) => inputs.owns_tensor(tensor)?,
                        None => false,
                    };
                    if other.feedback_origin_digest(tensor)?.is_some() || other_input {
                        return Err(publication_input_error(
                            "original content belongs to another native prepared step",
                        ));
                    }
                }
                if tensor.native_allocation.is_some() {
                    return Err(publication_input_error("native allocation origin has no original content producer in this prepared step"));
                }
                tensor_content_range(
                    &tensor.layout,
                    tensor.logical_begin,
                    tensor.logical_end,
                    tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                )?;
                let cells = prepared
                    .digests
                    .as_ref()
                    .filter(|cells| next < cells.len() / 4)
                    .ok_or_else(|| {
                        publication_input_error(
                            "original tensor roster exceeds its cold native content reservation",
                        )
                    })?;
                let offset = next * 4;
                next += 1;
                digests.push(CapturedTensorDigest::Tensor {
                    cells: Arc::clone(cells),
                    offset,
                    producer_sealed: false,
                });
            }
            let owner = self.steps.get_mut(&step.token).expect("checked fixed step");
            owner
                .prepared
                .as_mut()
                .expect("prepared content owner")
                .next_digest = next;
            owner.content[index].seals = TensorContentSeals::Captured(digests);
            let handle = SemanticTensorContentWitness {
                issuer: Arc::clone(&self.publication_issuer),
                reader_token: step.token,
                index,
                _witness: Arc::clone(&owner.content_witnesses),
            };
            self.record_prepared_tensor_content(step, &handle, false)?;
            Ok(handle)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn checked_prepared_content_witness(
        &self,
        step: &SemanticPreparedStep,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.checked_prepared_step(step, false)?;
        if witness.reader_token != step.token
            || !Arc::ptr_eq(&witness.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&witness._witness, &owner.content_witnesses)
            || witness.index >= owner.content.len()
        {
            return Err(publication_input_error(
                "original content witness belongs to another native prepared step",
            ));
        }
        Ok(())
    }

    pub fn record_prepared_tensor_content(
        &mut self,
        step: &SemanticPreparedStep,
        witness: &SemanticTensorContentWitness,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        self.checked_prepared_content_witness(step, witness)?;
        let owner = &self.steps[&step.token];
        let execute = &owner
            .prepared
            .as_ref()
            .expect("prepared content owner")
            .witness;
        owner.content[witness.index].enqueue(&self.domain, &mut self.poisoned, execute, verify)
    }

    /// Bind original forward outputs after their private witness was recorded.
    /// Numerical copies and native continuation preparation are recorded later
    /// by the complete step body; fresh authority bytes are uploaded at submit.
    pub fn bind_prepared_continuation(
        &mut self,
        step: &SemanticPreparedStep,
        continuation: SemanticContinuationInput,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, consumer_stream)?;
        self.checked_prepared_content_witness(step, witness)?;
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .continuation
            .is_some()
        {
            return Err(publication_input_error(
                "original prepared continuation binds once",
            ));
        }
        if !matches!(
            self.steps[&step.token].content[witness.index].seals,
            TensorContentSeals::Captured(_)
        ) {
            return Err(publication_input_error(
                "continuation requires its original transient producer witness",
            ));
        }
        let storage = Arc::clone(self.publication.as_ref().expect("prepared publication"));
        let tensor_count = continuation.tensors.len();
        let capacity = storage
            .contract_value
            .feedback_capacity
            .checked_add(32)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let service_layouts = continuation_service_layouts(tensor_count, capacity)?;
        let SemanticContinuationInput {
            mut tensors,
            authority_decisions,
            text_rows,
            text_row_count,
            selected_text,
            active_rows,
            active_row_count,
            numerical_admissibility,
        } = continuation;
        if authority_decisions.is_empty() {
            return Err(publication_input_error(
                "continuation requires its original authority snapshot geometry",
            ));
        }
        tensors.extend(
            [
                text_rows,
                text_row_count,
                selected_text,
                active_rows,
                active_row_count,
                numerical_admissibility,
            ]
            .into_iter()
            .zip(service_layouts)
            .map(|(tensor, layout)| SemanticTensorInput {
                tensor,
                layout,
                logical_begin: 0,
                logical_end: 0,
                native_allocation: None,
            }),
        );
        self.verify_prepared_tensor_content(step, witness, tensors, consumer_stream)?;
        let owner = &self.steps[&step.token];
        let original = &owner.content[witness.index].tensors;
        let services: [PreparedSemanticTensor; 6] =
            original[tensor_count..].to_vec().try_into().map_err(|_| {
                publication_input_error(
                    "continuation witness has the wrong original service roster",
                )
            })?;
        let binding = Arc::new(TextBindingStorage {
            inputs: services,
            _witness: witness.clone(),
            consumer_stream,
            parent: TextBindingParent::Prepared(Arc::clone(
                owner.inputs.as_ref().expect("fixed inputs"),
            )),
        });
        let kind = self
            .prepared_segment
            .as_ref()
            .expect("prepared scope")
            .requested_kind(step, &self.publication_issuer)?;
        let (uploads, layouts) = prepare_continuation_payloads(
            &storage,
            &original[..tensor_count],
            &authority_decisions,
            kind,
        )?;
        let training_selection = match (
            kind,
            owner
                .prepared
                .as_ref()
                .expect("prepared selection owner")
                .training_view
                .as_ref(),
        ) {
            (SemanticTransitionKind::Update, Some(view)) => Some(view.selection()),
            (SemanticTransitionKind::Update, None) => {
                return Err(publication_input_error(
                    "prepared update continuation has no native training selection",
                ));
            }
            (_, None) => None,
            (_, Some(_)) => {
                return Err(publication_input_error(
                    "non-update continuation owns an unexpected training selection",
                ));
            }
        };
        let training_selection_pointer = training_selection
            .as_ref()
            .map_or(0, |selection| *selection.device_ptr());
        let (model_update_bindings, model_update_binding_count, model_update_admissibility) = owner
            .prepared
            .as_ref()
            .expect("prepared model update owner")
            .model_update
            .as_ref()
            .map_or((0, 0, 0), PreparedModelUpdate::descriptor);
        let continuation = PreparedContinuation::prepare(
            &self.provider,
            storage,
            owner
                .prepared
                .as_ref()
                .expect("prepared lease")
                .reader
                .view(),
            Arc::clone(&binding),
            binding.continuation_inputs(
                kind,
                authority_decisions.len() as u64,
                training_selection_pointer,
                model_update_bindings,
                model_update_binding_count,
                model_update_admissibility,
            ),
            training_selection,
            &uploads,
            &layouts,
        )?;
        self.steps
            .get_mut(&step.token)
            .expect("checked prepared step")
            .prepared
            .as_mut()
            .expect("prepared owner")
            .continuation = Some(continuation);
        Ok(())
    }

    /// Bind the complete producer-owned model backing after the original
    /// selected-view backward and optimizer update were recorded. The output is
    /// copied into the inactive neural bank by a prepared CUDA node before the
    /// sole publication transition commits its pointer and generation state.
    pub fn bind_prepared_update_output(
        &mut self,
        step: &SemanticPreparedStep,
        mut model_memory: SemanticModelMemory,
        mut tensors: Vec<SemanticTensorInput>,
        admissibility: SemanticTensorInput,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, consumer_stream)?;
        self.checked_prepared_content_witness(step, witness)?;
        if self.prepared_transition_kind(step)? != SemanticTransitionKind::Update {
            return Err(publication_input_error(
                "model update output belongs only to a prepared update",
            ));
        }
        let owner = &self.steps[&step.token];
        let prepared = owner.prepared.as_ref().expect("prepared update owner");
        if prepared
            .model_update
            .as_ref()
            .is_none_or(|update| update.output.is_some())
        {
            return Err(publication_input_error(
                "prepared update output binds exactly once",
            ));
        }
        if !matches!(
            owner.content[witness.index].seals,
            TensorContentSeals::Captured(_)
        ) {
            return Err(publication_input_error(
                "model update output requires its original transient producer witness",
            ));
        }
        let allocation_count = model_memory.allocations.len();
        if allocation_count == 0 {
            return Err(publication_input_error(
                "model update output requires complete backing allocations",
            ));
        }
        let tensor_count = tensors.len();
        model_memory.allocations.append(&mut tensors);
        model_memory.allocations.push(admissibility);
        self.verify_prepared_tensor_content(
            step,
            witness,
            model_memory.allocations,
            consumer_stream,
        )?;
        let storage = Arc::clone(self.publication.as_ref().expect("prepared publication"));
        let outputs = self.steps[&step.token].content[witness.index]
            .tensors
            .clone();
        let (allocations, outputs) = outputs.split_at(allocation_count);
        let (tensors, admissibility) = outputs.split_at(tensor_count);
        let [admissibility] = admissibility else {
            return Err(publication_input_error(
                "model update output requires one numerical admissibility predicate",
            ));
        };
        let allocation_index = u64::try_from(allocation_count)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        if admissibility.layout
            != (SemanticTensorLayout {
                role: 0,
                index: allocation_index,
                element_bytes: 1,
                scalar_type: 8,
                rank: 1,
                logical_axis: u64::MAX,
                dimensions: [1, 0, 0, 0],
                strides_bytes: [1, 0, 0, 0],
            })
            || admissibility.logical_begin != 0
            || admissibility.logical_end != 0
        {
            return Err(publication_input_error(
                "model update numerical admissibility must be Bool8[1]",
            ));
        }
        let geometry = prepare_model_memory(
            allocations,
            model_memory.storages,
            model_memory.views,
            tensors,
        )?;
        let actual_layouts = tensors
            .iter()
            .map(|tensor| ((tensor.layout.role, tensor.layout.index), tensor.layout))
            .collect::<BTreeMap<_, _>>();
        let expected_layouts = storage
            .layouts
            .iter()
            .filter(|((role, _), _)| matches!(role, 18..=25))
            .map(|(key, layout)| (*key, *layout))
            .collect::<BTreeMap<_, _>>();
        if geometry != storage.model_memory || actual_layouts != expected_layouts {
            return Err(publication_input_error(
                "model update output differs from the acquired model memory contract",
            ));
        }
        for allocation in allocations {
            let bytes = tensor_layout_bytes(&allocation.layout)?;
            if step_input_overlap(allocation.data, bytes, admissibility.data, 1)? {
                return Err(publication_input_error(
                    "model update numerical admissibility aliases an output allocation",
                ));
            }
            for owned in &storage.allocations {
                if step_input_overlap(
                    allocation.data,
                    bytes,
                    owned.device_ptr_value(),
                    owned.len(),
                )? {
                    return Err(publication_input_error(
                        "model update output aliases native publication storage",
                    ));
                }
            }
        }
        let values = allocations
            .iter()
            .zip(&storage.model_slots)
            .zip(&geometry.allocation_bytes)
            .map(|((allocation, slots), &bytes)| ModelUpdateBinding {
                source: allocation.data,
                bytes,
                slots: [slots[0] as u64, slots[1] as u64],
            })
            .collect::<Vec<_>>();
        let retained_allocations = allocations.to_vec();
        let prepared = self
            .steps
            .get_mut(&step.token)
            .expect("checked update step");
        let update = prepared
            .prepared
            .as_mut()
            .expect("prepared update owner")
            .model_update
            .as_mut()
            .expect("prepared update binding allocation");
        if values.len() != update.bindings.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        update.output = Some(BoundModelUpdate {
            values,
            allocations: retained_allocations,
            admissibility: admissibility.clone(),
            _witness: witness.clone(),
        });
        Ok(())
    }

    pub fn verify_prepared_tensor_content(
        &mut self,
        step: &SemanticPreparedStep,
        witness: &SemanticTensorContentWitness,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_content_witness(step, witness)?;
        let build = self
            .prepared_segment
            .as_ref()
            .expect("checked prepared scope");
        let recorded = !build.finished;
        if recorded {
            self.check_prepared_content_stream(step, consumer_stream)?;
        } else if !build.completed
            || !self.steps[&step.token]
                .prepared
                .as_ref()
                .expect("prepared owner")
                .observed
        {
            return Err(publication_input_error(
                "retained content requires an actually completed prepared invocation",
            ));
        }
        let actual = prepare_semantic_tensors(&self.provider, tensors)?;
        let content = &mut self
            .steps
            .get_mut(&step.token)
            .expect("checked fixed step")
            .content[witness.index];
        let start = content.verification_inputs.len();
        content.verification_inputs.extend(actual);
        let actual = &content.verification_inputs[start..];
        if content.tensors.len() != actual.len()
            || content
                .tensors
                .iter()
                .zip(actual)
                .any(|(expected, actual)| !same_tensor_content_owner(expected, actual))
        {
            self.poisoned = true;
            return Err(publication_input_error("content verification requires the exact original tensor owners, layout and interval"));
        }
        if recorded {
            self.record_prepared_tensor_content(step, witness, true)
        } else {
            self.enqueue_content_witness(witness, consumer_stream, true)
        }
    }

    pub fn record_prepared_step_admission(
        &mut self,
        step: &SemanticPreparedStep,
        conditional_handle: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        if conditional_handle == 0
            || !self
                .prepared_segment
                .as_ref()
                .expect("checked build")
                .capturing
        {
            return Err(publication_input_error(
                "step admission requires the actual conditional capture handle",
            ));
        }
        let requested_kind = self
            .prepared_segment
            .as_mut()
            .expect("checked build")
            .enter(step, &self.publication_issuer)?;
        let storage = self.publication.as_ref().expect("prepared publication");
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease owner");
        let build = self
            .prepared_segment
            .as_ref()
            .expect("checked prepared scope");
        let previous = build.next.checked_sub(1).map(|index| {
            &self.steps[&build.tokens[index]]
                .prepared
                .as_ref()
                .expect("previous prepared step")
                .result
        });
        let mut recorder = self.domain.new_strict_recorder();
        storage.record(&mut recorder);
        recorder.write(&prepared.reader);
        if let Some(previous) = previous {
            recorder.read(previous);
        }
        let arguments = (
            storage.control.device_ptr_value(),
            prepared.reader.device_ptr_value(),
            requested_kind.code(),
            conditional_handle,
            previous.map_or(0, TrackedCudaSlice::device_ptr_value),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: actual parent control and unique lease are fixed owners;
            // the graph issues this IF handle and the kernel writes every replay.
            unsafe {
                prepared.admit.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    pub fn record_prepared_step_inputs(
        &mut self,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .inputs_recorded
        {
            return Err(publication_input_error(
                "prepared input producer records exactly once",
            ));
        }
        // The complete original read roster is validated before feedback,
        // codebook, causal, or numerical consumers can read any parent value.
        let keys = self.prepared_range_keys(step)?;
        self.guard_prepared_content(step, &keys, self.stream.cu_stream() as usize as u64)?;
        let owner = &self.steps[&step.token];
        let descriptor = self.descriptor();
        owner
            .inputs
            .as_ref()
            .expect("fixed input owner")
            .enqueue(&self.domain, &mut self.poisoned)?;
        owner.feedback[0].enqueue(
            &self.domain,
            &mut self.poisoned,
            &self.graph,
            self.publication.as_ref().expect("prepared publication"),
            &self
                .task
                .as_ref()
                .ok_or(SemanticTransitionError::InvalidTaskBank)?
                .1,
            &owner.prepared.as_ref().expect("prepared lease").reader,
            descriptor,
        )?;
        for content in &owner.content {
            if let TensorContentSeals::Model(seals) = &content.seals {
                seals.snapshot_prepared(&self.domain, &mut self.poisoned)?;
                content.enqueue(
                    &self.domain,
                    &mut self.poisoned,
                    &owner
                        .prepared
                        .as_ref()
                        .expect("prepared witness producer")
                        .witness,
                    true,
                )?;
            }
        }
        let selection = if let Some(selected) = owner
            .prepared
            .as_ref()
            .expect("prepared selection owner")
            .training_view
            .as_ref()
        {
            let selected_view = owner
                .inputs
                .as_ref()
                .expect("fixed input owner")
                .views
                .first()
                .expect("two publication input banks")
                .get(&(SemanticStateRole::TrainingView as u64, 0))
                .cloned()
                .ok_or_else(|| {
                    publication_input_error("prepared update has no acquired training-view input")
                })?;
            let coordinates = owner
                .inputs
                .as_ref()
                .expect("fixed input owner")
                .training_coordinates_view()?;
            selected.enqueue_device_selection(selected_view, coordinates)
        } else {
            Ok(())
        };
        if selection.is_err() {
            self.poisoned = true;
            return selection;
        }
        self.steps
            .get_mut(&step.token)
            .expect("checked step")
            .prepared
            .as_mut()
            .expect("prepared owner")
            .inputs_recorded = true;
        Ok(())
    }

    fn record_prepared_step_kind_gate(
        &mut self,
        step: &SemanticPreparedStep,
        expected_kind: SemanticTransitionKind,
        conditional_handle: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if conditional_handle == 0 {
            return Err(publication_input_error(
                "prepared kind gate requires the actual conditional capture handle",
            ));
        }
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease owner");
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&prepared.reader);
        let arguments = (
            prepared.reader.device_ptr_value(),
            expected_kind.code(),
            conditional_handle,
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the lease and conditional handle belong to this original
            // prepared step and remain live throughout the captured segment.
            unsafe {
                prepared.kind_gate.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    pub fn record_prepared_step_requested_bank_gate(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
        conditional_handle: u64,
    ) -> Result<(), SemanticTransitionError> {
        let expected = self.prepared_transition_kind(step)?;
        self.checked_prepared_step(step, true)?;
        if bank > 1 || conditional_handle == 0 {
            return Err(publication_input_error(
                "prepared bank gate requires bank zero or one and the actual conditional capture handle",
            ));
        }
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease owner");
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&prepared.reader);
        let arguments = (
            prepared.reader.device_ptr_value(),
            expected.code(),
            bank as u64,
            conditional_handle,
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the device lease selects exactly one resident input bank;
            // this conditional exposes neither the lease nor its bank to host code.
            unsafe {
                prepared.kind_bank_gate.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    #[cfg(feature = "semantic-policy")]
    pub fn begin_prepared_step_bank_capture(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        let prepared = self
            .steps
            .get_mut(&step.token)
            .expect("checked prepared bank step")
            .prepared
            .as_mut()
            .expect("prepared bank owner");
        let recorded = prepared.transition_recorded;
        let result = prepared
            .model_work
            .as_mut()
            .ok_or_else(|| publication_input_error("prepared bank requires original model work"))
            .and_then(|work| {
                work.begin_capture(bank, recorded)
                    .map_err(publication_input_error)
            });
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    pub fn record_prepared_step_drain_gate(
        &mut self,
        step: &SemanticPreparedStep,
        conditional_handle: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.record_prepared_step_kind_gate(step, SemanticTransitionKind::Drain, conditional_handle)
    }

    pub fn record_prepared_step_active_gate(
        &mut self,
        step: &SemanticPreparedStep,
        conditional_handle: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if conditional_handle == 0 {
            return Err(publication_input_error(
                "prepared release gate requires the actual conditional capture handle",
            ));
        }
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease owner");
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&prepared.reader);
        let arguments = (prepared.reader.device_ptr_value(), conditional_handle);
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: this device predicate exposes no lease value to the host;
            // it only releases an actually acquired original step.
            unsafe {
                prepared.active_gate.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
    }

    pub fn guard_prepared_content(
        &mut self,
        step: &SemanticPreparedStep,
        keys: &[(SemanticStateRole, u64)],
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        if dlpack_consumer_stream(consumer_stream)? != self.stream.cu_stream() || keys.is_empty() {
            return Err(publication_input_error(
                "prepared content guard requires original ranges on the native capture stream",
            ));
        }
        let storage = self.publication.as_ref().expect("prepared publication");
        let owner = &self.steps[&step.token];
        let prepared = owner.prepared.as_ref().expect("prepared lease");
        let mut distinct = BTreeSet::new();
        for &(role, index) in keys {
            if !distinct.insert((role as u64, index))
                || !storage.bank_templates[0]
                    .iter()
                    .any(|range| (range.role, range.index) == (role as u64, index))
            {
                return Err(publication_input_error(
                    "prepared guard names duplicate or undeclared original ranges",
                ));
            }
            let layout = if is_tensor_role(role as u64) {
                *storage
                    .layouts
                    .get(&(role as u64, index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
            } else {
                SemanticTensorLayout::default()
            };
            storage.enqueue_content_guard(
                &self.domain,
                &mut self.poisoned,
                prepared.reader.view(),
                &prepared.content_guard,
                role as u64,
                index,
                layout,
            )?;
        }
        if prepared.inputs_recorded {
            owner
                .inputs
                .as_ref()
                .expect("fixed input owner")
                .verify(&self.domain, &mut self.poisoned)?;
            owner.feedback[0].content.enqueue(
                &self.domain,
                &mut self.poisoned,
                &prepared.witness,
                true,
            )?;
        }
        Ok(())
    }

    pub fn record_prepared_step_release(
        &mut self,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, true)?;
        let storage = self.publication.as_ref().expect("prepared publication");
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared lease");
        if prepared.transition_recorded != PREPARED_TRANSITION_BANKS || !prepared.drain_recorded {
            return Err(publication_input_error(
                "release must follow this prepared step's complete native transition and drain branches",
            ));
        }
        let mut recorder = self.domain.new_strict_recorder();
        storage.record(&mut recorder);
        recorder.read_write(&prepared.reader);
        recorder.read(
            &self.steps[&step.token]
                .inputs
                .as_ref()
                .expect("fixed input owner")
                .header,
        );
        recorder.write(&prepared.result);
        if let Some(origin) = &prepared.training_origin {
            recorder.write(origin);
        }
        let arguments = (
            storage.control.device_ptr_value(),
            prepared.reader.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: save the actual successor or refusal while the exact
            // acquired parent and current bank still belong to this step.
            unsafe {
                prepared.result_kernel.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        storage.control.device_ptr_value(),
                        prepared.reader.device_ptr_value(),
                        prepared.result.device_ptr_value(),
                        self.steps[&step.token]
                            .inputs
                            .as_ref()
                            .expect("fixed input owner")
                            .header
                            .device_ptr_value(),
                        prepared
                            .training_origin
                            .as_ref()
                            .map_or(0, |origin| *origin.device_ptr()),
                    ),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
            // SAFETY: release follows this step's body in the actual IF graph;
            // the native lease remains owned for all later retained witnesses.
            unsafe {
                prepared.release.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        self.prepared_segment
            .as_mut()
            .expect("checked build")
            .leave(step, &self.publication_issuer)
    }

    pub fn finish_prepared_segment(
        &mut self,
        executable: &mut Option<SemanticPreparedExecutable>,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        let build = self
            .prepared_segment
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        if self.captured.is_some()
            || executable
                .as_ref()
                .is_none_or(|executable| !Arc::ptr_eq(&executable.scope, &build.scope))
        {
            return Err(publication_input_error(
                "executable differs from the original native segment construction",
            ));
        }
        build.finish()?;
        // Upload only immutable first-recording geometry, after capture and
        // before any graph launch. The original step retains this allocation.
        for token in &build.tokens {
            let prepared = self.steps[token]
                .prepared
                .as_ref()
                .expect("original prepared owner");
            let work = prepared.model_work.as_ref().ok_or_else(|| {
                publication_input_error("prepared body has no frozen original model work")
            })?;
            if work.recording.frozen_bound().is_none() {
                return Err(publication_input_error(
                    "original model work was not frozen by its transition",
                ));
            }
            let mut destination = work.device.view().slice(..work.recording.events().len());
            self.provider
                .htod_launch_metadata_sync_copy_into(work.recording.events(), &mut destination)
                .map_err(|error| runtime_error("original model work cold upload", error))?;
            if let Some(update) = &prepared.model_update {
                let output = update.output.as_ref().ok_or_else(|| {
                    publication_input_error("prepared update has no original model output binding")
                })?;
                let mut destination = update.bindings.view();
                self.provider
                    .htod_launch_metadata_sync_copy_into(&output.values, &mut destination)
                    .map_err(|error| runtime_error("model update binding cold upload", error))?;
            }
        }
        // Failure leaves the actual graph in the caller's owner. No graph
        // destructor or driver completion can run under its Session mutex.
        self.captured = Some(
            executable
                .take()
                .expect("checked authentic executable")
                .graph,
        );
        Ok(())
    }

    /// Detach the completed executable so the caller can destroy it outside its
    /// Session mutex. The actual allocation resources remain Session-owned until
    /// every original prepared step and its final consumers have retired.
    pub fn take_prepared_executable_for_retirement(
        &mut self,
    ) -> Result<Option<CapturedCudaGraph>, SemanticTransitionError> {
        if self
            .prepared_segment
            .as_ref()
            .is_none_or(|build| !build.completed)
        {
            return Err(publication_input_error(
                "prepared executable retirement requires known actual completion",
            ));
        }
        Ok(self.captured.take())
    }

    /// Return the original pool/model owners only after all joined step owners
    /// and their imported aliases are gone. Drop the returned owners unlocked.
    pub fn take_prepared_resources_for_retirement(
        &mut self,
    ) -> Result<Vec<Arc<dyn Send + Sync>>, SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !build.completed
            || self.captured.is_some()
            || build
                .tokens
                .iter()
                .any(|token| self.steps.contains_key(token))
        {
            return Err(publication_input_error(
                "original prepared resources still own an executable or retained step",
            ));
        }
        let resources = std::mem::take(&mut self.prepared_resources);
        self.prepared_segment = None;
        Ok(resources)
    }

    #[cfg(feature = "semantic-policy")]
    fn prepared_segment_recorder(&self) -> Result<LaunchRecorder, SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        // Shared arena, scratch, immutable catalogue and the actual feedback
        // descriptor owners remain part of this same strict recorded launch.
        let mut recorder = self.kernel_recorder();
        self.publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .record(&mut recorder);
        for token in &build.tokens {
            let owner = self
                .steps
                .get(token)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let prepared = owner
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            recorder.read_write(&prepared.reader);
            recorder.write(&prepared.result);
            if let Some(work) = &prepared.model_work {
                recorder.read(&work.device);
                recorder.read_write(&work.actual);
            }
            if let Some(digests) = &prepared.digests {
                recorder.read_write(digests.as_ref());
            }
            let inputs = owner
                .inputs
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            for bindings in &inputs.bindings {
                recorder.read(bindings);
            }
            recorder.write(&inputs.header);
            recorder.write(&inputs.source);
            recorder.write(&inputs.ranges);
            recorder.write(&inputs.metadata_digests);
            for allocation in &inputs.private {
                recorder.write(allocation);
            }
            for bank in &inputs.views {
                for view in bank.values() {
                    recorder.read_write(view);
                }
            }
            for feedback in &owner.feedback {
                recorder.write(&feedback.status);
            }
            for content in owner
                .content
                .iter()
                .chain(owner.feedback.iter().map(|feedback| &feedback.content))
            {
                for tensor in content.tensors.iter().chain(&content.verification_inputs) {
                    if let Some(source) = &tensor.source {
                        recorder.read_write(source);
                    }
                }
                match &content.seals {
                    TensorContentSeals::Captured(digests) => {
                        for digest in digests {
                            if let CapturedTensorDigest::Tensor { cells, .. } = digest {
                                recorder.read_write(cells.as_ref());
                            }
                        }
                    }
                    TensorContentSeals::Model(model) => {
                        recorder.read_write(&model.ranges);
                        recorder.read_write(&model.contract);
                        if let Some(source) = &model.prepared {
                            recorder.read(&source.roster);
                            recorder.read(&source.reader);
                        }
                    }
                }
            }
            if let Some(continuation) = &prepared.continuation {
                continuation.text_binding.record(&mut recorder);
                for tensor in &continuation.sources {
                    if let Some(source) = &tensor.source {
                        recorder.read(source);
                    }
                }
            }
            #[cfg(feature = "semantic-policy")]
            {
                recorder.write(&prepared.state);
                recorder.write(
                    prepared
                        .support
                        .as_ref()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                );
                recorder.write(
                    prepared
                        .receipts
                        .as_ref()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                );
                if let Some(policy) = &prepared.policy {
                    for buffer in [
                        &policy.parameters,
                        &policy.hidden,
                        &policy.scores,
                        &policy.recurrent,
                        &policy.text_logits,
                    ] {
                        recorder.read_write(buffer);
                    }
                    policy.text_binding.record(&mut recorder);
                } else {
                    recorder.read(
                        &prepared
                            .policy_buffers
                            .as_ref()
                            .ok_or(SemanticTransitionError::NotBound)?
                            .text_logits,
                    );
                }
                for tensor in &prepared.policy_sources {
                    if let Some(source) = &tensor.source {
                        recorder.read(source);
                    }
                }
            }
        }
        Ok(recorder)
    }

    /// Snapshot every supplied cold owner into fixed publication storage, then
    /// initialize and seal bank zero through the actual native command.
    pub fn bind_parent(
        &mut self,
        mut parent: SemanticParentBinding,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        // The handoff guard owns all managed inputs before any later validation
        // can return. Failed cold admission cannot drop an unfinished producer.
        let tensor_count = parent.tensors.len();
        let mut producers = std::mem::take(&mut parent.tensors);
        producers.append(&mut parent.model_memory.allocations);
        let producers = complete_tensor_handoff(&self.provider, producers)?;
        let mut tensors = prepare_semantic_tensors(&self.provider, producers)?;
        let allocations = tensors.split_off(tensor_count);
        let model_memory = prepare_model_memory(
            &allocations,
            std::mem::take(&mut parent.model_memory.storages),
            std::mem::take(&mut parent.model_memory.views),
            &tensors,
        )?;
        self.ensure_rebindable()?;
        if self.publication.is_some() {
            return Err(publication_input_error(
                "this Session already owns a canonical publication instance",
            ));
        }
        let task_identity = self
            .task_evaluation_identity()
            .ok_or(SemanticTransitionError::NotBound)?;
        if parent.authority_generation != self.task_epoch {
            return Err(publication_input_error(
                "parent authority generation differs from the retained task import",
            ));
        }
        if parent.recovered_instance.is_some()
            || parent.records.iter().any(|record| {
                matches!(
                    record.role,
                    SemanticStateRole::IntentEntries | SemanticStateRole::IntentPayload
                )
            })
        {
            return Err(publication_input_error("fresh parent initialization cannot replace recovered history or accept private intent records"));
        }
        parent.records.extend(initial_intent_records(
            &parent.intent_effect,
            parent.intent_entry_capacity,
            parent.intent_payload_capacity_bytes,
            parent.final_intent_payload_bytes,
        )?);
        validate_parent_records(&parent, &tensors)?;
        let mut layouts = BTreeMap::new();
        let mut plans = Vec::new();
        for tensor in tensors {
            let layout = if matches!(tensor.layout.role, 18..=25) {
                tensor.layout
            } else if matches!(tensor.layout.role, 4 | 5) {
                prefix_capacity_layout(tensor.layout, parent.prefix_capacity)?
            } else {
                canonical_tensor_layout(tensor.layout)?
            };
            let bytes = tensor_layout_bytes(&layout)?;
            if bytes == 0 && !empty_model_tensor_layout(&layout) {
                return Err(publication_input_error(
                    "non-active tensor destination has no owned extent",
                ));
            }
            layouts.insert((layout.role, layout.index), layout);
            let capacity = if matches!(layout.role, 18..=25) {
                let (allocation, offset) = model_memory.location(layout.role, layout.index)?;
                model_memory.allocation_bytes[allocation] as usize - offset
            } else {
                bytes
            };
            plans.push(PublicationAllocationPlan {
                role: layout.role,
                index: layout.index,
                capacity,
                length: bytes,
                logical_begin: tensor.logical_begin,
                logical_end: tensor.logical_end,
                payload: if matches!(layout.role, 18..=25) {
                    PublicationPayload::Uncomputed
                } else {
                    PublicationPayload::Tensor(tensor)
                },
            });
        }
        for layout in &parent.active_layouts {
            let layout = canonical_tensor_layout(*layout)?;
            let bytes = tensor_layout_bytes(&layout)?;
            layouts.insert((layout.role, layout.index), layout);
            plans.push(PublicationAllocationPlan {
                role: layout.role,
                index: layout.index,
                capacity: bytes,
                length: 0,
                logical_begin: 0,
                logical_end: 0,
                payload: PublicationPayload::Uncomputed,
            });
        }
        for record in &parent.records {
            plans.push(PublicationAllocationPlan {
                role: record.role as u64,
                index: record.index,
                capacity: record.capacity_bytes,
                length: record.bytes.len(),
                logical_begin: 0,
                logical_end: if record.role == SemanticStateRole::TokenProvenanceRecords {
                    parent.provenance_records
                } else {
                    0
                },
                payload: PublicationPayload::Metadata(record.bytes.clone()),
            });
        }
        let statements = self.feedback_statement_bytes()?.map(<[u8]>::to_vec);
        let layout_rows: Vec<_> = layouts.values().copied().collect();
        let prefix_capacity = usize::try_from(parent.prefix_capacity)
            .ok()
            .and_then(|n| n.checked_mul(size_of::<SemanticTextSlot>()))
            .ok_or_else(|| publication_input_error("prefix source capacity overflow"))?;
        let feedback_capacity = usize::try_from(parent.feedback_capacity)
            .ok()
            .and_then(|n| n.checked_mul(size_of::<RawFeedbackRecord>()))
            .ok_or_else(|| publication_input_error("feedback capacity overflow"))?;
        for (role, index, bytes, capacity, end) in [
            (
                1,
                0,
                publication_abi_bytes(&parent.prefix),
                prefix_capacity,
                parent.prefix.len() as u64,
            ),
            // Output storage only; native initialization fills the actual
            // source digest before this bank can be sealed or acquired.
            (3, 0, vec![0; 32], 32, 0),
            (
                14,
                0,
                vec![0; size_of::<CompletionCoverage>()],
                size_of::<CompletionCoverage>(),
                0,
            ),
            (15, 0, vec![0; feedback_capacity], feedback_capacity, 0),
            (16, 0, statements[0].clone(), statements[0].len(), 0),
            (16, 1, statements[1].clone(), statements[1].len(), 0),
            (16, 2, statements[2].clone(), statements[2].len(), 0),
            (
                33,
                0,
                vec![0; size_of::<AttemptReceipt>()],
                size_of::<AttemptReceipt>(),
                0,
            ),
            (
                47,
                0,
                include_bytes!("semantic_action_catalogue_v1.def").to_vec(),
                include_bytes!("semantic_action_catalogue_v1.def").len(),
                0,
            ),
            (
                48,
                0,
                publication_abi_bytes(&self.codebooks.words),
                self.codebooks.words.len() * 8,
                0,
            ),
            (
                55,
                0,
                tensor_table_bytes(&layout_rows, false, &[])?,
                tensor_table_capacity(layout_rows.len(), 32 + parent.feedback_capacity)?,
                0,
            ),
        ] {
            plans.push(PublicationAllocationPlan {
                role,
                index,
                capacity,
                length: bytes.len(),
                logical_begin: 0,
                logical_end: end,
                payload: PublicationPayload::Metadata(bytes),
            });
        }
        let mut instance = [0u8; 32];
        getrandom::fill(&mut instance)
            .map_err(|error| runtime_error("publication instance entropy", error))?;
        let instance = Identity256::from_bytes(instance);
        let contract = PublicationContract {
            abi: 1,
            range_capacity: plans.len() as u64 + 1,
            prefix_capacity: parent.prefix_capacity,
            window_capacity: 32,
            feedback_capacity: parent.feedback_capacity,
            max_position: parent.max_position,
            pad_token: parent.pad_token,
            terminal_token_count: parent.terminal_tokens.len() as u64,
            final_intent_payload_bytes: parent.final_intent_payload_bytes,
            model_generation: parent.model_generation,
            policy_generation: parent.policy_generation,
            authority_generation: parent.authority_generation,
            semantic_owner: self.graph.transition_arena()[1],
            task_identity,
            topology_identity: parent.topology_identity,
            table_identity: parent.table_identity,
            model_contract_layout: parent.model_contract_layout,
            role_count: 55,
            ..PublicationContract::default()
        };
        let runtime = initial_runtime_contract_record(
            contract,
            &parent.role_counts,
            &parent.terminal_tokens,
            &layouts,
            plans
                .iter()
                .map(|plan| (plan.role, plan.index, plan.capacity))
                .collect(),
            &parent.model_numerical_mode,
            &model_memory,
        )?;
        plans.push(PublicationAllocationPlan {
            role: runtime.role as u64,
            index: runtime.index,
            capacity: runtime.capacity_bytes,
            length: runtime.bytes.len(),
            logical_begin: 0,
            logical_end: 0,
            payload: PublicationPayload::Metadata(runtime.bytes),
        });
        plans.sort_by_key(|plan| (plan.role, plan.index));
        let model_geometry_digest = model_memory.digest(&layouts)?;
        let model_payloads = allocations
            .into_iter()
            .map(PublicationPayload::Tensor)
            .collect::<Vec<_>>();
        let (storage, uploads) = PublicationStorage::allocate(
            &self.provider,
            &plans,
            layouts,
            model_memory,
            &model_payloads,
            contract,
            &parent.terminal_tokens,
            instance,
        )?;
        // Install all destinations and producer owners before the first command
        // that may enqueue. A partial copy failure keeps these owners in Session.
        self.publication = Some(Arc::new(storage));
        self.publication_uploads = uploads;
        self.captured = None;
        let extents = self.base_snapshot.extents();
        let header = PublicationHeader {
            instance,
            ring_head: parent.ring_head,
            prefix_extent: parent.prefix.len() as u64,
            semantic_owner: self.graph.transition_arena()[1],
            semantic_slot: u64::from(self.root.slot()),
            semantic_generation: self.root.generation(),
            semantic_digest: Identity256::from_bytes(*self.base_snapshot.digest().as_bytes()),
            semantic_extents: [
                u64::from(extents.statements()),
                u64::from(extents.supports()),
                u64::from(extents.versions()),
            ],
            model_generation: parent.model_generation,
            neural_generation: parent.neural_generation,
            cache_generation: parent.cache_generation,
            authority_generation: parent.authority_generation,
            fuel: parent.fuel,
            proposal: u64::from(parent.rng.proposal),
            stream_serial: parent.rng.stream_serial,
            family_id: u64::from(parent.rng.family_id),
            training_cursor: parent.training_cursor,
            training_rng: parent.training_rng,
            range_count: contract.range_capacity,
            model_geometry_digest,
            ..PublicationHeader::default()
        };
        let state = DeviceState {
            model_generation: parent.rng.model_generation,
            family_id: u32::from(parent.rng.family_id),
            stream_serial: parent.rng.stream_serial,
            next_proposal: u64::from(parent.rng.proposal),
            catalogue_generation: self.binding().generation,
            catalogue_digest: Identity256::from_bytes(CATALOGUE_DIGEST),
            binding_digest: self.binding().digest,
            ..DeviceState::default()
        };
        let bank = PublicationBank {
            header,
            source: parent.source,
            state,
            receipts: [SemanticTransitionReceipt::default(); COMPONENT_COUNT],
        };
        let result = self.initialize_publication(
            bank,
            &parent.terminal_tokens,
            &parent.role_counts,
            parent.rng,
            false,
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Reconstruct native material into this fresh admitted owner. This performs
    /// actual graph insert/seal and native state hashing; it does not establish
    /// an episode's provenance, current use rights, or model replay equivalence.
    /// The trusted execution importer must verify those before issuing TaskUse.
    pub fn restore_state_material(
        &mut self,
        bytes: &[u8],
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some() || self.captured.is_some() || self.task.is_none() {
            return Err(publication_input_error(
                "state restoration requires a fresh uncaptured task-bound Session",
            ));
        }
        let mut material = PublicationMaterial::decode(bytes)?;
        let header = material.bank.header;
        let rng = header.rng_binding()?;
        if header.authority_generation == 0 {
            return Err(publication_input_error(
                "restored RNG or task generation exceeds its native domain",
            ));
        }
        // Validate immutable task statements before any graph mutation. Symbols
        // are compared through admitted canonical bytes, never process IDs.
        for (index, statement) in self.feedback_statement_bytes()?.iter().enumerate() {
            let saved = material
                .ranges
                .iter()
                .find(|item| item.range.role == 16 && item.range.index == index as u64)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if saved.bytes != *statement {
                return Err(publication_input_error(
                    "restored feedback statement differs from the actual admitted task",
                ));
            }
        }
        let result = (|| {
            let root = self
                .graph
                .restore_root(&material.graph)
                .map_err(SemanticTransitionError::Semantic)?;
            let snapshot = self
                .graph
                .snapshot(crate::SemanticView::Root(root))
                .map_err(SemanticTransitionError::Semantic)?;
            let admission = self
                .graph
                .admission()
                .ok_or(SemanticTransitionError::NotBound)?;
            let codebooks = ActionCodebooks::derive(admission, self.graph.transition_arena()[1])?;
            if codebooks.input_cells != self.codebooks.input_cells
                || codebooks.words.len() != self.device_codebooks.len()
            {
                return Err(publication_input_error(
                    "restored action catalogue changes its allocated shape",
                ));
            }
            let (task, device) = self
                .task
                .as_mut()
                .ok_or(SemanticTransitionError::NotBound)?;
            task.admission_identity = admission.identity();
            if task.identity() != material.contract.task_identity {
                return Err(publication_input_error(
                    "restored native task differs from its original complete admission",
                ));
            }
            let words = task.words(self.graph.transition_arena()[1]);
            self.provider
                .htod_sync_copy_into_tracked(&words, device)
                .map_err(|error| runtime_error("restored task owner upload", error))?;
            let codebook = material
                .ranges
                .iter_mut()
                .find(|item| item.range.role == 48)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            codebook.bytes = relocate_publication_codebooks(&codebook.bytes, &codebooks.words)?;
            self.codebooks = codebooks;
            self.upload_cold_codebooks()?;
            self.root = root;
            self.base_snapshot = snapshot;
            self.task_epoch = header.authority_generation;
            let mut instance = [0; 32];
            getrandom::fill(&mut instance)
                .map_err(|error| runtime_error("restored instance entropy", error))?;
            let instance = Identity256::from_bytes(instance);
            if instance == header.instance {
                return Err(publication_input_error(
                    "restoration must create a fresh instance",
                ));
            }
            let mut contract = material.contract;
            contract.semantic_owner = self.graph.transition_arena()[1];
            let plans: Vec<_> = material
                .ranges
                .into_iter()
                .map(|item| PublicationAllocationPlan {
                    role: item.range.role,
                    index: item.range.index,
                    capacity: item.capacity,
                    length: item.bytes.len(),
                    logical_begin: item.range.logical_begin,
                    logical_end: item.range.logical_end,
                    payload: if matches!(item.range.role, 18..=25) {
                        PublicationPayload::Uncomputed
                    } else {
                        PublicationPayload::Metadata(item.bytes)
                    },
                })
                .collect();
            let model_payloads = material
                .model_allocations
                .into_iter()
                .map(PublicationPayload::Metadata)
                .collect::<Vec<_>>();
            let (storage, uploads) = PublicationStorage::allocate(
                &self.provider,
                &plans,
                material.layouts,
                material.model_memory,
                &model_payloads,
                contract,
                &material.terminals,
                instance,
            )?;
            material.bank.header.abi = 0;
            material.bank.header.instance = instance;
            material.bank.header.recovered_instance = header.instance;
            material.bank.header.sealed_epoch = 0;
            material.bank.header.base_word = 0;
            material.bank.header.publication_word = 0;
            material.bank.header.semantic_owner = contract.semantic_owner;
            material.bank.header.semantic_slot = u64::from(root.slot());
            material.bank.header.semantic_generation = root.generation();
            material.bank.header.neural_bank = 0;
            // Historical receipts and feedback stay unchanged. Only the live
            // descriptor is rebound by initialize_publication; its kernel checks
            // fresh native hashes against the saved logical/state expectations.
            self.publication = Some(Arc::new(storage));
            self.publication_uploads = uploads;
            let restored = self.initialize_publication(
                material.bank,
                &material.terminals,
                &material.role_counts,
                rng,
                true,
            )?;
            if restored.logical_digest != header.logical_digest
                || restored.state_digest != header.state_digest
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            Ok(restored)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn initialize_publication(
        &mut self,
        bank: PublicationBank,
        terminal_tokens: &[u64],
        role_counts: &[u64; 55],
        rng: SemanticRngBinding,
        restored: bool,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .expect("installed publication owner"),
        );
        let mut recorder = self.domain.new_strict_recorder();
        storage.record(&mut recorder);
        for (_, payload) in &self.publication_uploads {
            if let PublicationPayload::Tensor(tensor) = payload {
                if let Some(source) = &tensor.source {
                    recorder.read(source);
                }
            }
        }
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            for allocation in &storage.allocations {
                // A genuine empty model buffer retains an owner and layout,
                // but has no device pointer on which to submit a memory write.
                if allocation.is_empty() {
                    continue;
                }
                // SAFETY: all actual fixed allocations were recorded before this
                // first may-enqueue boundary; zero bytes do not mark a bank valid.
                unsafe {
                    sys::cuMemsetD8Async(
                        allocation.device_ptr_value(),
                        0,
                        allocation.len(),
                        enqueue.stream().cu_stream(),
                    )
                }
                .result()
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            }
            for (slot, payload) in &self.publication_uploads {
                if let PublicationPayload::Tensor(tensor) = payload {
                    if let Some(source) = &tensor.source {
                        let destination = &storage.allocations[*slot];
                        let layout = if tensor.layout.role == 0 {
                            &tensor.layout
                        } else {
                            &storage.layouts[&(tensor.layout.role, tensor.layout.index)]
                        };
                        for (source_offset, destination_offset, bytes) in
                            tensor_copy_plan(&tensor.layout, layout)
                                .map_err(|error| XlogError::Kernel(error.to_string()))?
                        {
                            unsafe {
                                sys::cuMemcpyDtoDAsync_v2(
                                    destination.device_ptr_value() + destination_offset,
                                    *source.device_ptr() + source_offset,
                                    bytes,
                                    enqueue.stream().cu_stream(),
                                )
                            }
                            .result()
                            .map_err(|error| XlogError::Kernel(error.to_string()))?;
                        }
                    }
                }
            }
            Ok::<(), XlogError>(())
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "cold publication tensor snapshot",
            CudaStream::synchronize,
        )?;
        for (slot, payload) in &self.publication_uploads {
            if let PublicationPayload::Metadata(bytes) = payload {
                if !bytes.is_empty() {
                    let mut destination = storage.allocations[*slot].view().slice(..bytes.len());
                    self.provider
                        .htod_launch_metadata_sync_copy_into(bytes, &mut destination)
                        .map_err(|error| runtime_error("cold publication record upload", error))?;
                }
            }
        }
        upload_publication(&self.provider, &[bank], &storage.banks[0])?;
        upload_publication(&self.provider, &[bank], &storage.banks[1])?;
        for bank in 0..2 {
            upload_publication(
                &self.provider,
                &storage.bank_templates[bank],
                &storage.directories[bank],
            )?;
        }
        let entries: Vec<_> = storage
            .allocations
            .iter()
            .map(|allocation| PublicationStorageEntry {
                pointer: allocation.device_ptr_value(),
                bytes: allocation.len() as u64,
                generation: 1,
            })
            .collect();
        upload_publication(&self.provider, &entries, &storage.storage)?;
        upload_publication(&self.provider, &[storage.contract_value], &storage.contract)?;
        upload_publication(&self.provider, terminal_tokens, &storage.terminals)?;
        let counts: Vec<_> = role_counts
            .iter()
            .enumerate()
            .map(|(index, &count)| PublicationRoleCount {
                role: index as u64 + 1,
                count,
            })
            .collect();
        upload_publication(&self.provider, &counts, &storage.role_counts)?;
        upload_publication(
            &self.provider,
            &[PendingContinuation {
                abi: 1,
                ranges: storage.continuation_directory.device_ptr_value(),
                range_count: storage.continuation_templates.len() as u64,
                ..PendingContinuation::default()
            }],
            &storage.continuation,
        )?;
        upload_publication(
            &self.provider,
            &storage.continuation_templates,
            &storage.continuation_directory,
        )?;
        upload_publication(
            &self.provider,
            &[PublicationControl {
                abi: 1,
                instance: storage.instance,
                banks: storage.banks.each_ref().map(|bank| bank.device_ptr_value()),
                directories: storage
                    .directories
                    .each_ref()
                    .map(|directory| directory.device_ptr_value()),
                storage: storage.storage.device_ptr_value(),
                storage_count: storage.allocations.len() as u64,
                contract: storage.contract.device_ptr_value(),
                continuation: storage.continuation.device_ptr_value(),
                ..PublicationControl::default()
            }],
            &storage.control,
        )?;
        self.upload_rng_state(self.binding(), rng)?;
        self.publication_command(if restored { 4 } else { 1 }, None)?;
        let control = self.publication_read(storage.control.view())?[0];
        if control.refusal != 0 {
            return Err(SemanticTransitionError::PublicationRefused {
                status: control.refusal,
            });
        }
        let header = self.read_publication_header(0)?;
        if header.abi != 1
            || header.instance != storage.instance
            || header.publication_word != 0
            || control.word != 0
            || header.model_geometry_digest != bank.header.model_geometry_digest
            || (restored && header.model_numerical_digest != bank.header.model_numerical_digest)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.publication_uploads.clear();
        self.rng = Some(rng);
        self.next_proposal = u64::from(rng.proposal);
        Ok(SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        })
    }

    fn publication_command(
        &mut self,
        operation: u64,
        token: Option<u64>,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let mut descriptor = self.descriptor();
        descriptor.publication = PublicationCommand {
            control: storage.control.device_ptr_value(),
            operation,
            lease: token.map_or(0, |token| self.readers[&token].device.device_ptr_value()),
        };
        let mut recorder = self.kernel_recorder();
        if let Some(token) = token {
            recorder.read_write(&self.readers[&token].device);
        }
        let execute = self.execute.clone();
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |stream| {
            // SAFETY: command, bank directories, storage and optional lease all
            // have actual recorded owners, and the common descriptor ABI is fixed.
            unsafe {
                execute.launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (descriptor,),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "publication command completion",
            CudaStream::synchronize,
        )
    }

    fn publication_read<T: DeviceRepr + Copy>(
        &mut self,
        source: DeviceMemoryView<T>,
    ) -> Result<Vec<T>, SemanticTransitionError> {
        let bytes = source
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| publication_input_error("publication metadata read overflow"))?;
        let mut pinned = crate::device::PinnedHostBuffer::new(&self.stream, bytes)
            .map_err(|error| runtime_error("publication metadata staging", error))?;
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&source);
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |stream| {
            // SAFETY: recorder retains the complete source; pinned owner retains
            // its destination across enqueue failures and the final stream wait.
            unsafe {
                pinned.enqueue(stream.stream(), |destination| {
                    sys::cuMemcpyDtoHAsync_v2(
                        destination.cast(),
                        *source.device_ptr(),
                        bytes,
                        stream.stream().cu_stream(),
                    )
                    .result()
                })
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "publication metadata read completion",
            CudaStream::synchronize,
        )?;
        // SAFETY: the entire integral ABI slice completed its recorded copy.
        let result = unsafe { pinned.read_vec::<T>(source.len()) }
            .map_err(|error| runtime_error("publication metadata read", error))?;
        self.provider
            .record_final_observation_transfer(bytes as u64);
        Ok(result)
    }

    fn read_publication_header(
        &mut self,
        bank: usize,
    ) -> Result<PublicationHeader, SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        // SAFETY: PublicationBank starts with one initialized fixed Header and
        // the tracked allocation has sufficient alignment and checked ABI extent.
        let view = unsafe {
            storage.banks[bank]
                .view()
                .cast::<u8>()
                .expect("byte-aligned bank ABI")
                .slice(..size_of::<PublicationHeader>())
                .cast::<PublicationHeader>()
                .expect("aligned fixed header ABI")
        };
        Ok(self.publication_read(view)?[0])
    }

    pub fn acquire(&mut self) -> Result<SemanticPublishedLease, SemanticTransitionError> {
        if self.pending {
            let proposal = u32::try_from(self.next_proposal)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            // Complete and reconcile the actual CAS outcome before returning a
            // reader. A failed publication can never return the old base as progress.
            self.observe(proposal)?.into_published()?;
        }
        self.ensure_quiescent()?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let token = self
            .next_reader
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        self.next_reader = token;
        let device = allocate_publication(&self.provider, 1)?;
        self.steps.insert(token, StepContentStorage::new());
        self.readers.insert(
            token,
            PublishedReader {
                device,
                aliases: Arc::new(()),
                consumer_streams: BTreeSet::new(),
            },
        );
        // Lease storage is retained before upload or Acquire can enqueue.
        if let Err(error) = upload_publication(
            &self.provider,
            &[PublicationLease::default()],
            &self.readers[&token].device,
        ) {
            self.poisoned = true;
            return Err(error);
        }
        self.publication_command(2, Some(token))?;
        let lease = self.publication_read(self.readers[&token].device.view())?[0];
        if lease.status != 0 {
            self.readers.remove(&token);
            self.steps.remove(&token);
            return Err(SemanticTransitionError::PublicationRefused {
                status: lease.status,
            });
        }
        if lease.abi != 1
            || lease.active != 1
            || lease.instance != storage.instance
            || lease.bank > 1
            || lease.bank != (lease.word & 1)
            || lease.epoch != lease.word >> 1
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let header = self.read_publication_header(lease.bank as usize)?;
        let directory = self.publication_read(storage.directories[lease.bank as usize].view())?;
        // SAFETY: source follows Header in the checked fixed bank ABI; no private
        // task observation/result bytes are included in this metadata transfer.
        let source_view = unsafe {
            storage.banks[lease.bank as usize]
                .view()
                .cast::<u8>()
                .expect("byte-aligned bank ABI")
                .slice(
                    size_of::<PublicationHeader>()
                        ..size_of::<PublicationHeader>() + 32 * size_of::<SemanticTextSlot>(),
                )
                .cast::<SemanticTextSlot>()
                .expect("aligned fixed source ABI")
        };
        let source: [SemanticTextSlot; 32] = self
            .publication_read(source_view)?
            .try_into()
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        if header.abi != 1
            || header.instance != lease.instance
            || header.publication_word != lease.word
            || header.sealed_epoch != lease.epoch
            || header.range_count as usize != directory.len()
        {
            self.poisoned = true;
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let identity = SemanticPublishedIdentity {
            instance: lease.instance,
            word: lease.word,
            logical_digest: header.logical_digest,
            state_digest: header.state_digest,
        };
        self.steps
            .get_mut(&token)
            .expect("installed acquisition step")
            .identity = Some(identity);
        Ok(SemanticPublishedLease {
            issuer: Arc::clone(&self.publication_issuer),
            token,
            identity,
            header,
            source,
            directory,
            active: true,
        })
    }

    fn checked_reader(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<&PublishedReader, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if !lease.active || !Arc::ptr_eq(&lease.issuer, &self.publication_issuer) {
            return Err(publication_input_error(
                "lease is released or belongs to a different Session",
            ));
        }
        self.readers
            .get(&lease.token)
            .ok_or_else(|| publication_input_error("lease has no live native reader owner"))
    }

    fn checked_step(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<&StepContentStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        if !Arc::ptr_eq(&lease.issuer, &self.publication_issuer) {
            return Err(publication_input_error(
                "retained step belongs to a different Session",
            ));
        }
        let step = self
            .steps
            .get(&lease.token)
            .ok_or_else(|| publication_input_error("original step has been fully released"))?;
        if step.identity != Some(lease.identity)
            || lease.active != self.readers.contains_key(&lease.token)
        {
            return Err(publication_input_error(
                "retained step differs from its original acquired identity",
            ));
        }
        Ok(step)
    }

    /// Authenticate the original step even after its publication bank retires.
    /// This grants no publication access and refuses poisoned or released owners.
    pub fn retained_step_identity(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.checked_step(lease)?;
        Ok(lease.identity)
    }

    pub fn published_identity(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.checked_reader(lease)?;
        Ok(lease.identity)
    }

    /// Complete ordered role/index roster from this acquired directory, not a
    /// prediction from role counts or a read of a newer publication. This is
    /// cached structural metadata; it does not certify current tensor bytes.
    pub fn published_range_keys(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<Vec<(SemanticStateRole, u64)>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        lease
            .directory
            .iter()
            .map(|range| {
                SemanticStateRole::from_code(range.role)
                    .map(|role| (role, range.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect()
    }

    // These event edges are enqueue-only. The producer protocol targets the
    // legacy default stream; caller-declared hook writes target consumer_stream.
    // Every event/owner is retained before a may-enqueue boundary. No host wait
    // or digest/status transfer is performed on this successful path.
    fn order_content_inputs(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.readers
            .get_mut(&lease.token)
            .expect("validated publication reader")
            .consumer_streams
            .insert(consumer_stream);
        self.order_step_content_inputs(lease.token, consumer_stream)
    }

    fn order_step_content_inputs(
        &mut self,
        reader_token: u64,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let consumer = dlpack_consumer_stream(consumer_stream)?;
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| runtime_error("content context binding", e))?;
        self.steps
            .get_mut(&reader_token)
            .expect("validated content step")
            .consumer_streams
            .insert(consumer_stream);
        for source in [std::ptr::null_mut(), consumer] {
            let event = self
                .stream
                .context()
                .new_event(None)
                .map_err(|e| runtime_error("content producer event", e))?;
            self.release_events.push(event);
            let event = self.release_events.last().expect("retained content event");
            // SAFETY: the controlling application supplies a live same-context
            // CUDA stream, after all its writers. The native stream then waits
            // on this exact recorded event, not on future unrelated writes.
            unsafe { cudarc::driver::result::event::record(event.cu_event(), source) }
                .map_err(|e| runtime_error("content producer event record", e))?;
            self.stream
                .wait(event)
                .map_err(|e| runtime_error("content producer event join", e))?;
        }
        Ok(())
    }

    fn order_content_consumers(
        &mut self,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let event = self
            .stream
            .context()
            .new_event(None)
            .map_err(|e| runtime_error("content guard event", e))?;
        self.release_events.push(event);
        let event = self
            .release_events
            .last()
            .expect("retained content guard event");
        // SAFETY: record follows the guard on the admitted native stream. The
        // consumer must enqueue all dependent reads after this method returns.
        // CUDA events provide the dependency without a host synchronization.
        unsafe {
            cudarc::driver::result::event::record(event.cu_event(), self.stream.cu_stream())
                .map_err(|e| runtime_error("content guard event record", e))?;
            sys::cuStreamWaitEvent(
                dlpack_consumer_stream(consumer_stream)?,
                event.cu_event(),
                0,
            )
            .result()
            .map_err(|e| runtime_error("content guard consumer join", e))?;
        }
        Ok(())
    }

    /// Enqueue a live-content check against this reader's original range seals.
    /// Call after hooks/writers and before reads on the declared consumer stream.
    /// Integrity corruption invokes an unconditional device trap: the isolated
    /// task process must terminate, not reset or reuse its Session/tensors.
    /// A successful return means enqueued, not GPU-complete or accepted output.
    /// Concurrent external alias writes through the end of consumption are forbidden.
    pub fn guard_published_content(
        &mut self,
        lease: &SemanticPublishedLease,
        keys: &[(SemanticStateRole, u64)],
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        dlpack_consumer_stream(consumer_stream)?;
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
        let directory = &storage.directories[(lease.identity.word & 1) as usize];
        let mut distinct = BTreeSet::new();
        let mut ranges = Vec::with_capacity(keys.len());
        if keys.is_empty() {
            return Err(publication_input_error(
                "content guard requires actual publication ranges",
            ));
        }
        for &(role, index) in keys {
            let key = (role as u64, index);
            if !distinct.insert(key) {
                return Err(publication_input_error(
                    "content guard has duplicate publication ranges",
                ));
            }
            let (range_index, &range) = lease
                .directory
                .iter()
                .enumerate()
                .find(|(_, range)| (range.role, range.index) == key)
                .ok_or_else(|| {
                    publication_input_error("content guard names an absent publication range")
                })?;
            if range_index >= directory.len() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let allocation = storage
                .allocations
                .get(range.storage_slot as usize)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if range.generation != 1
                || range
                    .offset_bytes
                    .checked_add(range.length_bytes)
                    .is_none_or(|end| end > allocation.len() as u64)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let layout = if is_tensor_role(role as u64) {
                *storage
                    .layouts
                    .get(&key)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
            } else {
                SemanticTensorLayout::default()
            };
            // PrefixSource is sealed as raw records. Its synthetic U64 export
            // layout is intentionally not substituted into this hash domain.
            // The original seal remains on the device. Passing its host-cached
            // digest as a launch argument would itself be a digest transfer.
            ranges.push((role as u64, index, layout));
        }
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_publication_content_guard",
            )
            .ok_or_else(|| {
                runtime_error("kernel lookup", "publication content guard unavailable")
            })?;
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|e| runtime_error("content guard stream admission", e))?;
        let result = (|| {
            self.order_content_inputs(lease, consumer_stream)?;
            if let Some(inputs) = &inputs {
                // Existing guards must cover the actual exported private
                // inputs as well as the still-acquired publication bank.
                self.steps
                    .get_mut(&lease.token)
                    .expect("validated input step")
                    .consumer_streams
                    .insert(consumer_stream);
                inputs.verify(&self.domain, &mut self.poisoned)?;
            }
            for (role, index, layout) in ranges {
                let reader = self
                    .readers
                    .get(&lease.token)
                    .expect("validated native reader");
                storage.enqueue_content_guard(
                    &self.domain,
                    &mut self.poisoned,
                    reader.device.view(),
                    &execute,
                    role,
                    index,
                    layout,
                )?;
            }
            self.order_content_consumers(consumer_stream)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Capture private device digests after the trusted original computation,
    /// before public hooks. This establishes stability, never derivation or rights.
    /// The model owner must attach the issued witness to its original execution.
    /// Native feedback and fixed input aliases verify their original producer
    /// seals; their exact type, shape, interval and coordinate are preserved.
    /// Requires stream-ordered allocation support; a synchronizing allocator is
    /// never substituted for this enqueue-only path.
    pub fn capture_tensor_content(
        &mut self,
        lease: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<SemanticTensorContentWitness, SemanticTransitionError> {
        self.bind_tensor_content(lease, tensors, consumer_stream, false)
    }

    /// Bind the complete live model roster to this parent's originally sealed
    /// model content and generation. A new reader receives a new witness, not a
    /// new baseline. Initial physical layout/address may differ; later verify
    /// retains this invocation's exact producer storage and layout. Enqueue-only.
    pub fn bind_model_content(
        &mut self,
        lease: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<SemanticTensorContentWitness, SemanticTransitionError> {
        self.bind_tensor_content(lease, tensors, consumer_stream, true)
    }

    fn bind_tensor_content(
        &mut self,
        lease: &SemanticPublishedLease,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
        model: bool,
    ) -> Result<SemanticTensorContentWitness, SemanticTransitionError> {
        let empty = tensors.is_empty();
        let mut handoff = RetainedTensorAdmission {
            inputs: Some(tensors),
            stream: Arc::clone(self.provider.device().inner().stream()),
            ready: false,
        };
        self.checked_reader(lease)?;
        dlpack_consumer_stream(consumer_stream)?;
        if empty {
            return Err(publication_input_error(
                "content witness requires an actual tensor roster",
            ));
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|e| runtime_error("content binding stream admission", e))?;
        if !self
            .provider
            .device()
            .inner()
            .stream()
            .context()
            .has_async_alloc()
        {
            return Err(publication_input_error(
                "device content capture requires stream-ordered CUDA allocation support",
            ));
        }
        // Include even rejected handoffs in final retirement. Post-import
        // interval/allocation errors must not hide this caller's writer stream.
        self.steps
            .get_mut(&lease.token)
            .expect("validated content step")
            .consumer_streams
            .insert(consumer_stream);
        let tensors = prepare_semantic_tensors(
            &self.provider,
            handoff
                .inputs
                .take()
                .expect("retained original content inputs"),
        )?;
        let reader = self
            .steps
            .get_mut(&lease.token)
            .expect("validated content step");
        let index = reader.content.len();
        reader.content.push(TensorContentBuffers {
            tensors,
            seals: TensorContentSeals::Captured(Vec::new()),
            verification_inputs: Vec::new(),
        });
        let reader = &self.steps[&lease.token];
        let content = &reader.content[index];
        validate_content_coordinates(
            content
                .tensors
                .iter()
                .map(|tensor| (tensor.layout.role, tensor.layout.index)),
        )?;
        let mut native_origins = Vec::with_capacity(content.tensors.len());
        for tensor in &content.tensors {
            let mut original = None;
            if !model {
                // Shared prefix/key-value storage can belong to older steps
                // too. Match the actual current step before checking foreign
                // private owners, preserving its original logical interval.
                if let Some(inputs) = &reader.inputs {
                    if inputs.owns_tensor(tensor)? {
                        original = Some(Arc::clone(inputs));
                    }
                }
                if original.is_none() {
                    for (&token, owner) in &self.steps {
                        if token == lease.token {
                            continue;
                        }
                        if let Some(inputs) = &owner.inputs {
                            if inputs.owns_tensor(tensor)? {
                                return Err(publication_input_error(
                                    "native step content belongs to another acquired reader",
                                ));
                            }
                        }
                    }
                }
            }
            if original.is_none() {
                tensor_content_range(
                    &tensor.layout,
                    tensor.logical_begin,
                    tensor.logical_end,
                    tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                )?;
            }
            native_origins.push(original);
        }
        let model_plan = if model {
            let storage = self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let ranges = model_content_ranges(
                &lease.directory,
                &storage.layouts,
                content.tensors.iter().map(|tensor| {
                    (
                        tensor.layout,
                        tensor.logical_begin,
                        tensor.logical_end,
                        tensor.source.as_ref().map_or(0, DeviceMemoryView::len),
                    )
                }),
            )?;
            Some(model_content_copy_plan(
                &lease.directory,
                &ranges,
                &storage
                    .allocations
                    .iter()
                    .map(TrackedCudaSlice::len)
                    .collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        let seals = if let Some(plan) = &model_plan {
            TensorContentSeals::Model(ModelContentSeals::allocate(&self.provider, plan)?)
        } else {
            let mut digests = Vec::with_capacity(content.tensors.len());
            for (tensor, native) in content.tensors.iter().zip(native_origins) {
                if let Some(inputs) = native {
                    digests.push(CapturedTensorDigest::Publication(inputs));
                    continue;
                }
                let mut origin = None;
                for (&token, owner) in &self.steps {
                    if let Some(digest) = owner.feedback_origin_digest(tensor)? {
                        if token != lease.token {
                            return Err(publication_input_error(
                                "native feedback content belongs to another acquired reader",
                            ));
                        }
                        origin = Some(digest);
                        break;
                    }
                }
                digests.push(match origin {
                    Some(original) => original,
                    None => {
                        if tensor.native_allocation.is_some() {
                            return Err(publication_input_error("native allocation origin has no original content producer in this acquired reader"));
                        }
                        CapturedTensorDigest::Tensor {
                            cells: Arc::new(allocate_publication(&self.provider, 4)?), offset: 0, producer_sealed: false,
                        }
                    }
                });
            }
            TensorContentSeals::Captured(digests)
        };
        let handle = SemanticTensorContentWitness {
            issuer: Arc::clone(&self.publication_issuer),
            reader_token: lease.token,
            index,
            _witness: Arc::clone(&reader.content_witnesses),
        };
        self.steps
            .get_mut(&lease.token)
            .expect("validated content step")
            .content[index]
            .seals = seals;
        if let Some(plan) = model_plan {
            // Validate the original contract while the actual reader still pins
            // its directory. Then preserve those expected seals and contract
            // bytes, never a new digest of the supplied numerical model.
            self.guard_published_content(
                lease,
                &[(SemanticStateRole::ModelContract, 0)],
                consumer_stream,
            )?;
            let storage = self
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let TensorContentSeals::Model(model) = &self.steps[&lease.token].content[index].seals
            else {
                return Err(SemanticTransitionError::ObservationMismatch);
            };
            model.snapshot(
                &self.domain,
                &mut self.poisoned,
                &storage.directories[(lease.identity.word & 1) as usize],
                &storage.allocations[plan.contract_storage_slot],
                &plan,
            )?;
        }
        self.enqueue_tensor_content(lease, &handle, consumer_stream, model)?;
        Ok(handle)
    }

    /// Verify the actual consumer tensors against the original private witness.
    /// Metadata/pointer substitution is a typed input refusal; later byte mutation
    /// is a terminal device integrity error. Return is enqueue-only. Ordinary
    /// completion/observation must succeed before activating outputs or replay.
    pub fn verify_tensor_content(
        &mut self,
        lease: &SemanticPublishedLease,
        witness: &SemanticTensorContentWitness,
        tensors: Vec<SemanticTensorInput>,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let mut handoff = RetainedTensorAdmission {
            inputs: Some(tensors),
            stream: Arc::clone(self.provider.device().inner().stream()),
            ready: false,
        };
        self.checked_content_witness(lease, witness)?;
        dlpack_consumer_stream(consumer_stream)?;
        self.steps
            .get_mut(&lease.token)
            .expect("validated witness step")
            .consumer_streams
            .insert(consumer_stream);
        let tensors = prepare_semantic_tensors(
            &self.provider,
            handoff
                .inputs
                .take()
                .expect("retained original verification inputs"),
        )?;
        let content = &mut self
            .steps
            .get_mut(&lease.token)
            .expect("validated witness step")
            .content[witness.index];
        let offset = content.verification_inputs.len();
        content.verification_inputs.extend(tensors);
        let actual = &content.verification_inputs[offset..];
        if actual.len() != content.tensors.len()
            || content
                .tensors
                .iter()
                .zip(actual)
                .any(|(expected, actual)| !same_tensor_content_owner(expected, actual))
        {
            return Err(publication_input_error("content verification must use the exact original tensor storage, layout and interval"));
        }
        self.enqueue_tensor_content(lease, witness, consumer_stream, true)
    }

    fn checked_content_witness(
        &self,
        lease: &SemanticPublishedLease,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        let reader = self.checked_step(lease)?;
        if !Arc::ptr_eq(&self.publication_issuer, &witness.issuer)
            || witness.reader_token != lease.token
            || !Arc::ptr_eq(&reader.content_witnesses, &witness._witness)
            || witness.index >= reader.content.len()
        {
            return Err(publication_input_error(
                "content witness belongs to another parent or Session",
            ));
        }
        Ok(())
    }

    fn enqueue_tensor_content(
        &mut self,
        lease: &SemanticPublishedLease,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_content_witness(lease, witness)?;
        lease.model_context_header()?;
        // Model witnesses retain the record authenticated by their own reader,
        // including after its release. A later publication must not replace that
        // baseline, nor force it to match the initial admission generation.
        self.enqueue_content_witness(witness, consumer_stream, verify)
    }

    // The handle is issued only by the original reader. A retained policy may
    // check it after publication without manufacturing a second acquired lease.
    fn guard_continuation_content(
        &mut self,
        binding: &TextBindingStorage,
    ) -> Result<(), SemanticTransitionError> {
        let witness = &binding._witness;
        let reader = self.steps.get(&witness.reader_token).ok_or_else(|| {
            publication_input_error("original continuation step has been released")
        })?;
        let original_parent = match &binding.parent {
            TextBindingParent::Published { .. } => reader.identity.is_some(),
            TextBindingParent::Prepared(inputs) => {
                reader
                    .inputs
                    .as_ref()
                    .is_some_and(|original| Arc::ptr_eq(original, inputs))
                    && reader
                        .prepared
                        .as_ref()
                        .is_some_and(|prepared| prepared.observed)
                    && self
                        .prepared_segment
                        .as_ref()
                        .is_some_and(|build| build.completed)
            }
        };
        if !Arc::ptr_eq(&self.publication_issuer, &witness.issuer)
            || !original_parent
            || !Arc::ptr_eq(&reader.content_witnesses, &witness._witness)
            || reader
                .content
                .get(witness.index)
                .is_none_or(|content| !matches!(content.seals, TensorContentSeals::Captured(_)))
        {
            return Err(publication_input_error(
                "continuation no longer owns its original captured content",
            ));
        }
        self.enqueue_content_witness(witness, binding.consumer_stream, true)
    }

    fn enqueue_content_witness(
        &mut self,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        let execute = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_tensor_content_witness",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "tensor content witness unavailable"))?;
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|e| runtime_error("tensor content stream admission", e))?;
        let result = (|| {
            self.order_step_content_inputs(witness.reader_token, consumer_stream)?;
            let content = &self.steps[&witness.reader_token].content[witness.index];
            content.enqueue(&self.domain, &mut self.poisoned, &execute, verify)?;
            self.order_content_consumers(consumer_stream)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn ensure_step_inputs(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<Arc<PreparedStepInputs>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        self.checked_step(lease)?;
        lease.model_context_header()?;
        if let Some(inputs) = &self.steps[&lease.token].inputs {
            return Ok(Arc::clone(inputs));
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("step input preparation stream admission", error))?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let inputs = Arc::new(PreparedStepInputs::allocate(
            &self.provider,
            storage,
            self.readers[&lease.token].device.view(),
        )?);
        // Retain every output before the first upload or producer enqueue. A
        // partially initialized step remains owned by the poisoned Session.
        self.steps
            .get_mut(&lease.token)
            .expect("validated input step")
            .inputs = Some(Arc::clone(&inputs));
        let result = inputs
            .initialize(&self.provider)
            .and_then(|()| inputs.enqueue(&self.domain, &mut self.poisoned));
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        Ok(inputs)
    }

    /// Export private step-owned physical source slots without host copying or rotation.
    pub fn published_source(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.export_publication_bank_field(lease, PublicationBankField::Source, consumer_stream)
    }

    /// Export the original acquired terminal flag as a read-only U64[1] view
    /// into the same private step inputs used by source and model context.
    /// The ordinary cold input preparation and retained consumer lifetime apply;
    /// no terminal value is inferred from source tokens or model metadata.
    pub fn published_terminal(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.export_publication_bank_field(lease, PublicationBankField::Terminal, consumer_stream)
    }

    /// Join retained acquisition metadata with fixed private prefix identity,
    /// extent, ring head and source views. Device acquisition supplies every
    /// value once; the step retains these inputs through the final consumer.
    pub fn published_model_context(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<SemanticModelContext, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        self.checked_reader(lease)?;
        let header = lease.model_context_header()?;
        let contract = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value;
        let prefix_range = lease
            .directory
            .iter()
            .find(|range| {
                range.role == SemanticStateRole::PrefixIdentity as u64 && range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if prefix_range.length_bytes != 32 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let prefix_identity =
            self.published_record(lease, SemanticStateRole::PrefixIdentity, 0, consumer_stream)?;
        let prefix_extent = self.export_publication_bank_field(
            lease,
            PublicationBankField::PrefixExtent,
            consumer_stream,
        )?;
        let ring_head = self.export_publication_bank_field(
            lease,
            PublicationBankField::RingHead,
            consumer_stream,
        )?;
        let source = self.published_source(lease, consumer_stream)?;
        Ok(SemanticModelContext {
            parent: lease.identity,
            descriptor_digest: header.descriptor_digest,
            numerical_state: (header.model_geometry_digest, header.model_numerical_digest),
            semantic_root: (
                header.semantic_owner,
                header.semantic_slot,
                header.semantic_generation,
                header.semantic_digest,
                header.semantic_extents,
            ),
            prefix: (prefix_identity, prefix_extent),
            ring_head,
            source,
            generations: (
                header.model_generation,
                header.neural_bank,
                header.neural_generation,
                header.cache_generation,
                header.authority_generation,
            ),
            topology_identity: contract.topology_identity,
            table_identity: contract.table_identity,
            capacities: (
                contract.prefix_capacity,
                contract.feedback_capacity,
                contract.max_position,
            ),
        })
    }

    /// Export the selected resident neural bank without the private step copy
    /// used by per-tensor late-backward exports. Full backing allocations occur
    /// once and retain the acquired reader, while the returned geometry names
    /// every typed view in roles 18..=25. Callers must keep these aliases
    /// read-only and release the parent only after every consumer stream joins.
    pub fn published_resident_model_memory(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<SemanticResidentModelMemory, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        self.checked_reader(lease)?;
        let header = lease.model_context_header()?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        storage.model_memory.validate(&storage.layouts)?;
        if storage.model_slots.len() != storage.model_memory.allocation_bytes.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let keys = lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 18..=25))
            .map(|range| {
                SemanticStateRole::from_code(range.role)
                    .map(|role| (role, range.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if keys.len() != storage.model_memory.views.len() {
            return Err(publication_input_error(
                "resident model memory requires the complete selected model roster",
            ));
        }
        self.guard_published_content(lease, &keys, consumer_stream)?;

        let bank = (lease.identity.word & 1) as usize;
        let mut allocations = Vec::with_capacity(storage.model_slots.len());
        for (&bytes, slots) in storage
            .model_memory
            .allocation_bytes
            .iter()
            .zip(&storage.model_slots)
        {
            let allocation = storage
                .allocations
                .get(slots[bank])
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if allocation.len() as u64 != bytes {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let view = allocation.view();
            let provenance = view.allocation_provenance().ok_or_else(|| {
                publication_input_error("resident model backing has no native allocation owner")
            })?;
            let extent =
                i64::try_from(bytes).map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            let tensor = self.export_reader_view(
                lease,
                view,
                vec![extent],
                vec![1],
                (1, 8),
                consumer_stream,
            )?;
            allocations.push((provenance, tensor));
        }
        let views = storage
            .model_memory
            .views
            .iter()
            .cloned()
            .map(|view| {
                let layout = storage
                    .layouts
                    .get(&(view.role, view.index))
                    .copied()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                Ok((view, layout))
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        Ok(SemanticResidentModelMemory {
            parent: lease.identity,
            generations: (
                header.model_generation,
                header.neural_bank,
                header.neural_generation,
            ),
            numerical_state: (header.model_geometry_digest, header.model_numerical_digest),
            allocations,
            storages: storage.model_memory.storages.clone(),
            views,
        })
    }

    /// Export this acquired parent's prefix storage as U64[capacity, 8]. Columns
    /// follow SemanticTextSlot exactly, including source/generated provenance and
    /// its native ledger index. Only [0, prefix_extent) is committed, including
    /// the legal empty prefix; unused capacity must not be consumed as content.
    /// The export neither acquires again nor copies rows to host.
    /// The managed owner retains the same reader and append-only prefix storage.
    /// Consumers must not modify this sealed view and must join their final use
    /// through release, just as for published_tensor.
    pub fn published_prefix(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<(SemanticPublishedIdentity, DlpackManagedTensor), SemanticTransitionError> {
        self.checked_reader(lease)?;
        let range = *lease
            .directory
            .iter()
            .find(|range| range.role == SemanticStateRole::PrefixSource as u64 && range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.logical_end != lease.header.prefix_extent {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let capacity = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .contract_value
            .prefix_capacity;
        let layout = committed_prefix_layout(&range, capacity)?;
        let tensor = self.export_publication_range(lease, range, layout, consumer_stream)?;
        Ok((lease.identity, tensor))
    }

    /// Read RNG coordinates from the actual parent of the pending continuation.
    /// The application cannot supply a replacement stream or proposal identity.
    pub fn continuation_rng(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticRngBinding, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if self.continuation_base != Some(lease.identity.word)
            || self
                .admitted_transition
                .is_none_or(|(token, word, _)| token != lease.token || word != lease.identity.word)
        {
            return Err(publication_input_error(
                "execution requires this exact acquired reader's admitted continuation",
            ));
        }
        lease.header.rng_binding()
    }

    /// A FINAL acquired bank must contain its actual native terminal intent.
    /// This readiness check does not deliver an effect or issue use authority.
    pub fn admit_external_activation(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        if lease.header.terminal != 2 {
            return Err(publication_input_error(
                "external activation requires a genuinely published FINAL parent",
            ));
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let range = lease
            .directory
            .iter()
            .find(|range| range.role == 30)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let payload = lease
            .directory
            .iter()
            .find(|range| range.role == 31)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let owner = storage
            .allocations
            .get(range.storage_slot as usize)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.offset_bytes != 0
            || range.length_bytes < size_of::<IntentQueueHeader>() as u64
            || range.length_bytes > owner.len() as u64
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        // SAFETY: checked header range is integral, aligned and initialized by
        // the acquired sealed directory's actual native queue owner.
        let header_view = unsafe {
            owner
                .view()
                .slice(..size_of::<IntentQueueHeader>())
                .cast::<IntentQueueHeader>()
        }
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let header = self.publication_read(header_view)?[0];
        if header.abi != 1
            || header.count == 0
            || header.count > header.capacity
            || header
                .count
                .checked_mul(size_of::<IntentEntry>() as u64)
                .and_then(|n| n.checked_add(size_of::<IntentQueueHeader>() as u64))
                != Some(range.length_bytes)
            || header.payload_used_bytes != payload.length_bytes
            || header.payload_used_bytes > header.payload_capacity_bytes
            || header.effect_length_bytes == 0
            || header
                .effect_offset_bytes
                .checked_add(header.effect_length_bytes)
                .is_none_or(|n| n > header.payload_used_bytes)
        {
            return Err(publication_input_error(
                "FINAL bank has no complete native terminal intent queue",
            ));
        }
        let begin =
            size_of::<IntentQueueHeader>() + (header.count as usize - 1) * size_of::<IntentEntry>();
        // SAFETY: exact final entry extent follows from the checked queue count.
        let entry_view = unsafe {
            owner
                .view()
                .slice(begin..begin + size_of::<IntentEntry>())
                .cast::<IntentEntry>()
        }
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let entry = self.publication_read(entry_view)?[0];
        if entry.audit_instance != lease.identity.instance
            || entry.audit_epoch != lease.identity.word >> 1
            || entry.result_logical != lease.identity.logical_digest
            || entry.chain != header.chain_head
            || entry.payload_len < 8
            || entry
                .payload_offset
                .checked_add(entry.payload_len)
                .is_none_or(|n| n > header.payload_used_bytes)
            || entry.stable_identity == Identity256::default()
            || entry.effect_digest == Identity256::default()
        {
            return Err(publication_input_error(
                "terminal intent is not the actual final effect of this acquired publication",
            ));
        }
        Ok(())
    }

    /// Export the cold physical-capacity tensor shape, retaining its original
    /// sealed logical range. Unused rows are not content. Shared prefix caches
    /// retain the reader; private input caches retain their original step.
    pub fn published_tensor(
        &mut self,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        let (range, layout) = self.published_tensor_layout(lease, role, index)?;
        if matches!(role as u64, 18..=25 | 51..=54) {
            self.guard_published_content(lease, &[(role, index)], consumer_stream)?;
        }
        self.export_publication_range(lease, range, layout, consumer_stream)
    }

    /// Select the exact published training view from the cold resident arena.
    /// The host supplies neither an ordinal nor an identity: both the cursor and
    /// RNG coordinates come from this acquired publication, while CUDA verifies
    /// the complete role-29 bytes before exposing fixed output ports.
    pub fn select_training_view(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticSelectedTrainingView, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let arena = Arc::clone(
            self.training_views
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let range = *lease
            .directory
            .iter()
            .find(|range| range.role == SemanticStateRole::TrainingView as u64 && range.index == 0)
            .ok_or_else(|| publication_input_error("acquired training-view range is absent"))?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let allocation = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let begin = usize::try_from(range.offset_bytes)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let length = usize::try_from(range.length_bytes)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let end = begin
            .checked_add(length)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let selected = allocation
            .view()
            .try_slice(begin..end)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let result = arena.enqueue_selection(
            selected,
            lease.header.training_cursor,
            lease.header.training_rng,
            self.training_origins.clone(),
        );
        if matches!(
            &result,
            Err(SemanticTransitionError::Runtime {
                operation: "training-view device selection" | "training-view launch commit",
                ..
            })
        ) {
            self.poisoned = true;
        }
        result
    }

    /// Allocation origin and complete backing bytes of the exact view exported
    /// by `published_tensor`, retaining that view's reader/step access grant.
    /// Exported capacity is not evidence that unused contents are model inputs.
    pub fn published_tensor_allocation(
        &mut self,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<(DeviceAllocationProvenance, DlpackManagedTensor), SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let (range, layout) = self.published_tensor_layout(lease, role, index)?;
        if matches!(role as u64, 18..=25 | 51..=54) {
            self.guard_published_content(lease, &[(role, index)], consumer_stream)?;
        }
        let view = self.publication_tensor_view(lease, range, layout)?;
        let provenance = view.allocation_provenance().ok_or_else(|| {
            publication_input_error("published tensor has no complete native allocation owner")
        })?;
        let allocation = view
            .allocation_view()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bytes = i64::try_from(allocation.len())
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let tensor = self.export_publication_tensor_view(
            lease,
            range.role,
            allocation,
            vec![bytes],
            vec![1],
            (1, 8),
            consumer_stream,
        )?;
        Ok((provenance, tensor))
    }

    fn published_tensor_layout(
        &self,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
        index: u64,
    ) -> Result<(PublicationRange, SemanticTensorLayout), SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let layout = *storage
            .layouts
            .get(&(role as u64, index))
            .ok_or_else(|| publication_input_error("acquired role has no model tensor layout"))?;
        let range = *lease
            .directory
            .iter()
            .find(|range| range.role == role as u64 && range.index == index)
            .ok_or_else(|| publication_input_error("acquired tensor range is absent"))?;
        Ok((range, layout))
    }

    /// Read the map from this acquired bank's sealed record, not a later pending
    /// invocation. Reading these small identity records never reads tensor values.
    pub fn published_active_rows(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<SemanticActiveRows, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let bytes = self.read_published_record_bytes(lease, SemanticStateRole::TensorLayout)?;
        let (_, layouts, active) = decode_tensor_table(&bytes)?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if layouts != storage.layouts.values().copied().collect::<Vec<_>>()
            || active.rows.len() as u64 > 32 + storage.contract_value.feedback_capacity
            || lease
                .directory
                .iter()
                .filter(|range| matches!(range.role, 51..=54))
                .any(|range| {
                    range.logical_begin != 0
                        || range.logical_end != active.rows.len() as u64
                        || (active.rows.is_empty() != (range.length_bytes == 0))
                })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(active)
    }

    /// Cold read of the original model contribution from this acquired parent.
    /// The application freezes external writers; registered consumer prefixes
    /// are joined before reading and verifying the original sealed bytes.
    /// This is never a host query inside a resident reasoning loop.
    pub fn published_model_numerical_mode(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let streams = self
            .checked_reader(lease)?
            .consumer_streams
            .iter()
            .chain(&self.checked_step(lease)?.consumer_streams)
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        self.complete_step_consumers(lease, &streams)?;
        let record = self.read_published_material_range(lease, 43, 0)?;
        if record.original_record_digest() != record.range.digest {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let table =
            self.read_published_material_range(lease, SemanticStateRole::TensorLayout as u64, 0)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let contract = self.publication_read(storage.contract.view())?[0];
        if publication_abi_bytes(&[contract]) != publication_abi_bytes(&[storage.contract_value]) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let counts = self.publication_read(storage.role_counts.view())?;
        let mut role_counts = [0; 55];
        for (index, count) in counts.iter().enumerate() {
            if count.role != index as u64 + 1 {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            role_counts[index] = count.count;
        }
        let terminals = self.publication_read(storage.terminals.view())?;
        let capacities = lease
            .directory
            .iter()
            .map(|range| {
                let allocation = usize::try_from(range.storage_slot)
                    .ok()
                    .and_then(|slot| storage.allocations.get(slot))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let capacity = usize::try_from(range.offset_bytes)
                    .ok()
                    .and_then(|offset| allocation.len().checked_sub(offset))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                Ok((range.role, range.index, capacity))
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        Ok(verified_runtime_model_mode(
            &record.bytes,
            contract,
            &role_counts,
            &terminals,
            &storage.layouts,
            &capacities,
            &table,
            &storage.model_memory,
        )?
        .to_vec())
    }

    /// Cold export of the complete native material reachable from this acquired
    /// parent. The bytes carry content, not a use grant or publication authority.
    /// The canonical execution carrier must bind them to its original invocation.
    pub fn published_state_material(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_reader(lease)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let bank = self.read_published_bank(lease)?;
        let graph = self
            .graph
            .export_transition_root_parts(
                bank.header.semantic_owner,
                bank.header.semantic_slot,
                bank.header.semantic_generation,
            )
            .map_err(SemanticTransitionError::Semantic)?;
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
        let mut ranges = Vec::with_capacity(lease.directory.len());
        let mut model_allocations = Vec::with_capacity(storage.model_slots.len());
        for slots in &storage.model_slots {
            let allocation = &storage.allocations[slots[(lease.identity.word & 1) as usize]];
            model_allocations.push(if allocation.is_empty() {
                Vec::new()
            } else {
                self.publication_read(allocation.view())?
            });
        }
        for range in &lease.directory {
            if matches!(range.role, 18..=25) {
                let (allocation, offset) =
                    storage.model_memory.location(range.role, range.index)?;
                let end = usize::try_from(range.length_bytes)
                    .ok()
                    .and_then(|bytes| offset.checked_add(bytes))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let bytes = model_allocations[allocation]
                    .get(offset..end)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .to_vec();
                ranges.push(PublicationMaterialRange {
                    range: *range,
                    capacity: model_allocations[allocation].len() - offset,
                    bytes,
                });
            } else {
                ranges.push(self.read_published_material_range(lease, range.role, range.index)?);
            }
        }
        let mut contract = storage.contract_value;
        // Device addresses and owner handles are relocated from fresh actual
        // allocations. They can never be imported as executable pointers.
        contract.terminal_tokens = 0;
        contract.role_counts = 0;
        contract.semantic_owner = 0;
        PublicationMaterial {
            bank,
            contract,
            role_counts,
            terminals,
            layouts: storage.layouts.clone(),
            model_memory: storage.model_memory.clone(),
            model_allocations,
            ranges,
            graph,
        }
        .encode()
    }

    /// Cold export of native evidence from two genuinely held adjacent
    /// publications. Only the successor identity and its six publication-only
    /// records enter the evidence; the successor's full state is not exported.
    /// The application must freeze further writers and register all consumer
    /// streams before this call. Both later release rosters must include one.
    pub fn published_replay_evidence(
        &mut self,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        self.guard_replay_publications(predecessor, successor)?;
        let predecessor =
            PublicationMaterial::decode(&self.published_state_material(predecessor)?)?;
        self.read_replay_evidence(&predecessor, successor)?.encode()
    }

    /// Cold export of the original authority envelope and actual successor
    /// continuation decision from held adjacent publications. The execution
    /// carrier stores this as its reconstruction provenance root, not evidence
    /// of current rights and not a second copy of the complete predecessor.
    /// Further writers must be frozen and every consumer stream registered;
    /// both later release rosters must include the guard's stream one.
    pub fn published_replay_provenance(
        &mut self,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        self.guard_replay_publications(predecessor, successor)?;
        let identity = predecessor.identity;
        let predecessor =
            PublicationMaterial::decode(&self.published_state_material(predecessor)?)?;
        self.read_replay_evidence(&predecessor, successor)?;
        let authority = predecessor
            .ranges
            .iter()
            .find(|item| item.range.role == 38 && item.range.index == 0)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .clone();
        let decision = self.read_published_material_range(successor, 39, 0)?;
        let provenance = PublicationReplayProvenance {
            predecessor: identity,
            authority,
            decision,
        };
        provenance.validate(&predecessor)?;
        provenance.encode()
    }

    /// Restore already-decoded complete predecessor material through the sole
    /// native cold restoration path. Current use authority remains external.
    pub fn restore_replay_material(
        &mut self,
        material: &SemanticReplayMaterial,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.restore_state_material(&material.predecessor.encode()?)
    }

    /// Compare a genuine rerun with the expected complete logical/state hashes
    /// and normalized publication records. Runtime instance/word coordinates
    /// differ after restoration and are checked as new lineage, not byte-equal.
    /// This cold check neither issues a use grant nor proves historical origin.
    /// It guards every actual range of both held readers before cold readback;
    /// cached seals alone cannot detect writes through shared tensor aliases.
    /// The method owns consumer stream one for both readers. Their later release
    /// rosters must include one as well as every application consumer stream.
    pub fn verify_replay_publication(
        &mut self,
        material: &SemanticReplayMaterial,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.guard_replay_publications(predecessor, successor)?;
        // The following ordinary cold reads wait for both guards on the native
        // stream and propagate a fatal integrity error before accepting bytes.
        let actual_predecessor =
            PublicationMaterial::decode(&self.published_state_material(predecessor)?)?;
        let header = actual_predecessor.bank.header;
        let expected = material.predecessor_identity();
        if header.instance == expected.instance
            || header.recovered_instance != expected.instance
            || header.publication_word != 0
            || header.logical_digest != expected.logical_digest
            || header.state_digest != expected.state_digest
        {
            return Err(publication_input_error(
                "replay comparison requires the freshly restored complete predecessor",
            ));
        }
        let actual = self.read_replay_evidence(&actual_predecessor, successor)?;
        if actual.successor.logical_digest != material.evidence.successor.logical_digest
            || actual.successor.state_digest != material.evidence.successor.state_digest
        {
            return Err(publication_input_error(
                "genuine replay successor differs from the expected complete native state",
            ));
        }
        for (actual, expected) in actual.ranges.iter().zip(&material.evidence.ranges) {
            if actual.logical_record_digest()? != expected.logical_record_digest()? {
                return Err(publication_input_error(
                    "genuine replay differs from the expected normalized publication records",
                ));
            }
        }
        Ok(())
    }

    // Fence all registered writer/consumer prefixes before hashing either bank.
    // This does not retire the original graph's aliases or witnesses and cannot
    // fence future writes; the controlling application freezes those separately.
    fn guard_replay_publications(
        &mut self,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        for lease in [predecessor, successor] {
            let streams = self
                .checked_reader(lease)?
                .consumer_streams
                .iter()
                .chain(&self.checked_step(lease)?.consumer_streams)
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            self.complete_step_consumers(lease, &streams)?;
        }
        let predecessor_ranges = self.published_range_keys(predecessor)?;
        let successor_ranges = self.published_range_keys(successor)?;
        self.guard_published_content(predecessor, &predecessor_ranges, 1)?;
        self.guard_published_content(successor, &successor_ranges, 1)
    }

    fn read_replay_evidence(
        &mut self,
        predecessor: &PublicationMaterial,
        successor: &SemanticPublishedLease,
    ) -> Result<PublicationReplayEvidence, SemanticTransitionError> {
        let bank = self.read_published_bank(successor)?;
        let mut ranges = Vec::with_capacity(REPLAY_EVIDENCE_ROLES.len());
        for role in REPLAY_EVIDENCE_ROLES {
            ranges.push(self.read_published_material_range(successor, role, 0)?);
        }
        let evidence = PublicationReplayEvidence {
            successor: successor.identity,
            ranges,
        };
        let coverage = self.read_published_material_range(successor, 14, 0)?;
        let codebook = self.read_published_material_range(successor, 48, 0)?;
        evidence.validate_observed(predecessor, &bank, &coverage, &codebook)?;
        Ok(evidence)
    }

    fn read_published_bank(
        &mut self,
        lease: &SemanticPublishedLease,
    ) -> Result<PublicationBank, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let bank =
            self.publication_read(storage.banks[(lease.identity.word & 1) as usize].view())?[0];
        if publication_abi_bytes(&[bank.header]) != publication_abi_bytes(&[lease.header])
            || publication_abi_bytes(&bank.source) != publication_abi_bytes(&lease.source)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(bank)
    }

    fn read_published_material_range(
        &mut self,
        lease: &SemanticPublishedLease,
        role: u64,
        index: u64,
    ) -> Result<PublicationMaterialRange, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let range = lease
            .directory
            .iter()
            .find(|range| (range.role, range.index) == (role, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let owner = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let begin = usize::try_from(range.offset_bytes)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let end = usize::try_from(range.length_bytes)
            .ok()
            .and_then(|n| begin.checked_add(n))
            .filter(|&end| end <= owner.len())
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.generation != 1 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let bytes = if begin == end {
            Vec::new()
        } else {
            self.publication_read(owner.view().slice(begin..end))?
        };
        Ok(PublicationMaterialRange {
            range: *range,
            capacity: owner.len() - begin,
            bytes,
        })
    }

    fn read_published_record_bytes(
        &mut self,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        Ok(self
            .read_published_material_range(lease, role as u64, 0)?
            .bytes)
    }

    /// Export the actual bytes of one acquired non-tensor owner, including native
    /// raw feedback. This performs no host tensor-value read or numeric cast.
    pub fn published_record(
        &mut self,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if is_tensor_role(role as u64) {
            return Err(publication_input_error(
                "tensor roles require their exact model layout export",
            ));
        }
        let range = *lease
            .directory
            .iter()
            .find(|range| range.role == role as u64 && range.index == index)
            .ok_or_else(|| publication_input_error("acquired record range is absent"))?;
        self.export_publication_range(
            lease,
            range,
            publication_record_layout(&range),
            consumer_stream,
        )
    }

    /// Encode the exact acquired raw slots and current lineage on the device,
    /// without packing or status/validity readback. The projection kernel traps
    /// on invalid provenance before the consumer event can become ready.
    /// All six outputs receive their original private content seals before
    /// export. Later tensor-content capture verifies those seals, not a new
    /// baseline made from potentially modified consumer aliases.
    pub fn published_feedback(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_stream: u64,
    ) -> Result<SemanticFeedback, SemanticTransitionError> {
        self.checked_reader(lease)?;
        dlpack_consumer_stream(consumer_stream)?;
        if self
            .admitted_transition
            .is_none_or(|(token, word, _)| token != lease.token || word != lease.identity.word)
        {
            return Err(publication_input_error(
                "feedback encoding requires this acquired parent's admitted segment",
            ));
        }
        // Keep the complete ordinary call outside capture, including the gaps
        // between allocation, producer launches, original seals and exports.
        // The private prepared enqueue remains usable by the segment builder.
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("feedback stream admission", error))?;
        let schema = SemanticFeedbackSchema::new(self.feedback_statement_bytes()?)?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let descriptor = self.descriptor();
        let buffers = FeedbackBuffers::allocate(
            &self.provider,
            schema,
            storage.contract_value.feedback_capacity,
            descriptor.arena[4],
        )?;
        let schema = buffers.schema.clone();
        let slots = buffers.slots;
        let support_capacity = buffers.support_capacity;
        let offset_cells = buffers.support_offsets.len();
        // Install the permanent step owner before any kernel or export can
        // enqueue. A later error or unwind cannot replace/drop unfinished
        // outputs through a subsequent feedback call.
        let reader = self
            .steps
            .get_mut(&lease.token)
            .expect("validated feedback step");
        let output_index = reader.feedback.len();
        reader.feedback.push(buffers);
        let result = (|| {
            let reader = self
                .readers
                .get(&lease.token)
                .expect("validated native reader");
            let buffers = &self.steps[&lease.token].feedback[output_index];
            buffers.enqueue(
                &self.domain,
                &mut self.poisoned,
                &self.graph,
                &storage,
                &self
                    .task
                    .as_ref()
                    .ok_or(SemanticTransitionError::InvalidTaskBank)?
                    .1,
                &reader.device,
                descriptor,
            )?;
            // SAFETY: f32 has a padding-free byte representation. The returned
            // view retains the complete original tracked feature allocation.
            let features = unsafe { buffers.features.view().cast::<u8>() }
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let device_validity = buffers.validity.view();
            let queries = buffers.query_receipts.view();
            let originals = buffers.original_statements.view();
            let offsets = buffers.support_offsets.view();
            let support_rows = buffers.supports.view();
            let features = self.export_step_view(
                lease,
                features,
                vec![slots as i64, schema.feature_width as i64],
                vec![schema.feature_width as i64, 1],
                (2, 32),
                consumer_stream,
            )?;
            let device_validity = self.export_step_view(
                lease,
                device_validity,
                vec![slots as i64],
                vec![1],
                (6, 8),
                consumer_stream,
            )?;
            let mut export_words =
                |view: DeviceMemoryView<u64>, shape: Vec<i64>, strides: Vec<i64>| {
                    // SAFETY: u64 has no padding; the complete tracked owner stays
                    // attached to each byte view and acquired-reader export.
                    let bytes = unsafe { view.cast::<u8>() }
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    self.export_step_view(lease, bytes, shape, strides, (1, 64), consumer_stream)
                };
            let query_receipts = export_words(queries, vec![slots as i64, 42], vec![42, 1])?;
            let original_statements = export_words(originals, vec![slots as i64], vec![1])?;
            let support_offsets = export_words(offsets, vec![offset_cells as i64], vec![1])?;
            let supports =
                export_words(support_rows, vec![support_capacity as i64, 12], vec![12, 1])?;
            Ok(SemanticFeedback {
                parent: lease.identity,
                schema,
                features,
                device_validity,
                query_receipts,
                original_statements,
                support_offsets,
                supports,
            })
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn publication_tensor_view(
        &mut self,
        lease: &SemanticPublishedLease,
        range: PublicationRange,
        layout: SemanticTensorLayout,
    ) -> Result<DeviceMemoryView<u8>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let view = if matches!(range.role, 1 | 3..=13 | 18..=25) {
            let inputs = self.ensure_step_inputs(lease)?;
            let view = inputs
                .views
                .get((lease.header.publication_word & 1) as usize)
                .expect("two publication input banks")
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let local = PublicationRange {
                offset_bytes: 0,
                ..range
            };
            view.slice(publication_export_span(&local, &layout, view.len())?)
        } else {
            let storage = Arc::clone(
                self.publication
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            );
            let allocation = storage
                .allocations
                .get(range.storage_slot as usize)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            allocation
                .view()
                .slice(publication_export_span(&range, &layout, allocation.len())?)
        };
        Ok(view)
    }

    fn export_publication_range(
        &mut self,
        lease: &SemanticPublishedLease,
        range: PublicationRange,
        layout: SemanticTensorLayout,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let view = self.publication_tensor_view(lease, range, layout)?;
        let shape = layout.dimensions[..layout.rank as usize]
            .iter()
            .map(|&n| {
                i64::try_from(n)
                    .map_err(|_| publication_input_error("DLPack shape exceeds signed extent"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let strides = layout.strides_bytes[..layout.rank as usize]
            .iter()
            .map(|&n| {
                i64::try_from(n / layout.element_bytes)
                    .map_err(|_| publication_input_error("DLPack stride exceeds signed extent"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = match layout.scalar_type {
            1 => (1, 8),
            2 => (1, 32),
            3 => (1, 64),
            4 => (2, 16),
            5 => (4, 16),
            6 => (2, 32),
            7 => (0, 64),
            8 => (6, 8),
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        self.export_publication_tensor_view(
            lease,
            range.role,
            view,
            shape,
            strides,
            dtype,
            consumer_stream,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "publication export keeps authenticated role and DLPack geometry explicit"
    )]
    fn export_publication_tensor_view(
        &mut self,
        lease: &SemanticPublishedLease,
        role: u64,
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        if matches!(role, 3 | 6..=13 | 18..=25) {
            self.export_step_view(lease, view, shape, strides, dtype, consumer_stream)
        } else {
            self.export_reader_view(lease, view, shape, strides, dtype, consumer_stream)
        }
    }

    fn export_publication_bank_field(
        &mut self,
        lease: &SemanticPublishedLease,
        field: PublicationBankField,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let inputs = self.ensure_step_inputs(lease)?;
        let (_, shape, strides) = field.layout();
        self.export_step_view(
            lease,
            inputs.field_view(field)?,
            shape,
            strides,
            (1, 64),
            consumer_stream,
        )
    }

    fn export_reader_view(
        &mut self,
        lease: &SemanticPublishedLease,
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let guard = Arc::clone(&self.checked_reader(lease)?.aliases);
        self.readers
            .get_mut(&lease.token)
            .expect("validated native reader")
            .consumer_streams
            .insert(consumer_stream);
        self.export_owned_view(view, shape, strides, dtype, guard, consumer_stream)
    }

    fn export_step_view(
        &mut self,
        lease: &SemanticPublishedLease,
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        dlpack_consumer_stream(consumer_stream)?;
        let guard = Arc::clone(&self.checked_step(lease)?.aliases);
        self.steps
            .get_mut(&lease.token)
            .expect("validated output step")
            .consumer_streams
            .insert(consumer_stream);
        self.export_owned_view(view, shape, strides, dtype, guard, consumer_stream)
    }

    fn export_owned_view(
        &mut self,
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
        guard: Arc<()>,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("owned export stream admission", error))?;
        // The actual bank or step has retained this stream before enqueue.
        // Its final retirement joins use even if the capsule was deleted early.
        if let Err(error) = self.order_content_consumers(consumer_stream) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(export_owned_allocation(
            view,
            shape,
            strides,
            dtype,
            self.provider.device().ordinal() as i32,
            guard,
            storage,
            None,
            None,
        ))
    }

    // Shared cold completion for export, final comparison and release. It joins
    // submitted work only, without dropping content, aliases or the native lease.
    fn complete_step_consumers(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_step(lease)?;
        self.complete_step_consumers_by_token(lease.token, consumer_streams)
    }

    fn complete_step_consumers_by_token(
        &mut self,
        token: u64,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        let step = self
            .steps
            .get(&token)
            .ok_or_else(|| publication_input_error("original step has been fully released"))?;
        let mut streams = consumer_streams.iter().copied().collect::<BTreeSet<_>>();
        if !step.consumer_streams.is_subset(&streams)
            || self
                .readers
                .get(&token)
                .is_some_and(|reader| !reader.consumer_streams.is_subset(&streams))
        {
            return Err(publication_input_error(
                "release omits a consumer stream that received a published alias",
            ));
        }
        for &stream in &streams {
            dlpack_consumer_stream(stream)?;
        }
        // Managed tensor producers negotiated readiness onto the legacy default
        // stream. Join it even when validation rejected a handoff before hashing.
        if !step.content.is_empty() {
            streams.insert(1);
        }
        let result = (|| {
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|error| runtime_error("release context binding", error))?;
            for stream in streams {
                let event = self
                    .stream
                    .context()
                    .new_event(None)
                    .map_err(|error| runtime_error("consumer completion event", error))?;
                self.release_events.push(event);
                let event = self.release_events.last().expect("retained consumer event");
                // SAFETY: trusted consumer supplied its actual stream on this
                // device; the driver validates it. Record after the last use.
                unsafe {
                    cudarc::driver::result::event::record(
                        event.cu_event(),
                        dlpack_consumer_stream(stream)?,
                    )
                }
                .map_err(|error| runtime_error("consumer final event record", error))?;
                self.stream
                    .wait(event)
                    .map_err(|error| runtime_error("consumer final event join", error))?;
            }
            wait_on_stream(
                &self.stream,
                &mut self.poisoned,
                &mut self.stream_waits,
                "published reader consumer completion",
                CudaStream::synchronize,
            )?;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        } else {
            // Every retained event's recorded work and native-stream wait has
            // completed. Retire them here, even when export keeps readers live;
            // CudaEvent::drop binds its retained context before destruction.
            self.release_events.clear();
        }
        result
    }

    /// Retire all aliases and their final consumer use while keeping the actual
    /// native reader acquired. This cold boundary performs the same completion
    /// checks as release, but no native decrement. It permits final guarded
    /// comparison after callbacks have relinquished their writable aliases.
    pub fn quiesce_published_reader(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("reader quiescence stream admission", error))?;
        self.checked_reader(lease)?;
        self.quiesce_step_content(lease, consumer_streams)
    }

    fn quiesce_step_content(
        &mut self,
        lease: &SemanticPublishedLease,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        let step = self.checked_step(lease)?;
        if Arc::strong_count(&step.content_witnesses) != 1 {
            return Err(publication_input_error(
                "content witnesses still retain this step",
            ));
        }
        if step.content.is_empty()
            && (Arc::strong_count(&step.aliases) != 1
                || self
                    .readers
                    .get(&lease.token)
                    .is_some_and(|reader| Arc::strong_count(&reader.aliases) != 1))
        {
            return Err(publication_input_error(
                "published tensor aliases still retain this reader",
            ));
        }
        self.complete_step_consumers(lease, consumer_streams)?;
        // Captured producers may themselves be exports of this same parent.
        // Retire those internal aliases only after their final device use and
        // after all witnesses are gone; then distinguish real external aliases.
        let step = self
            .steps
            .get_mut(&lease.token)
            .expect("completed content step");
        step.content.clear();
        if Arc::strong_count(&step.aliases) != 1 {
            return Err(publication_input_error(
                "output aliases still retain this step",
            ));
        }
        if self
            .readers
            .get(&lease.token)
            .is_some_and(|reader| Arc::strong_count(&reader.aliases) != 1)
        {
            return Err(publication_input_error(
                "published tensor aliases still retain this reader",
            ));
        }
        Ok(())
    }

    /// Retire only the actual publication reader after final bank use. Original
    /// numerical content, producer seals and policy outputs retain this step.
    /// A live bank alias still prevents reuse; no retained content is discarded.
    pub fn release_published_reader(
        &mut self,
        lease: &mut SemanticPublishedLease,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        // This public cold boundary waits and reads completion metadata. Keep
        // the entire call outside capture, including the final bank decrement.
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("reader retirement stream admission", error))?;
        self.checked_step(lease)?;
        let reader = self.checked_reader(lease)?;
        if Arc::strong_count(&reader.aliases) != 1
            || self
                .admitted_transition
                .is_some_and(|(token, _, _)| token == lease.token)
        {
            return Err(publication_input_error(
                "publication reader still owns aliases or an admitted transition",
            ));
        }
        self.complete_step_consumers(lease, consumer_streams)?;
        self.retire_published_reader(lease)
    }

    fn retire_published_reader(
        &mut self,
        lease: &mut SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let reader = self.checked_reader(lease)?;
        if Arc::strong_count(&reader.aliases) != 1
            || (self
                .admitted_transition
                .is_some_and(|(token, _, _)| token == lease.token)
                && (self.continuation_base.is_some() || self.text_binding.is_some()))
        {
            return Err(publication_input_error(
                "publication reader still owns aliases or an unfinished transition",
            ));
        }
        let result = (|| {
            self.publication_command(3, Some(lease.token))?;
            let result = self.publication_read(self.readers[&lease.token].device.view())?[0];
            if result.status != 0
                || result.active != 0
                || result.instance != lease.identity.instance
                || result.word != lease.identity.word
            {
                return Err(SemanticTransitionError::PublicationRefused {
                    status: result.status,
                });
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
            return result;
        }
        lease.active = false;
        self.readers.remove(&lease.token);
        if self
            .admitted_transition
            .is_some_and(|(token, _, _)| token == lease.token)
        {
            self.admitted_transition = None;
        }
        Ok(())
    }

    /// All aliases must have returned their managed owners. Completion events
    /// are recorded after final consumer use, joined, and completed before the
    /// sole native reader decrement, unless that reader has already retired.
    /// Failure retains the original step and every unfinished owner.
    pub fn release(
        &mut self,
        lease: &mut SemanticPublishedLease,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("step retirement stream admission", error))?;
        self.quiesce_step_content(lease, consumer_streams)?;
        if lease.active {
            self.retire_published_reader(lease)?;
        }
        self.steps.remove(&lease.token);
        self.release_events.clear();
        Ok(())
    }

    /// Join final consumers and retire one actually completed prepared owner.
    /// The device reader was released by its recorded body; this cold operation
    /// never issues a second decrement or replaces its original private seals.
    pub fn release_prepared_step(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("prepared step retirement stream admission", error))?;
        self.quiesce_prepared_step(step, consumer_streams)?;
        let owner = self
            .steps
            .get_mut(&step.token)
            .expect("completed prepared step");
        let prepared = owner.prepared.as_mut().expect("prepared owner");
        // These are internal handles, not outstanding consumer witnesses.
        // Actual tensor owners remain in content until the final checks below.
        prepared.continuation = None;
        #[cfg(feature = "semantic-policy")]
        {
            prepared.policy_witness = None;
            prepared.policy = None;
            prepared.policy_sources.clear();
        }
        if Arc::strong_count(&owner.content_witnesses) != 1 {
            return Err(publication_input_error(
                "content witnesses still retain this prepared step",
            ));
        }
        owner.content.clear();
        if Arc::strong_count(&owner.aliases) != 1 {
            return Err(publication_input_error(
                "output aliases still retain this prepared step",
            ));
        }
        self.steps.remove(&step.token);
        Ok(())
    }

    /// Establish final device use without discarding the producer owners. This
    /// permits the caller to retire callback-owned aliases outside its mutex.
    pub fn quiesce_prepared_step(
        &mut self,
        step: &SemanticPreparedStep,
        consumer_streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_step(step, false)?;
        if !self
            .prepared_segment
            .as_ref()
            .expect("checked prepared scope")
            .completed
        {
            return Err(publication_input_error(
                "prepared step cannot retire before actual segment completion",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        if self
            .policy_tapes
            .iter()
            .any(|tape| tape.policy.text_binding._witness.reader_token == step.token)
        {
            return Err(SemanticTransitionError::UnconsumedPolicyTape);
        }
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("prepared step quiescence stream admission", error))?;
        self.complete_step_consumers_by_token(step.token, consumer_streams)
    }

    pub fn admit_transition(
        &mut self,
        lease: &SemanticPublishedLease,
        kind: SemanticTransitionKind,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        if kind == SemanticTransitionKind::Update {
            return Err(publication_input_error(
                "updates require a prepared step with native training and pending-output owners",
            ));
        }
        if self.admitted_transition.is_some() || self.continuation_base.is_some() {
            return Err(publication_input_error(
                "an admitted transition already owns the pending invocation",
            ));
        }
        let control = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let current = self.publication_read(control.control.view())?[0];
        if current.instance != lease.identity.instance || current.word != lease.identity.word {
            return Err(publication_input_error(
                "transition requires the current acquired parent, not an older retained reader",
            ));
        }
        if lease.header.terminal == 2 {
            return Err(publication_input_error(
                "final publication admits reads, not another transition",
            ));
        }
        let reserve = if kind == SemanticTransitionKind::Proposal {
            2
        } else {
            1
        };
        if lease.header.fuel < reserve {
            return Err(publication_input_error(
                "acquired publication has insufficient transition fuel",
            ));
        }
        if (lease.header.terminal == 1) != (kind == SemanticTransitionKind::Drain) {
            return Err(SemanticTransitionError::PublicationRefused { status: 4 });
        }
        self.admitted_transition = Some((lease.token, lease.identity.word, kind));
        Ok(())
    }

    /// Snapshot original forward outputs against this exact admitted reader.
    /// Completion records and outputs must come from the forward that consumed
    /// the acquired parent; metadata alone never grants use authority.
    pub fn bind_continuation(
        &mut self,
        lease: &SemanticPublishedLease,
        continuation: SemanticContinuationInput,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let mut handoff = RetainedTensorAdmission {
            inputs: Some(vec![continuation]),
            stream: Arc::clone(self.provider.device().inner().stream()),
            ready: false,
        };
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("continuation stream admission", error))?;
        self.checked_reader(lease)?;
        self.ensure_rebindable()?;
        self.checked_content_witness(lease, witness)?;
        dlpack_consumer_stream(consumer_stream)?;
        let (_, base_word, kind) = self
            .admitted_transition
            .filter(|(token, word, _)| *token == lease.token && *word == lease.identity.word)
            .ok_or_else(|| {
                publication_input_error(
                    "continuation has no admitted invocation for this exact acquired reader",
                )
            })?;
        if self.continuation_base.is_some() {
            return Err(publication_input_error(
                "an unconsumed continuation already owns the pending bank",
            ));
        }
        if !matches!(
            self.steps[&lease.token].content[witness.index].seals,
            TensorContentSeals::Captured(_)
        ) {
            return Err(publication_input_error(
                "continuation requires its original transient producer witness",
            ));
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let continuation = handoff
            .inputs
            .as_ref()
            .expect("retained continuation")
            .first()
            .expect("one continuation");
        let tensor_count = continuation.tensors.len();
        let active_capacity = storage
            .contract_value
            .feedback_capacity
            .checked_add(32)
            .ok_or_else(|| publication_input_error("continuation active capacity overflow"))?;
        let service_layouts = continuation_service_layouts(tensor_count, active_capacity)?;
        if continuation.authority_decisions.is_empty() {
            return Err(publication_input_error(
                "continuation lacks its actual fresh authority decision snapshot",
            ));
        }
        let continuation = handoff
            .inputs
            .take()
            .expect("retained continuation")
            .pop()
            .expect("one continuation");
        let SemanticContinuationInput {
            mut tensors,
            authority_decisions,
            text_rows,
            text_row_count,
            selected_text,
            active_rows,
            active_row_count,
            numerical_admissibility,
        } = continuation;
        tensors.extend(
            [
                text_rows,
                text_row_count,
                selected_text,
                active_rows,
                active_row_count,
                numerical_admissibility,
            ]
            .into_iter()
            .zip(service_layouts)
            .map(|(tensor, layout)| SemanticTensorInput {
                layout,
                logical_begin: 0,
                logical_end: 0,
                tensor,
                native_allocation: None,
            }),
        );
        // Verify the same roster against its original private digest. This is
        // an event-ordered device guard, not a new capture or a host value read.
        self.verify_tensor_content(lease, witness, tensors, consumer_stream)?;
        let original = &self.steps[&lease.token].content[witness.index].tensors;
        let services: [PreparedSemanticTensor; 6] =
            original[tensor_count..].to_vec().try_into().map_err(|_| {
                publication_input_error(
                    "continuation witness has the wrong original service roster",
                )
            })?;
        let text_binding = Arc::new(TextBindingStorage {
            inputs: services,
            _witness: witness.clone(),
            consumer_stream,
            parent: TextBindingParent::Published {
                bank: (lease.identity.word & 1) as usize,
                header: lease.header,
            },
        });
        let authority_bytes = authority_decisions.len() as u64;
        let (uploads, pending_layouts) = prepare_continuation_payloads(
            &storage,
            &original[..tensor_count],
            &authority_decisions,
            kind,
        )?;
        let prepared = PreparedContinuation::prepare(
            &self.provider,
            Arc::clone(&storage),
            self.checked_reader(lease)?.device.view(),
            Arc::clone(&text_binding),
            text_binding.continuation_inputs(kind, authority_bytes, 0, 0, 0, 0),
            None,
            &uploads,
            &pending_layouts,
        )?;
        // Install the same concrete owners before any upload/copy can enqueue.
        // On error or unwind, step retirement and Session quarantine retain
        // sources, original witness/services, destination storage and the lease.
        self.publication_uploads = uploads;
        self.text_binding = Some(Arc::clone(&text_binding));
        let result = (|| {
            for (slot, payload) in &self.publication_uploads {
                if let PublicationPayload::Metadata(bytes) = payload {
                    if !bytes.is_empty() {
                        let mut destination =
                            storage.allocations[*slot].view().slice(..bytes.len());
                        self.provider
                            .htod_launch_metadata_sync_copy_into(bytes, &mut destination)
                            .map_err(|error| {
                                runtime_error("continuation fresh authority upload", error)
                            })?;
                    }
                }
            }
            prepared.enqueue(&self.domain, &mut self.poisoned)
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        if let Err(error) = self.order_content_consumers(consumer_stream) {
            self.poisoned = true;
            return Err(error);
        }
        self.publication_uploads.clear();
        self.continuation_base = Some(base_word);
        if kind != SemanticTransitionKind::Proposal {
            let rng = self.continuation_rng(lease)?;
            self.captured = None;
            self.rng = Some(rng);
            self.next_proposal = u64::from(rng.proposal);
        }
        Ok(())
    }

    pub fn policy_layout(&self) -> Result<SemanticPolicyLayout, SemanticTransitionError> {
        self.ensure_quiescent()?;
        SemanticPolicyLayout::from_components(&self.codebooks.components)
    }

    /// All original category spans in native catalogue order. Inactive TEXT
    /// retains its reserved span, but the device emits singleton NULL without
    /// reading that span. Selection never changes subsequent category offsets.
    pub fn policy_support_layout(
        &self,
    ) -> Result<(Vec<std::ops::Range<usize>>, usize), SemanticTransitionError> {
        self.ensure_quiescent()?;
        Ok(self.codebooks.support_layout())
    }

    pub fn new(
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
    ) -> Result<Self, SemanticTransitionError> {
        let mut graph = provider
            .allocate_semantic_hypergraph(
                domain,
                SemanticHypergraphCapacities::try_new(3, 4, 4, 4)
                    .map_err(SemanticTransitionError::Semantic)?,
            )
            .map_err(SemanticTransitionError::Semantic)?;
        let base = graph.empty_root();
        graph
            .admit_records(
                base,
                SemanticAdmissionRecords {
                    predicates: vec![],
                    records: vec![],
                    supports: vec![],
                },
                SemanticAdmissionLimits {
                    max_records: 0,
                    max_terms: 0,
                    max_references: 0,
                    max_utf8_bytes: 0,
                },
            )
            .map_err(SemanticTransitionError::Semantic)?;
        Self::from_hypergraph(graph)
    }

    /// Consumes the sole admitted semantic owner. The acquired base, accepted
    /// symbol bytes, descriptors and arena remain inseparable for capture/replay.
    pub fn from_hypergraph(graph: SemanticHypergraph) -> Result<Self, SemanticTransitionError> {
        let (provider, domain) = graph
            .transition_owner()
            .map_err(SemanticTransitionError::Semantic)?;
        validate_execution_domain(&provider, &domain)
            .map_err(|e| runtime_error("domain validation", e))?;
        let execute = provider
            .device()
            .inner()
            .get_func("xlog_semantic_transition", "semantic_transition_execute")
            .ok_or_else(|| {
                runtime_error("kernel lookup", "semantic transition kernel unavailable")
            })?;
        let admission = graph
            .admission()
            .expect("transition owner validated admission");
        let root = admission.base();
        let base_snapshot = *admission.base_snapshot();
        let codebooks = ActionCodebooks::derive(admission, graph.transition_arena()[1])?;
        let bytes = codebooks.input_cells * (size_of::<f32>() + size_of::<u8>())
            + SCRATCH_BYTES
            + OBSERVATION_BYTES
            + codebooks.words.len() * 8
            + COMPONENT_COUNT * size_of::<SemanticComponent>();
        let mut reservation = provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|e| runtime_error("reservation", e))?;
        let logits = reservation
            .alloc::<f32>(codebooks.input_cells)
            .map_err(|e| runtime_error("logits allocation", e))?;
        let support = reservation
            .alloc::<u8>(codebooks.input_cells)
            .map_err(|e| runtime_error("support allocation", e))?;
        let scratch = reservation
            .alloc::<u64>(SCRATCH_BYTES / 8)
            .map_err(|e| runtime_error("scratch allocation", e))?;
        let receipts = reservation
            .alloc::<SemanticTransitionReceipt>(COMPONENT_COUNT)
            .map_err(|e| runtime_error("receipt allocation", e))?;
        let state = reservation
            .alloc::<DeviceState>(1)
            .map_err(|e| runtime_error("state allocation", e))?;
        let device_codebooks = reservation
            .alloc::<u64>(codebooks.words.len())
            .map_err(|e| runtime_error("codebook allocation", e))?;
        let device_components = reservation
            .alloc::<SemanticComponent>(COMPONENT_COUNT)
            .map_err(|e| runtime_error("component allocation", e))?;
        debug_assert_eq!(reservation.remaining_bytes(), 0);
        let stream = provider
            .memory()
            .runtime()
            .and_then(|r| r.stream_pool().resolve(domain.stream_id()))
            .ok_or_else(|| runtime_error("stream resolution", "owned stream is unavailable"))?;
        let mut session = Self {
            captured: None,
            provider,
            domain: domain.clone(),
            stream: stream.clone(),
            graph: std::mem::ManuallyDrop::new(graph),
            root,
            base_snapshot,
            codebooks,
            device_codebooks,
            device_components,
            task: None,
            training_views: None,
            training_origins: None,
            task_epoch: 0,
            task_observed: false,
            publication: None,
            publication_uploads: Vec::new(),
            publication_issuer: Arc::new(()),
            readers: BTreeMap::new(),
            steps: BTreeMap::new(),
            next_reader: 0,
            prepared_segment: None,
            prepared_resources: Vec::new(),
            continuation_base: None,
            text_binding: None,
            admitted_transition: None,
            release_events: Vec::new(),
            logits,
            support,
            scratch,
            receipts,
            state,
            pinned: PinnedObservation::new(stream.clone())?,
            execute,
            #[cfg(feature = "semantic-policy")]
            policy: None,
            #[cfg(feature = "semantic-policy")]
            policy_tapes: Vec::new(),
            rng: None,
            next_proposal: 0,
            pending: false,
            poisoned: false,
            stream_waits: 0,
        };
        session.validate_ranges()?;
        session.upload_cold_codebooks()?;
        Ok(session)
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned || self.graph.ensure_not_poisoned().is_err()
    }

    /// Irreversibly close this owner after trusted application validation fails,
    /// including a failure discovered after cold parent binding has completed.
    /// All existing readers/controllers consult this same poison state before
    /// further execution. Repeated aborts are harmless; no resource is freed,
    /// no device work is enqueued, and no poisoned CUDA context is reset.
    /// Existing recorded retirement/quarantine retains outstanding owners.
    /// A fatal CUDA error still requires termination of the isolated process.
    pub fn abort(&mut self) {
        self.poisoned = true;
    }

    /// Inspect the actual cold root, or a decoded replay predecessor, through
    /// the retained native admission before importing its task authority.
    /// The selected replay is not restored here, and these records establish
    /// neither historical execution nor any current use permission.
    pub fn task_observation_roots(
        &mut self,
        spec: &SemanticTaskEvaluationSpec,
        replay: Option<&SemanticReplayMaterial>,
    ) -> Result<SemanticTaskObservationRoots, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some() || self.captured.is_some() {
            return Err(publication_input_error(
                "task observation roots require a cold unpublished Session",
            ));
        }
        let admission = self
            .graph
            .admission()
            .ok_or(SemanticTransitionError::NotBound)?;
        spec.validate_records(admission.records())?;
        if let Some(replay) = replay {
            return replay
                .predecessor
                .graph
                .task_observation_roots(admission, spec.statement_records)
                .map_err(SemanticTransitionError::Semantic);
        }
        let material = self
            .graph
            .export_transition_root(self.root)
            .map_err(SemanticTransitionError::Semantic)?;
        let admission = self
            .graph
            .admission()
            .ok_or(SemanticTransitionError::NotBound)?;
        material
            .task_observation_roots(admission, spec.statement_records)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Execute the supplied native observer and bind its outputs to this admission.
    /// This data binding does not issue publication, inference, or training rights.
    /// The trusted controller must independently import those rights before use.
    pub fn bind_task_evaluation(
        &mut self,
        spec: SemanticTaskEvaluationSpec,
    ) -> Result<Identity256, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some() {
            return Err(publication_input_error(
                "task data is immutable after its canonical parent is initialized",
            ));
        }
        let admission = self
            .graph
            .admission()
            .ok_or(SemanticTransitionError::NotBound)?;
        let next_epoch = self
            .task_epoch
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        spec.validate_records(admission.records())?;
        let observation = spec
            .program
            .observe(Arc::clone(&self.provider))
            .inspect_err(|_error| {
                self.poisoned = true;
            })?;
        let binding = TaskEvaluationBinding::bind(admission, spec, observation)?;
        let words = binding.words(self.graph.transition_arena()[1]);
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes((words.len() * size_of::<u64>()) as u64)
            .map_err(|error| runtime_error("private task reservation", error))?;
        let device = reservation
            .alloc::<u64>(words.len())
            .map_err(|error| runtime_error("private task allocation", error))?;
        // Retain the new owner before the first may-enqueue boundary. A failed
        // upload/adoption poisons reuse and keeps its storage until Session drop.
        self.captured = None;
        let identity = binding.identity();
        self.task = Some((binding, device));
        let (_, device) = self.task.as_mut().expect("private task installed above");
        self.provider
            .htod_sync_copy_into_tracked(&words, device)
            .map_err(|error| {
                self.poisoned = true;
                runtime_error("private task upload", error)
            })?;
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read_write(device);
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |_| {
            Ok::<(), XlogError>(())
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "private task adoption wait",
            CudaStream::synchronize,
        )?;
        self.validate_ranges()?;
        self.task_epoch = next_epoch;
        Ok(identity)
    }

    /// Identity of the immutable task data bank, not a use-authority token.
    pub fn task_evaluation_identity(&self) -> Option<Identity256> {
        self.task.as_ref().map(|(binding, _)| binding.identity())
    }

    /// Bind every admitted training view once while the task is still cold.
    /// No row is selected here and the arena cannot be replaced after binding.
    pub fn bind_training_view_arena(
        &mut self,
        rows: Vec<SemanticTrainingViewRow>,
        objective: SemanticTrainingObjective,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some() || self.training_views.is_some() || self.task.is_none() {
            return Err(publication_input_error(
                "training-view arena requires one cold task binding and cannot be replaced",
            ));
        }
        self.training_views = Some(SemanticTrainingViewArena::allocate(
            &self.provider,
            &self.domain,
            rows,
            objective,
        )?);
        Ok(())
    }

    /// Exact original admission selections in task query order, not inferred
    /// from canonical statement values. Support order and duplicates are kept
    /// as supplied to the validated binding. These indices grant no use rights.
    pub fn task_evaluation_spec(
        &self,
    ) -> Result<&SemanticTaskEvaluationSpec, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let (task, _) = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        Ok(task.spec())
    }

    /// Native learned-value projection for the three fixed queries, in query order.
    /// Each payload is a length-delimited canonical predicate/arity/argument
    /// byte sequence followed by the ordered, length-delimited qualifier values.
    /// Encoding-domain and schema-digest service headers are excluded. The
    /// original qualified key identity remains separately bound to the task.
    /// These schema bytes contain no observer answers or use authority.
    pub fn feedback_statement_bytes(&self) -> Result<[&[u8]; 3], SemanticTransitionError> {
        let selected = self.task_evaluation_spec()?.statement_records;
        let (task, _) = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        Ok(selected.map(|record| task.statement_bytes[&record].as_slice()))
    }

    /// Identity of this build's actual native feedback projection source, not
    /// any statement value, task identity, observer answer or authority decision.
    pub fn feedback_adapter_identity() -> Identity256 {
        let mut hash = Sha256::new();
        hash.update(b"xlog.native.flat-feedback-adapter.v1\0");
        hash.update(include_bytes!("semantic_transition.rs"));
        hash.update(include_bytes!("../kernels/semantic_transition.cu"));
        hash.update(include_bytes!("../kernels/semantic_feedback_encoding.cuh"));
        Identity256::from_bytes(hash.finalize().into())
    }

    /// Changes on every successful task import, even when data bytes are identical.
    pub fn task_evaluation_epoch(&self) -> u64 {
        self.task_epoch
    }

    pub fn binding(&self) -> SemanticCatalogueBinding {
        self.codebooks.binding
    }
    pub fn components(&self) -> &[SemanticComponent] {
        &self.codebooks.components
    }
    pub fn descriptor_meaning(
        &self,
        field: u32,
        category: u32,
    ) -> Option<&SemanticActionDescriptor> {
        let entries = match field {
            2 => &self.codebooks.targets,
            3..=6 => &self.codebooks.operands,
            11 => &self.codebooks.qualifiers,
            17 => &self.codebooks.leaves,
            _ => return None,
        };
        entries.get(category.checked_sub(1)? as usize)
    }

    pub fn host_io_stats(&self) -> SemanticTransitionHostIoStats {
        let io = self.provider.host_transfer_stats();
        let metadata = self.provider.host_launch_metadata_transfer_stats();
        let observation = self.provider.final_observation_transfer_stats();
        SemanticTransitionHostIoStats {
            htod_bytes: io.htod_bytes,
            dtoh_bytes: io.dtoh_bytes,
            htod_calls: io.htod_calls,
            dtoh_calls: io.dtoh_calls,
            launch_metadata_bytes: metadata.htod_bytes,
            launch_metadata_calls: metadata.htod_calls,
            metadata_dtoh_calls: self.provider.untracked_metadata_dtoh_count(),
            observation_bytes: observation.dtoh_bytes,
            observation_calls: observation.dtoh_calls,
            session_stream_waits: self.stream_waits,
        }
    }

    pub fn root_snapshot(&mut self) -> Result<SemanticRootSnapshot, SemanticTransitionError> {
        self.ensure_quiescent()?;
        Ok(self.base_snapshot)
    }

    /// Validates the entire binding before touching resident state. The cold owner
    /// supplies a globally unique stream namespace and the acquired proposal. This
    /// is reconstruction of pending state, not publication or advancement of the
    /// canonical RNG. Rebinding after observation keeps the captured addresses.
    pub fn bind_inputs(
        &mut self,
        binding: SemanticCatalogueBinding,
        rng: SemanticRngBinding,
        inputs: &[SemanticComponentInput],
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if binding != self.binding() {
            return Err(SemanticTransitionError::CatalogueMismatch);
        }
        validate_bound_inputs(&self.codebooks.components, rng, inputs)?;
        self.validate_ranges()?;
        let logits: Vec<f32> = inputs
            .iter()
            .flat_map(|r| r.logits.iter().copied())
            .collect();
        let support: Vec<u8> = inputs
            .iter()
            .flat_map(|r| r.product_support.iter().copied())
            .collect();
        #[cfg(feature = "semantic-policy")]
        if self.policy.is_some() {
            self.captured = None;
            self.policy = None;
        }
        // The provider's synchronous uploader uses its setup stream. Complete it
        // before adopting allocations onto the distinct resident execution stream.
        let upload = (|| {
            self.provider
                .htod_sync_copy_into_tracked(&logits, &mut self.logits)?;
            self.provider
                .htod_sync_copy_into_tracked(&support, &mut self.support)?;
            Ok::<(), XlogError>(())
        })();
        upload.map_err(|e| {
            self.poisoned = true;
            runtime_error("input upload", e)
        })?;
        self.upload_rng_state(binding, rng)?;
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read_write(&self.logits);
        recorder.read_write(&self.support);
        recorder.read_write(&self.state);
        recorder.write(&self.scratch);
        recorder.write(&self.receipts);
        recorder.read(&self.device_codebooks);
        recorder.read(&self.device_components);
        // SAFETY: completed setup copies precede this ownership adoption. No work
        // escapes the supplied stream, and no output payload is initialized here.
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |_| {
            Ok::<(), XlogError>(())
        })?;
        self.stream_waits += 1;
        self.stream.synchronize().map_err(|e| {
            self.poisoned = true;
            runtime_error("input adoption wait", e)
        })?;
        self.rng = Some(rng);
        self.next_proposal = u64::from(rng.proposal);
        Ok(())
    }

    /// Takes independent device snapshots of the exact proposal's model inputs.
    /// Text logits retain all 32 rows of the original MASK matrix;
    /// both lanes read the same original row. Structural scores are
    /// produced by the pinned numerical policy inside the component loop.
    ///
    /// Parameters use `policy_layout()`. Source allocations must be tracked in
    /// this execution domain; the recorder joins their producer streams before
    /// copying. No caller-owned parameter buffer is used by subsequent replay.
    #[cfg(feature = "semantic-policy")]
    pub fn bind_policy(
        &mut self,
        binding: SemanticCatalogueBinding,
        rng: SemanticRngBinding,
        text_logits: &TrackedCudaSlice<f32>,
        product_support: &TrackedCudaSlice<u8>,
        parameters: &TrackedCudaSlice<f32>,
        component_baselines: &TrackedCudaSlice<f32>,
    ) -> Result<(), SemanticTransitionError> {
        self.bind_policy_views(
            binding,
            rng,
            text_logits.view(),
            product_support.view(),
            parameters.view(),
            component_baselines.view(),
        )
    }

    /// Import genuine DLPack owners and snapshot one proposal's policy inputs.
    ///
    /// All tensors are contiguous CUDA storage on this session's device:
    /// F32 `[32, text_cardinality()]`, Bool8 `[policy_support_layout().1]`,
    /// F32 `[policy_layout().parameter_cells]`, and FP32 `[1, 136]` component
    /// baselines. Both lanes share the original
    /// mapped MASK rows; unused capacity is not a categorical factor.
    /// Support contains only active TEXT and the structural rows in roster order;
    /// inactive TEXT has no address. The roster itself still has 136 factors.
    ///
    /// The canonical importer completes the producer handoff on the provider's
    /// legacy-default stream before this session's recorded device-to-device
    /// snapshot. It uploads only bounded row-count metadata, not tensor values.
    /// The caller must meet `DlpackManagedTensor::from_raw`'s producer-readiness
    /// and external-access contract; these handles confer no task/use authority.
    /// Original model/autograd owners remain the caller's separate responsibility.
    #[cfg(feature = "semantic-policy")]
    pub fn bind_policy_dlpack(
        &mut self,
        binding: SemanticCatalogueBinding,
        rng: SemanticRngBinding,
        text_logits: DlpackManagedTensor,
        product_support: DlpackManagedTensor,
        parameters: DlpackManagedTensor,
        component_baselines: DlpackManagedTensor,
    ) -> Result<(), SemanticTransitionError> {
        // Own all actual producers before the first fallible binding/shape check.
        // A failed handoff retains the entire set through canonical retirement.
        let mut handoff = RetainedTensorAdmission {
            inputs: Some(vec![
                text_logits,
                product_support,
                parameters,
                component_baselines,
            ]),
            stream: Arc::clone(self.provider.device().inner().stream()),
            ready: false,
        };
        self.ensure_rebindable()?;
        if binding != self.binding() {
            return Err(SemanticTransitionError::CatalogueMismatch);
        }
        if rng.stream_serial >= (1u64 << 56) {
            return Err(SemanticTransitionError::InvalidInput {
                detail: "policy stream namespace does not match the native catalogue".into(),
            });
        }
        let layout = self.policy_layout()?;
        let layouts = policy_producer_layouts(self.codebooks.input_cells, layout.parameter_cells)?;
        // Preserve the actual rank-two MASK matrix and each producer owner.
        // The typed importer validates original metadata; no column adapter,
        // compacted rows or replacement managed tensor is manufactured.
        let inputs = handoff
            .inputs
            .take()
            .expect("retained policy producers")
            .into_iter()
            .zip(layouts)
            .map(|(tensor, layout)| SemanticTensorInput {
                layout,
                logical_begin: 0,
                logical_end: 0,
                tensor,
                native_allocation: None,
            })
            .collect();
        let views = prepare_semantic_tensors(&self.provider, inputs)?
            .into_iter()
            .map(|input| {
                input
                    .source
                    .ok_or_else(|| publication_input_error("policy producer storage is empty"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [text_logits, product_support, parameters, component_baselines]: [_; 4] = views
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact policy producer roster"));
        // SAFETY: the original typed metadata establishes F32 representation,
        // alignment and exact extents. Casts retain the foreign allocation.
        let text_logits = unsafe { text_logits.cast::<f32>() }.ok_or_else(|| {
            SemanticTransitionError::InvalidInput {
                detail: "policy text DLPack span is not aligned F32 storage".into(),
            }
        })?;
        // SAFETY: the same exact F32 admission holds for the parameter column.
        let parameters = unsafe { parameters.cast::<f32>() }.ok_or_else(|| {
            SemanticTransitionError::InvalidInput {
                detail: "policy parameter DLPack span is not aligned F32 storage".into(),
            }
        })?;
        // SAFETY: the fourth original producer has exact contiguous
        // FP32[1, COMPONENT_COUNT] metadata from policy_producer_layouts.
        let component_baselines =
            unsafe { component_baselines.cast::<f32>() }.ok_or_else(|| {
                SemanticTransitionError::InvalidInput {
                    detail: "component baseline DLPack span is not aligned F32 storage".into(),
                }
            })?;
        self.bind_policy_views(
            binding,
            rng,
            text_logits,
            product_support,
            parameters,
            component_baselines,
        )
    }

    #[cfg(feature = "semantic-policy")]
    fn allocate_policy_buffers(&self) -> Result<PolicyBuffers, SemanticTransitionError> {
        let layout = self.policy_layout()?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(policy_buffer_bytes(&layout)? as u64)
            .map_err(|error| runtime_error("policy reservation", error))?;
        self.allocate_policy_buffers_reserved(&mut reservation)
    }

    #[cfg(feature = "semantic-policy")]
    fn allocate_policy_buffers_reserved(
        &self,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<PolicyBuffers, SemanticTransitionError> {
        let layout = self.policy_layout()?;
        let text_cells = 32 * TEXT_CARDINALITY;
        Ok(PolicyBuffers {
            parameters: reservation
                .alloc::<f32>(layout.parameter_cells)
                .map_err(|error| runtime_error("policy parameter allocation", error))?,
            hidden: reservation
                .alloc::<f32>(128)
                .map_err(|error| runtime_error("policy hidden allocation", error))?,
            scores: reservation
                .alloc::<f32>(layout.score_cells())
                .map_err(|error| runtime_error("policy score allocation", error))?,
            recurrent: reservation
                .alloc::<f32>(layout.recurrent_cells())
                .map_err(|error| runtime_error("policy recurrent allocation", error))?,
            text_logits: reservation
                .alloc::<f32>(text_cells)
                .map_err(|error| runtime_error("original text snapshot allocation", error))?,
            component_baselines: reservation
                .alloc::<f32>(COMPONENT_COUNT)
                .map_err(|error| runtime_error("component baseline snapshot allocation", error))?,
            adjoints: Some(PolicyAdjointBuffers {
                parameters: reservation
                    .alloc::<f32>(layout.parameter_cells)
                    .map_err(|error| runtime_error("parameter adjoint allocation", error))?,
                text_logits: reservation
                    .alloc::<f32>(text_cells)
                    .map_err(|error| runtime_error("text adjoint allocation", error))?,
                recurrent: reservation
                    .alloc::<f32>(layout.recurrent_cells())
                    .map_err(|error| runtime_error("recurrent adjoint allocation", error))?,
                scores: reservation
                    .alloc::<f32>(layout.score_cells())
                    .map_err(|error| runtime_error("score adjoint allocation", error))?,
                coefficients: reservation
                    .alloc::<f64>(COMPONENT_COUNT)
                    .map_err(|error| runtime_error("cotangent snapshot allocation", error))?,
                status: reservation
                    .alloc::<u64>(1)
                    .map_err(|error| runtime_error("policy backward status allocation", error))?,
            }),
            layout,
        })
    }

    #[cfg(feature = "semantic-policy")]
    fn prepare_step_policy_storage(
        &self,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<PreparedPolicyStorage, SemanticTransitionError> {
        let buffers = self.allocate_policy_buffers_reserved(reservation)?;
        let support = reservation
            .alloc(self.codebooks.input_cells)
            .map_err(|error| runtime_error("prepared support allocation", error))?;
        let receipts = reservation
            .alloc(COMPONENT_COUNT)
            .map_err(|error| runtime_error("prepared receipt allocation", error))?;
        let state = reservation
            .alloc(1)
            .map_err(|error| runtime_error("prepared state allocation", error))?;
        Ok((buffers, support, receipts, state))
    }

    /// Retain the original policy producers and bind their already reserved
    /// native banks. No host publication identity or policy draw is created.
    #[cfg(feature = "semantic-policy")]
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared policy binding retains each typed producer and witness explicitly"
    )]
    pub fn bind_prepared_policy(
        &mut self,
        step: &SemanticPreparedStep,
        binding: SemanticCatalogueBinding,
        text_logits: DlpackManagedTensor,
        product_support: DlpackManagedTensor,
        parameters: DlpackManagedTensor,
        component_baselines: DlpackManagedTensor,
        witness: &SemanticTensorContentWitness,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, consumer_stream)?;
        if self.prepared_transition_kind(step)? != SemanticTransitionKind::Proposal {
            return Err(publication_input_error(
                "only a proposal step binds a learned policy",
            ));
        }
        if binding != self.binding() {
            return Err(SemanticTransitionError::CatalogueMismatch);
        }
        let layouts = policy_producer_layouts(
            self.codebooks.input_cells,
            self.policy_layout()?.parameter_cells,
        )?;
        let tensors = [
            text_logits,
            product_support,
            parameters,
            component_baselines,
        ]
        .into_iter()
        .zip(layouts)
        .map(|(tensor, layout)| SemanticTensorInput {
            layout,
            logical_begin: 0,
            logical_end: 0,
            tensor,
            native_allocation: None,
        })
        .collect();
        self.verify_prepared_tensor_content(step, witness, tensors, consumer_stream)?;
        let sources = self.steps[&step.token].content[witness.index]
            .tensors
            .clone();
        let owner = self
            .steps
            .get_mut(&step.token)
            .expect("checked original step");
        let prepared = owner.prepared.as_mut().expect("prepared policy owner");
        if prepared.policy.is_some() || !prepared.policy_sources.is_empty() {
            return Err(publication_input_error(
                "prepared policy already owns its original invocation",
            ));
        }
        let text_binding = Arc::clone(
            &prepared
                .continuation
                .as_ref()
                .ok_or_else(|| {
                    publication_input_error(
                        "prepared policy requires its original model continuation",
                    )
                })?
                .text_binding,
        );
        if text_binding._witness.reader_token != step.token {
            return Err(publication_input_error(
                "prepared policy and continuation belong to different original steps",
            ));
        }
        let buffers = prepared.policy_buffers.take().ok_or_else(|| {
            publication_input_error("prepared policy has no unused cold allocation")
        })?;
        prepared.policy_sources = sources;
        prepared.policy_witness = Some(witness.clone());
        prepared.policy = Some(PolicyStorage {
            buffers,
            text_binding,
            replacement: None,
            tape_live: false,
        });
        Ok(())
    }

    /// Consume the original numerical body through the canonical policy,
    /// resolver and publication kernel, with this step's distinct late tape.
    #[cfg(feature = "semantic-policy")]
    pub fn enqueue_prepared_transition(
        &mut self,
        step: &SemanticPreparedStep,
        bank: usize,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        if bank > 1 {
            return Err(publication_input_error(
                "prepared transition bank must be zero or one",
            ));
        }
        let kind = self.prepared_transition_kind(step)?;
        let finish = self
            .steps
            .get_mut(&step.token)
            .expect("checked original step")
            .prepared
            .as_mut()
            .expect("prepared owner")
            .model_work
            .as_mut()
            .ok_or_else(|| {
                publication_input_error("prepared transition requires original model work")
            })
            .and_then(|work| work.finish_capture(bank).map_err(publication_input_error));
        if let Err(error) = finish {
            self.poisoned = true;
            return Err(error);
        }
        let owner = &self.steps[&step.token];
        let prepared = owner.prepared.as_ref().expect("prepared transition owner");
        if prepared.transition_recorded & (1 << bank) != 0
            || prepared.continuation.is_none()
            || (kind == SemanticTransitionKind::Proposal && prepared.policy.is_none())
            || (kind != SemanticTransitionKind::Proposal && prepared.policy.is_some())
            || (kind == SemanticTransitionKind::Update
                && prepared
                    .model_update
                    .as_ref()
                    .is_none_or(|update| update.output.is_none()))
        {
            return Err(publication_input_error("prepared transition requires its unused original continuation, a policy only for proposals, and a bound model output for updates"));
        }
        let model_witnesses = owner
            .content
            .iter()
            .enumerate()
            .filter(|(_, content)| matches!(content.seals, TensorContentSeals::Model(_)))
            .map(|(index, _)| SemanticTensorContentWitness {
                issuer: Arc::clone(&self.publication_issuer),
                reader_token: step.token,
                index,
                _witness: Arc::clone(&owner.content_witnesses),
            })
            .collect::<Vec<_>>();
        if model_witnesses.is_empty() {
            return Err(publication_input_error(
                "prepared transition has no original admitted model content",
            ));
        }
        let policy_witness = if kind == SemanticTransitionKind::Proposal {
            Some(prepared.policy_witness.clone().ok_or_else(|| {
                publication_input_error("prepared transition has no original policy content")
            })?)
        } else {
            None
        };
        let continuation_witness = prepared
            .continuation
            .as_ref()
            .expect("checked continuation")
            .text_binding
            ._witness
            .clone();
        for witness in &model_witnesses {
            self.record_prepared_tensor_content(step, witness, true)?;
        }
        if let Some(witness) = &policy_witness {
            self.record_prepared_tensor_content(step, witness, true)?;
        }
        self.record_prepared_tensor_content(step, &continuation_witness, true)?;
        let result = (|| {
            let prepared = self.steps[&step.token]
                .prepared
                .as_ref()
                .expect("retained original step");
            if let Some(policy) = &prepared.policy {
                let sources = policy_source_views(&prepared.policy_sources)?;
                enqueue_policy_snapshots(
                    &self.domain,
                    &mut self.poisoned,
                    &policy.buffers,
                    prepared.support.as_ref().expect("original support bank"),
                    &sources,
                )?;
            }
            prepared
                .continuation
                .as_ref()
                .expect("checked original continuation")
                .enqueue(&self.domain, &mut self.poisoned)?;
            if let Some(update) = &prepared.model_update {
                update.enqueue_copy(
                    &self.domain,
                    &mut self.poisoned,
                    self.publication.as_ref().expect("prepared publication"),
                    &prepared.reader,
                )?;
            }
            let io = self.prepared_kernel_io(step)?;
            self.validate_ranges_with(&io)?;
            let descriptor = self.descriptor_with(&io);
            let recorder = self.kernel_recorder_with(&io);
            let execute = self.execute.clone();
            enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
                // SAFETY: the same selected original buffers define both the
                // kernel arguments and every retained recorder allocation.
                unsafe {
                    execute.launch_in(
                        enqueue,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (descriptor,),
                    )
                }
                .map_err(|error| XlogError::Kernel(error.to_string()))
            })
        })();
        if result.is_err() {
            self.poisoned = true;
            return result;
        }
        let prepared = self
            .steps
            .get_mut(&step.token)
            .expect("retained original step")
            .prepared
            .as_mut()
            .expect("retained original prepared step");
        prepared.transition_recorded |= 1 << bank;
        Ok(())
    }

    /// Record the device-selected terminal drain without executing the model,
    /// continuation producer, optimizer, or training-output path.
    #[cfg(feature = "semantic-policy")]
    pub fn enqueue_prepared_drain(
        &mut self,
        step: &SemanticPreparedStep,
    ) -> Result<(), SemanticTransitionError> {
        self.check_prepared_content_stream(step, self.stream.cu_stream() as u64)?;
        let prepared = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared drain owner");
        if prepared.transition_recorded != PREPARED_TRANSITION_BANKS || prepared.drain_recorded {
            return Err(publication_input_error(
                "prepared drain requires its recorded requested branch and unused drain branch",
            ));
        }
        let publication = self.publication.as_ref().expect("prepared publication");
        let mut recorder = self.domain.new_strict_recorder();
        publication.record(&mut recorder);
        recorder.read(&prepared.reader);
        let arguments = (
            publication.control.device_ptr_value(),
            prepared.reader.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the device-selected drain uses the same acquired lease and
            // publication owner as the following canonical transition kernel.
            unsafe {
                prepared.drain_prepare.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        let io = self.prepared_drain_kernel_io(step)?;
        self.validate_ranges_with(&io)?;
        let descriptor = self.descriptor_with(&io);
        let recorder = self.kernel_recorder_with(&io);
        let execute = self.execute.clone();
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the drain descriptor retains the canonical publication,
            // task, graph arena, result, and lease owners but no model outputs.
            unsafe {
                execute.launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (descriptor,),
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        self.steps
            .get_mut(&step.token)
            .expect("prepared drain owner")
            .prepared
            .as_mut()
            .expect("prepared drain storage")
            .drain_recorded = true;
        Ok(())
    }

    /// Move an actually observed proposal's original banks into late backward
    /// ownership. The caller supplies coordinates from its saved native header.
    #[cfg(feature = "semantic-policy")]
    fn take_prepared_policy_tape(
        &mut self,
        step: &SemanticPreparedStep,
        invocation: SemanticRngBinding,
        refusal: Option<SemanticTransitionRefusal>,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&step.issuer, &self.publication_issuer)
            || self
                .policy_tapes
                .iter()
                .any(|tape| tape.invocation == invocation)
        {
            return Err(publication_input_error(
                "observed prepared policy differs from its original Session or invocation",
            ));
        }
        let binding = self.binding();
        let prepared = self
            .steps
            .get_mut(&step.token)
            .and_then(|owner| owner.prepared.as_mut())
            .ok_or_else(|| {
                publication_input_error("observed prepared policy has no retained original step")
            })?;
        if prepared.transition_recorded != PREPARED_TRANSITION_BANKS
            || prepared.policy.is_none()
            || prepared.support.is_none()
            || prepared.receipts.is_none()
        {
            return Err(publication_input_error(
                "observed prepared policy was not recorded or has already retired",
            ));
        }
        // A captured nominal proposal may be skipped or forced to drain. Only
        // actual proposal completion creates this original backward obligation.
        prepared
            .policy
            .as_mut()
            .expect("checked original policy")
            .tape_live = true;
        self.policy_tapes.push(PolicyTape {
            invocation,
            refusal,
            binding,
            policy: prepared.policy.take().expect("checked original policy"),
            support: prepared.support.take().expect("checked original support"),
            receipts: prepared.receipts.take().expect("checked original receipts"),
            components: self.device_components.view(),
            codebooks: self.device_codebooks.view(),
        });
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    fn bind_policy_views(
        &mut self,
        binding: SemanticCatalogueBinding,
        rng: SemanticRngBinding,
        text_logits: DeviceMemoryView<f32>,
        product_support: DeviceMemoryView<u8>,
        parameters: DeviceMemoryView<f32>,
        component_baselines: DeviceMemoryView<f32>,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if binding != self.binding() {
            return Err(SemanticTransitionError::CatalogueMismatch);
        }
        let layout = self.policy_layout()?;
        let text_binding = Arc::clone(
            self.text_binding
                .as_ref()
                .filter(|_| self.continuation_base.is_some())
                .ok_or_else(|| {
                    publication_input_error(
                        "policy inputs require an admitted original model continuation",
                    )
                })?,
        );
        if rng != text_binding.published_parent()?.1.rng_binding()? {
            return Err(publication_input_error(
                "policy invocation differs from its actual acquired parent",
            ));
        }
        let text_cells = 32 * TEXT_CARDINALITY;
        let support_cells = self.codebooks.input_cells;
        if self
            .admitted_transition
            .is_none_or(|(_, _, kind)| kind != SemanticTransitionKind::Proposal)
            || self.policy_tapes.iter().any(|tape| tape.invocation == rng)
        {
            return Err(publication_input_error(
                "policy binding requires a fresh admitted proposal invocation",
            ));
        }
        if rng.stream_serial >= (1u64 << 56)
            || text_logits.len() != text_cells
            || product_support.len() != support_cells
            || parameters.len() != layout.parameter_cells
            || component_baselines.len() != COMPONENT_COUNT
        {
            return Err(SemanticTransitionError::InvalidInput {
                detail:
                    "policy input shape or stream namespace does not match the native catalogue"
                        .into(),
            });
        }
        let policy = PolicyStorage {
            // Reserve the next working banks before the proposal may publish.
            // Successful observation moves, rather than overwrites, all exact
            // original buffers into its late tape without fallible allocation.
            replacement: Some(PolicyReplacement {
                support: allocate_publication(&self.provider, self.codebooks.input_cells)?,
                receipts: allocate_publication(&self.provider, COMPONENT_COUNT)?,
            }),
            buffers: self.allocate_policy_buffers()?,
            tape_live: false,
            text_binding,
        };
        // Changing captured pointer arguments invalidates the old executable.
        // All previous work is quiescent and no unconsumed tape can reach here.
        self.captured = None;
        self.rng = None;
        self.policy = Some(policy);
        self.validate_ranges()?;
        let original = Arc::clone(&self.policy.as_ref().expect("installed policy").text_binding);
        self.guard_continuation_content(&original)?;
        let policy = self
            .policy
            .as_ref()
            .expect("policy storage installed above");
        enqueue_policy_snapshots(
            &self.domain,
            &mut self.poisoned,
            &policy.buffers,
            &self.support,
            &PolicyProducerViews {
                text_logits,
                product_support,
                parameters,
                component_baselines,
            },
        )?;
        if let Err(error) = self.order_content_consumers(original.consumer_stream) {
            self.poisoned = true;
            return Err(error);
        }
        self.rng = Some(rng);
        self.next_proposal = u64::from(rng.proposal);
        Ok(())
    }

    /// Differentiates the selected scores after successful observation. The
    /// caller supplies one FP64 scalar cotangent per original component;
    /// advantages and importance weights are not recomputed or differentiated
    /// here. The owned FP64 snapshot stays at that precision until the selected
    /// logit adjoints enter the existing FP32 model backward.
    ///
    /// This is a terminal device operation, not another transition: forward
    /// receipts, recurrent values, semantic state and RNG are only read. The
    /// original tape's preallocated outputs cannot be overwritten by a later
    /// proposal; backward neither reserves nor allocates device memory.
    #[cfg(feature = "semantic-policy")]
    pub fn backward_policy(
        &mut self,
        invocation: SemanticRngBinding,
        score_cotangents: &TrackedCudaSlice<f64>,
    ) -> Result<SemanticPolicyGradients, SemanticTransitionError> {
        self.backward_policy_view(invocation, score_cotangents.view(), 1)
    }

    /// Consume the actual original loss's 136 FP64 selected-score cotangents.
    /// The DLPack producer is joined before validation and retained on failures;
    /// no Python-selected scores or replacement autograd leaf are introduced.
    #[cfg(feature = "semantic-policy")]
    pub fn backward_policy_dlpack(
        &mut self,
        invocation: SemanticRngBinding,
        score_cotangents: DlpackManagedTensor,
        consumer_stream: u64,
    ) -> Result<SemanticPolicyGradients, SemanticTransitionError> {
        let mut handoff = RetainedTensorAdmission {
            inputs: Some(vec![score_cotangents]),
            stream: Arc::clone(self.provider.device().inner().stream()),
            ready: false,
        };
        dlpack_consumer_stream(consumer_stream)?;
        let pointer = handoff
            .inputs
            .as_ref()
            .expect("retained cotangent producer")[0]
            .as_ptr();
        if pointer.is_null() {
            return Err(publication_input_error(
                "selected-score cotangent managed owner is null",
            ));
        }
        // SAFETY: the genuine managed owner retains the original descriptor and
        // shape/stride arrays throughout this metadata-only admission.
        let (data, bytes) = unsafe {
            policy_score_cotangent_metadata(
                &(*pointer).dl_tensor,
                self.provider.device().ordinal() as i32,
            )
        }?;
        handoff.ready = true;
        let tensor = handoff
            .inputs
            .take()
            .expect("retained cotangent producer")
            .pop()
            .expect("one cotangent");
        // SAFETY: canonical metadata validates the exact original span. The
        // column transfers, rather than replaces, its retained producer owner.
        let bytes = unsafe { CudaColumn::dlpack(data, bytes, Arc::clone(&handoff.stream), tensor) }
            .device_view();
        // SAFETY: original metadata proves the full contiguous F64 roster.
        // The continuation guard orders producer events before its device use.
        let values = unsafe { bytes.cast::<f64>() }.ok_or_else(|| {
            publication_input_error("selected-score cotangent span is not aligned F64 storage")
        })?;
        self.backward_policy_view(invocation, values, consumer_stream)
    }

    #[cfg(feature = "semantic-policy")]
    fn checked_prepared_policy_tape(
        &self,
        step: &SemanticPreparedStep,
        invocation: SemanticRngBinding,
    ) -> Result<usize, SemanticTransitionError> {
        let owner = self.checked_prepared_step(step, false)?;
        if !self
            .prepared_segment
            .as_ref()
            .is_some_and(|build| build.completed)
            || !owner
                .prepared
                .as_ref()
                .is_some_and(|prepared| prepared.observed)
        {
            return Err(publication_input_error(
                "prepared policy requires its actual completed native outcome",
            ));
        }
        self.policy_tapes
            .iter()
            .position(|tape| {
                tape.invocation == invocation
                    && tape.policy.text_binding._witness.reader_token == step.token
            })
            .ok_or_else(|| {
                publication_input_error(
                    "prepared policy has no unconsumed original tape for this step",
                )
            })
    }

    /// Authenticate the actual completed step and its unconsumed original tape
    /// before a binding exposes this invocation to its numerical producer.
    #[cfg(feature = "semantic-policy")]
    pub fn require_prepared_policy_invocation(
        &self,
        step: &SemanticPreparedStep,
        invocation: SemanticRngBinding,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_prepared_policy_tape(step, invocation)
            .map(|_| ())
    }

    /// Differentiate the original tape of this actually completed native step.
    #[cfg(feature = "semantic-policy")]
    pub fn backward_prepared_policy_dlpack(
        &mut self,
        step: &SemanticPreparedStep,
        invocation: SemanticRngBinding,
        score_cotangents: DlpackManagedTensor,
        consumer_stream: u64,
    ) -> Result<SemanticPolicyGradients, SemanticTransitionError> {
        self.checked_prepared_policy_tape(step, invocation)?;
        self.backward_policy_dlpack(invocation, score_cotangents, consumer_stream)
    }

    #[cfg(feature = "semantic-policy")]
    pub fn finish_prepared_policy_invocation(
        &mut self,
        step: &SemanticPreparedStep,
        invocation: SemanticRngBinding,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let index = self.checked_prepared_policy_tape(step, invocation)?;
        self.finish_policy_tape(index, consumer_stream)
    }

    /// Retire an exact values-only invocation after the caller's final device
    /// use. This does not differentiate, reacquire a parent, or release readers.
    /// Final step release remains the cold completion boundary.
    #[cfg(feature = "semantic-policy")]
    pub fn finish_policy_invocation(
        &mut self,
        lease: &SemanticPublishedLease,
        invocation: SemanticRngBinding,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_step(lease)?;
        dlpack_consumer_stream(consumer_stream)?;
        let index = self
            .policy_tapes
            .iter()
            .position(|tape| tape.invocation == invocation)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let original = Arc::clone(&self.policy_tapes[index].policy.text_binding);
        self.checked_content_witness(lease, &original._witness)?;
        self.finish_policy_tape(index, consumer_stream)
    }

    #[cfg(feature = "semantic-policy")]
    fn finish_policy_tape(
        &mut self,
        index: usize,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        dlpack_consumer_stream(consumer_stream)?;
        let original = Arc::clone(&self.policy_tapes[index].policy.text_binding);
        let result = (|| {
            self.order_step_content_inputs(original._witness.reader_token, consumer_stream)?;
            self.guard_continuation_content(&original)?;
            self.order_content_consumers(consumer_stream)
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        self.policy_tapes.remove(index);
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    #[expect(
        clippy::too_many_arguments,
        reason = "policy VJP recording retains each native owner explicitly"
    )]
    fn prepare_policy_vjp_recording(
        &self,
        policy: &PolicyStorage,
        support: &TrackedCudaSlice<u8>,
        receipts: &TrackedCudaSlice<SemanticTransitionReceipt>,
        state: &TrackedCudaSlice<DeviceState>,
        components: &DeviceMemoryView<SemanticComponent>,
        codebooks: &DeviceMemoryView<u64>,
        score_cotangents: &DeviceMemoryView<f64>,
    ) -> Result<PolicyVjpRecording, SemanticTransitionError> {
        if score_cotangents.len() != COMPONENT_COUNT {
            return Err(SemanticTransitionError::InvalidInput {
                detail: "selected-score cotangents must follow the complete component roster"
                    .into(),
            });
        }
        let PolicyAdjointBuffers {
            parameters,
            text_logits,
            recurrent,
            scores,
            coefficients,
            status,
        } = policy.adjoints.as_ref().ok_or_else(|| {
            publication_input_error("original policy adjoint banks have already been consumed")
        })?;
        let io = TransitionKernelIo {
            logits: &policy.text_logits,
            support,
            receipts,
            state,
            policy: Some(policy),
            text: Some(&policy.text_binding),
            lease: None,
            model_work: None,
            model_update: None,
            training_selection: None,
        };
        let mut descriptor = self.descriptor_with(&io);
        descriptor.components = *components.device_ptr();
        descriptor.codebooks = *codebooks.device_ptr();
        descriptor.publication = PublicationCommand::default();
        descriptor.backward = PolicyBackward {
            cotangents: coefficients.device_ptr_value(),
            parameters: parameters.device_ptr_value(),
            text: text_logits.device_ptr_value(),
            recurrent: recurrent.device_ptr_value(),
            scores: scores.device_ptr_value(),
            status: status.device_ptr_value(),
            parameter_cells: policy.layout.parameter_cells as u64,
            text_cells: policy.text_logits.len() as u64,
        };
        self.validate_ranges_with(&io)?;
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(support);
        recorder.read(receipts);
        recorder.read(codebooks);
        recorder.read(components);
        recorder.read(&policy.parameters);
        recorder.read(&policy.recurrent);
        recorder.read(&policy.text_logits);
        policy.text_binding.record(&mut recorder);
        recorder.read(score_cotangents);
        recorder.write(&self.scratch);
        recorder.write(&policy.hidden);
        recorder.write(&policy.scores);
        for buffer in [parameters, text_logits, recurrent, scores] {
            recorder.write(buffer);
        }
        recorder.write(coefficients);
        recorder.write(status);
        Ok(PolicyVjpRecording {
            descriptor,
            recorder,
        })
    }

    #[cfg(feature = "semantic-policy")]
    fn backward_policy_view(
        &mut self,
        invocation: SemanticRngBinding,
        score_cotangents: DeviceMemoryView<f64>,
        consumer_stream: u64,
    ) -> Result<SemanticPolicyGradients, SemanticTransitionError> {
        self.ensure_quiescent()?;
        dlpack_consumer_stream(consumer_stream)?;
        let tape_index = self
            .policy_tapes
            .iter()
            .position(|tape| tape.invocation == invocation)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if let Some(refusal) = self.policy_tapes[tape_index].refusal {
            return Err(refusal.into_error());
        }
        let original = Arc::clone(&self.policy_tapes[tape_index].policy.text_binding);
        self.guard_continuation_content(&original)?;
        if let Err(error) =
            self.order_step_content_inputs(original._witness.reader_token, consumer_stream)
        {
            self.poisoned = true;
            return Err(error);
        }
        let publication = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let step_aliases = Arc::clone(&self.steps[&original._witness.reader_token].aliases);
        let tape = &self.policy_tapes[tape_index];
        let policy = &tape.policy;
        let state = self.steps[&original._witness.reader_token]
            .prepared
            .as_ref()
            .map_or(&self.state, |prepared| &prepared.state);
        let PolicyVjpRecording {
            descriptor,
            recorder,
        } = self.prepare_policy_vjp_recording(
            policy,
            &tape.support,
            &tape.receipts,
            state,
            &tape.components,
            &tape.codebooks,
            &score_cotangents,
        )?;
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|e| runtime_error("policy backward stream admission", e))?;
        let execute = self.execute.clone();
        let layout = policy.layout.clone();
        let binding = tape.binding;
        let support_cells = tape.support.len();
        // Consume only after every fallible admission check. The recorder owns
        // the scratch banks through execution; exported outputs retain their
        // original step until the final consumer-stream join.
        let PolicyAdjointBuffers {
            parameters,
            text_logits,
            recurrent,
            scores,
            coefficients,
            status,
        } = self.policy_tapes[tape_index]
            .policy
            .buffers
            .adjoints
            .take()
            .expect("checked original adjoint banks");
        let vjp_workspace = Arc::new(PolicyVjpWorkspace {
            _recurrent: recurrent,
            _scores: scores,
            coefficients,
            _status: status,
            score_cotangents,
        });
        // A capsule can be deleted before its consumer finishes. The original
        // step keeps the actual outputs until its final consumer-stream join,
        // without keeping a witness (which would create a release cycle).
        let output_views = [&parameters, &text_logits].into_iter().map(|buffer| {
            // SAFETY: F32 storage has an exact, aligned byte representation.
            unsafe { buffer.view().cast::<u8>() }.expect("F32 adjoint byte view")
        });
        let step_owner = self
            .steps
            .get_mut(&original._witness.reader_token)
            .expect("retained original step");
        step_owner.adjoints.extend(output_views);
        step_owner
            .policy_vjp_workspaces
            .push(Arc::clone(&vjp_workspace));
        record_policy_vjp(
            &self.domain,
            &mut self.poisoned,
            recorder,
            execute,
            descriptor,
            &vjp_workspace.coefficients,
            &vjp_workspace.score_cotangents,
        )?;
        if let Err(error) = self.order_content_consumers(consumer_stream) {
            self.poisoned = true;
            return Err(error);
        }
        self.policy_tapes.remove(tape_index);
        Ok(SemanticPolicyGradients {
            binding,
            invocation,
            layout,
            parameters,
            text_logits,
            provider: Arc::clone(&self.provider),
            step_aliases,
            publication,
            continuation: original,
            vjp_workspace,
            support_cells,
        })
    }

    // Only construction and fresh native restoration may initialize these
    // private immutable allocations. Proposal tapes keep views of the same bytes.
    fn upload_cold_codebooks(&mut self) -> Result<(), SemanticTransitionError> {
        let upload = (|| {
            self.provider
                .htod_sync_copy_into_tracked(&self.codebooks.words, &mut self.device_codebooks)?;
            self.provider.htod_sync_copy_into_tracked(
                &self.codebooks.components,
                &mut self.device_components,
            )
        })();
        upload.map_err(|error| {
            self.poisoned = true;
            runtime_error("cold codebook upload", error)
        })
    }

    fn upload_rng_state(
        &mut self,
        binding: SemanticCatalogueBinding,
        rng: SemanticRngBinding,
    ) -> Result<(), SemanticTransitionError> {
        let state = DeviceState {
            model_generation: rng.model_generation,
            family_id: u32::from(rng.family_id),
            stream_serial: rng.stream_serial,
            next_proposal: u64::from(rng.proposal),
            catalogue_generation: binding.generation,
            catalogue_digest: Identity256::from_bytes(CATALOGUE_DIGEST),
            binding_digest: binding.digest,
            ..DeviceState::default()
        };
        let upload = self
            .provider
            .htod_sync_copy_into_tracked(&[state], &mut self.state);
        upload.map_err(|e| {
            self.poisoned = true;
            runtime_error("RNG state upload", e)
        })
    }

    pub fn capture(&mut self) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some()
            && (self.continuation_base.is_none()
                || self
                    .admitted_transition
                    .is_none_or(|(_, word, _)| Some(word) != self.continuation_base))
        {
            return Err(publication_input_error(
                "publication capture requires the admitted acquired-parent continuation",
            ));
        }
        if self.rng.is_none() {
            return Err(SemanticTransitionError::NotBound);
        }
        self.validate_ranges()?;
        let descriptor = self.descriptor();
        let execute = self.execute.clone();
        let recorder = self.kernel_recorder();
        self.graph.enter_transition();
        // Remain poisoned through EndCapture and instantiation, including unwind.
        // Admission inside capture transfers the actual owners to the graph.
        self.poisoned = true;
        let captured =
            CapturedCudaGraph::capture_on_stream_retaining(&self.stream, Vec::new(), || {
                // SAFETY: exact ABI, recorded disjoint allocations and bound stream.
                let enqueued = unsafe {
                    self.domain.enqueue(recorder, |enqueue| {
                        execute
                            .launch_in(
                                enqueue,
                                LaunchConfig {
                                    grid_dim: (1, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (descriptor,),
                            )
                            .map_err(|error| XlogError::Kernel(error.to_string()))
                    })
                }
                .map_err(LaunchEnqueueError::into_xlog_error)?;
                enqueued.commit().map_err(|error| {
                    XlogError::Kernel(format!("semantic transition capture commit: {error}"))
                })
            })
            .map_err(|error| runtime_error("capture", error))?;
        self.captured = Some(captured);
        self.poisoned = false;
        Ok(())
    }

    /// Nonblocking replay. All 136 distributions and RNG blocks execute on device.
    pub fn launch(&mut self) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if let Some(binding) = self.text_binding.as_ref().map(Arc::clone) {
            self.guard_continuation_content(&binding)?;
        }
        let graph = self
            .captured
            .as_ref()
            .ok_or(SemanticTransitionError::NotCaptured)?;
        if self.next_proposal > u64::from(u32::MAX) {
            return Err(SemanticTransitionError::GenerationExhausted);
        }
        let recorder = self.kernel_recorder();
        // SAFETY: every captured allocation is registered, owned, and cannot be
        // rebound or observed until terminal completion. Graph has no host nodes.
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |stream| {
            graph.launch_in(stream)
        })?;
        self.pending = true;
        #[cfg(feature = "semantic-policy")]
        if let Some(policy) = &mut self.policy {
            policy.tape_live = true;
        }
        Ok(())
    }

    /// Waits for completion, copies the typed bank into pinned memory, then observes
    /// the actual immutable hypergraph root. Unreconciled failures poison reuse;
    /// verified ordinary refusals retain the original invocation for finalization.
    pub fn observe(
        &mut self,
        expected_proposal: u32,
    ) -> Result<SemanticTransitionOutcome, SemanticTransitionError> {
        if self.is_poisoned() {
            return Err(SemanticTransitionError::Poisoned);
        }
        if !self.pending {
            return Err(SemanticTransitionError::NoPendingLaunch);
        }
        let result = self.observe_terminal(expected_proposal);
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.pending = false;
            #[cfg(feature = "semantic-policy")]
            self.retain_policy_tape(match &result {
                Ok(SemanticTransitionOutcome::Refused(refusal)) => Some(*refusal),
                _ => None,
            });
            self.task_observed = self.task.is_some() && self.publication.is_none();
            if self.publication.is_some() {
                self.continuation_base = None;
                self.admitted_transition = None;
                self.text_binding = None;
            }
        }
        result
    }

    #[cfg(feature = "semantic-policy")]
    fn retain_policy_tape(&mut self, refusal: Option<SemanticTransitionRefusal>) {
        let Some(mut policy) = self.policy.take() else {
            return;
        };
        let replacement = policy
            .replacement
            .take()
            .expect("proposal reserved its next working banks");
        let invocation = self
            .rng
            .take()
            .expect("observed policy retains its original invocation");
        self.captured = None;
        self.policy_tapes.push(PolicyTape {
            invocation,
            refusal,
            binding: self.binding(),
            policy,
            support: std::mem::replace(&mut self.support, replacement.support),
            receipts: std::mem::replace(&mut self.receipts, replacement.receipts),
            components: self.device_components.view(),
            codebooks: self.device_codebooks.view(),
        });
    }

    fn observe_terminal(
        &mut self,
        expected_proposal: u32,
    ) -> Result<SemanticTransitionOutcome, SemanticTransitionError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)
            .map_err(|error| runtime_error("terminal observation stream admission", error))?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "terminal wait",
            CudaStream::synchronize,
        )?;
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(&self.state);
        recorder.read(&self.receipts);
        // SAFETY: pinned destination and tracked sources live through the second
        // wait, including cleanup on an enqueue/commit error.
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |stream| {
            self.pinned.enqueue_copy(
                self.state.device_ptr_value(),
                self.receipts.device_ptr_value(),
                stream.stream(),
            )
        })?;
        wait_on_stream(
            &self.stream,
            &mut self.poisoned,
            &mut self.stream_waits,
            "observation wait",
            CudaStream::synchronize,
        )?;
        // SAFETY: both transfers and the final stream wait succeeded.
        let (state, components) = unsafe { self.pinned.read()? };
        self.provider
            .record_final_observation_transfer(size_of::<DeviceState>() as u64);
        self.provider.record_final_observation_transfer(
            (COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>()) as u64,
        );
        if matches!(state.status, 2 | 5 | 11 | 12..=14) && self.publication.is_some() {
            let refusal = self.reconcile_refusal(&state, expected_proposal)?;
            return Ok(SemanticTransitionOutcome::Refused(refusal));
        }
        if state.status == 2 {
            return Err(SemanticTransitionError::InvalidFinalSupport {
                completed_draws: state.blocks,
            });
        }
        if state.status == 5 {
            return Err(SemanticTransitionError::NonFinitePolicyInput {
                completed_draws: state.blocks,
            });
        }
        if state.status == 11 {
            return Err(SemanticTransitionError::WorkCounterOverflow {
                completed_draws: state.blocks,
            });
        }
        if matches!(state.status, 12..=14) {
            return Err(SemanticTransitionError::InvalidTrainingView {
                status: state.status - 11,
            });
        }
        match state.status {
            7 => return Err(SemanticTransitionError::InvalidTaskBank),
            8 => return Err(SemanticTransitionError::InvalidTaskBase),
            9 => return Err(SemanticTransitionError::TaskOwnerFailure),
            _ => {}
        }
        let published = if self.publication.is_some() {
            Some(self.reconcile_publication(&state, expected_proposal)?)
        } else {
            None
        };
        let rng = self
            .rng
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if expected_proposal != rng.proposal || u64::from(expected_proposal) != self.next_proposal {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let policy_text = false;
        #[cfg(feature = "semantic-policy")]
        let policy_text = self.policy.is_some() || policy_text;
        self.finish_observed_transition(state, components, rng, published, policy_text)
    }

    fn finish_observed_transition(
        &mut self,
        state: DeviceState,
        components: Vec<SemanticTransitionReceipt>,
        rng: SemanticRngBinding,
        published: Option<(
            PublicationHeader,
            (SemanticRootHandle, SemanticRootSnapshot),
            SemanticTransitionKind,
        )>,
        policy_text: bool,
    ) -> Result<SemanticTransitionOutcome, SemanticTransitionError> {
        let proposal = u64::from(rng.proposal);
        if state.model_generation != rng.model_generation
            || state.family_id != u32::from(rng.family_id)
            || state.stream_serial != rng.stream_serial
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some((header, (root, snapshot), kind)) = published {
            if kind != SemanticTransitionKind::Proposal {
                if state.status != 0
                    || state.blocks != 0
                    || state.proposal != proposal
                    || state.next_proposal != proposal
                    || state.importance_weight != 1.0
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                self.root = root;
                self.base_snapshot = snapshot;
                self.next_proposal = header.proposal;
                return Ok(SemanticTransitionOutcome::Published(
                    SemanticTransitionObservation {
                        components: Vec::new(),
                        root_handle: root,
                        root: snapshot,
                        next_proposal: header.proposal,
                        lanes: std::array::from_fn(|_| SemanticTransitionLane {
                            root: None,
                            task_refusal: None,
                            edits: [Ok(None), Ok(None)],
                            work: SemanticTransitionWork::default(),
                        }),
                        importance_weight: state.importance_weight,
                        task_evaluation: None,
                    },
                ));
            }
        }
        if state.semantic_receipts[0][0] != 0 {
            self.graph
                .observe_transition_root(state.semantic_receipts[0])
                .map_err(SemanticTransitionError::Semantic)?;
        }
        let binding = SemanticActionCatalogue::current().binding();
        if state.status != 0
            || state.proposal != proposal
            || state.next_proposal != proposal + 1
            || components.len() != COMPONENT_COUNT
            || state.blocks != COMPONENT_COUNT as u64
            || state.catalogue_generation != binding.generation
            || state.catalogue_digest != binding.digest
            || state.binding_digest != self.binding().digest
            || state.work.iter().any(|work| !work.is_valid())
            || (published.is_none() && state.retired_roots_mask != 0)
            || state.retired_roots_mask & !3 != 0
            || !state.importance_weight.is_finite()
            || state.importance_weight <= 0.0
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (row, component) in components.iter().zip(&self.codebooks.components) {
            let inactive_text = policy_text
                && component.kind == COMPONENT_KIND_TEXT as u32
                && canonical_text_null(row);
            if row.ordinal != component.ordinal
                || row.lane != component.lane
                || row.slot != component.slot
                || row.field != component.field
                || row.kind != component.kind
                || row.proposal != state.proposal
                || row.catalogue_generation != binding.generation
                || row.catalogue_digest != binding.digest
                || row.admission_binding != self.binding().digest
                || if inactive_text {
                    !canonical_text_null(row)
                } else {
                    row.choice >= component.cardinality
                }
                || row.cdf_start > row.draw
                || row.draw >= row.cdf_end
                || row.cdf_end > 1u64 << 63
                || row.cdf_end - row.cdf_start != row.mass
                || row.mass < 1u64 << 32
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        }
        let mut integrity_error = None;
        let task_evaluation = if let Some((binding, _)) = &self.task {
            let task = &state.task_evaluation;
            if task.query_count < 3
                || task.query_count > 9
                || !task.query_count.is_multiple_of(3)
                || task.winner > 2
                || task.facts[task.winner as usize].eligible != 1
                || task.lane_refusal.iter().any(|&code| code > 2)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            Some(SemanticTaskEvaluation {
                binding: binding.identity(),
                query_count: task.query_count,
                winner: task.winner,
                return_value: task.return_value,
                facts: task.facts,
            })
        } else {
            None
        };
        let lanes = std::array::from_fn(|i| {
            let task_refusal = if task_evaluation.is_some() {
                match state.task_evaluation.lane_refusal[i] {
                    1 => Some(SemanticTaskRefusal::Scope),
                    2 => Some(SemanticTaskRefusal::HardConstraint),
                    _ => None,
                }
            } else {
                None
            };
            let root =
                (task_refusal.is_none() && state.retired_roots_mask & (1 << i) == 0).then(|| {
                    self.graph
                        .observe_transition_root(state.semantic_receipts[3 + i * 3])
                });
            if self.graph.ensure_not_poisoned().is_err() {
                if let Some(Err(error)) = &root {
                    integrity_error.get_or_insert_with(|| error.clone());
                }
            }
            let edits = std::array::from_fn(|j| {
                let result = self
                    .graph
                    .observe_transition_edit(state.semantic_receipts[1 + i * 3 + j]);
                if self.graph.ensure_not_poisoned().is_err() {
                    if let Err(error) = &result {
                        integrity_error.get_or_insert_with(|| error.clone());
                    }
                }
                result
            });
            SemanticTransitionLane {
                root,
                task_refusal,
                edits,
                work: state.work[i],
            }
        });
        if let Some(error) = integrity_error {
            return Err(SemanticTransitionError::Semantic(error));
        }
        if let Some((_, (root, snapshot), _)) = published {
            self.root = root;
            self.base_snapshot = snapshot;
        }
        self.next_proposal = state.next_proposal;
        Ok(SemanticTransitionOutcome::Published(
            SemanticTransitionObservation {
                components,
                root_handle: self.root,
                root: self.base_snapshot,
                next_proposal: state.next_proposal,
                lanes,
                importance_weight: state.importance_weight,
                task_evaluation,
            },
        ))
    }

    fn reconcile_refusal(
        &mut self,
        state: &DeviceState,
        expected_proposal: u32,
    ) -> Result<SemanticTransitionRefusal, SemanticTransitionError> {
        let rng = self
            .rng
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if rng.proposal != expected_proposal || u64::from(expected_proposal) != self.next_proposal {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let refusal = refused_terminal(
            state,
            rng,
            SemanticActionCatalogue::current().binding(),
            self.binding().digest,
        )?;
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let (token, base_word, _) = self
            .admitted_transition
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let original = Arc::clone(
            self.text_binding
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        let reader = self
            .readers
            .get(&token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let step = self
            .steps
            .get(&token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if self.continuation_base != Some(base_word)
            || original._witness.reader_token != token
            || original.published_parent()?.0 != (base_word & 1) as usize
            || !Arc::ptr_eq(&original._witness.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&original._witness._witness, &step.content_witnesses)
            || step
                .content
                .get(original._witness.index)
                .is_none_or(|content| !matches!(content.seals, TensorContentSeals::Captured(_)))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let lease_view = reader.device.view();
        let lease = self.publication_read(lease_view)?[0];
        let control = self.publication_read(storage.control.view())?[0];
        if lease.abi != 1
            || lease.status != 0
            || lease.active != 1
            || lease.instance != storage.instance
            || lease.word != base_word
            || lease.bank != base_word & 1
            || lease.epoch != base_word >> 1
            || control.reader_counts[lease.bank as usize] == 0
            || control.abi != 1
            || control.instance != storage.instance
            || control.word != base_word
            || control.reader_gate != 0
            || control.refusal != 0
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let (bank, parent_header) = original.published_parent()?;
        let header = self.read_publication_header(bank)?;
        if header != parent_header
            || header.abi != 1
            || header.instance != storage.instance
            || header.publication_word != base_word
            || header.sealed_epoch != base_word >> 1
            || header.proposal != state.proposal
            || header.model_generation != u64::from(state.model_generation)
            || header.family_id != u64::from(state.family_id)
            || header.stream_serial != state.stream_serial
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        // The kernel snapshots this exact protected root after ordinary cleanup,
        // including the pre-staging numerical refusal. No success receipt is invented.
        let (root, snapshot) = self.observe_publication_root(state, &header, 0)?;
        if root != self.root || snapshot != self.base_snapshot {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        refused_root_cleanup(
            state,
            self.graph.transition_arena()[1],
            u64::from(root.slot()),
            root.generation(),
        )?;
        Ok(refusal)
    }

    fn observe_publication_root(
        &mut self,
        state: &DeviceState,
        header: &PublicationHeader,
        index: usize,
    ) -> Result<(SemanticRootHandle, SemanticRootSnapshot), SemanticTransitionError> {
        let (root, snapshot) = self
            .graph
            .observe_transition_root(state.semantic_receipts[index])
            .map_err(SemanticTransitionError::Semantic)?;
        let extents = snapshot.extents();
        if header.semantic_owner != self.graph.transition_arena()[1]
            || header.semantic_slot != u64::from(root.slot())
            || header.semantic_generation != root.generation()
            || header.semantic_digest.as_bytes() != snapshot.digest().as_bytes()
            || header.semantic_extents
                != [
                    u64::from(extents.statements()),
                    u64::from(extents.supports()),
                    u64::from(extents.versions()),
                ]
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok((root, snapshot))
    }

    fn reconcile_publication(
        &mut self,
        state: &DeviceState,
        expected_proposal: u32,
    ) -> Result<
        (
            PublicationHeader,
            (SemanticRootHandle, SemanticRootSnapshot),
            SemanticTransitionKind,
        ),
        SemanticTransitionError,
    > {
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let (_, base_word, kind) = self
            .admitted_transition
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let control = self.publication_read(storage.control.view())?[0];
        if control.refusal != 0 || state.status == 10 {
            return Err(SemanticTransitionError::PublicationRefused {
                status: control.refusal,
            });
        }
        let epoch = (base_word >> 1)
            .checked_add(1)
            .filter(|&epoch| epoch <= u64::MAX >> 1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let expected_word = (epoch << 1) | ((base_word ^ 1) & 1);
        if state.status != 0
            || control.abi != 1
            || control.instance != storage.instance
            || control.word != expected_word
            || self.continuation_base != Some(base_word)
            || u64::from(expected_proposal) != self.next_proposal
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let header = self.read_publication_header((expected_word & 1) as usize)?;
        let next_proposal = self
            .next_proposal
            .checked_add(u64::from(kind == SemanticTransitionKind::Proposal))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if header.abi != 1
            || header.instance != control.instance
            || header.base_word != base_word
            || header.publication_word != expected_word
            || header.sealed_epoch != epoch
            || header.proposal != next_proposal
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let winner = if kind == SemanticTransitionKind::Proposal {
            state.task_evaluation.winner
        } else {
            0
        };
        let index = match winner {
            0 => 0,
            1 => 3,
            2 => 6,
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        if winner > 0 && state.retired_roots_mask & (1 << (winner - 1)) != 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let root = self.observe_publication_root(state, &header, index)?;
        Ok((header, root, kind))
    }

    fn ensure_quiescent(&self) -> Result<(), SemanticTransitionError> {
        if self.is_poisoned() {
            Err(SemanticTransitionError::Poisoned)
        } else if self.pending {
            Err(SemanticTransitionError::OverlappingLaunch)
        } else {
            Ok(())
        }
    }

    fn ensure_rebindable(&self) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        validate_rebinding_ownership(
            self.prepared_segment.as_ref(),
            self.steps.values().any(|step| step.identity.is_none()),
        )?;
        if self.task_observed {
            return Err(SemanticTransitionError::UnpublishedTask);
        }
        #[cfg(feature = "semantic-policy")]
        if self.policy.as_ref().is_some_and(|policy| policy.tape_live) {
            return Err(SemanticTransitionError::UnconsumedPolicyTape);
        }
        Ok(())
    }

    fn transition_kernel_io(&self) -> TransitionKernelIo<'_> {
        let logits = &self.logits;
        let text = self.text_binding.as_deref();
        #[cfg(feature = "semantic-policy")]
        let (logits, text) = self.policy.as_ref().map_or((logits, text), |policy| {
            (&policy.text_logits, Some(policy.text_binding.as_ref()))
        });
        TransitionKernelIo {
            logits,
            support: &self.support,
            receipts: &self.receipts,
            state: &self.state,
            text,
            model_work: None,
            model_update: None,
            lease: self
                .admitted_transition
                .map(|(token, _, _)| &self.readers[&token].device),
            #[cfg(feature = "semantic-policy")]
            policy: self.policy.as_ref(),
            training_selection: None,
        }
    }

    #[cfg(feature = "semantic-policy")]
    fn prepared_kernel_io(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<TransitionKernelIo<'_>, SemanticTransitionError> {
        let prepared = self
            .steps
            .get(&step.token)
            .and_then(|owner| owner.prepared.as_ref())
            .ok_or_else(|| {
                publication_input_error("prepared transition has no original storage")
            })?;
        let policy = prepared.policy.as_ref();
        let text = &prepared
            .continuation
            .as_ref()
            .ok_or_else(|| {
                publication_input_error("prepared transition has no original continuation")
            })?
            .text_binding;
        let logits = match policy {
            Some(policy) => &policy.text_logits,
            None => {
                &prepared
                    .policy_buffers
                    .as_ref()
                    .ok_or_else(|| {
                        publication_input_error("prepared transition has no retained cold buffers")
                    })?
                    .text_logits
            }
        };
        Ok(TransitionKernelIo {
            // Nonproposal execution returns before reading logits or support.
            // Retain the original cold allocation; do not invent policy values.
            logits,
            support: prepared
                .support
                .as_ref()
                .ok_or_else(|| publication_input_error("prepared support has retired"))?,
            receipts: prepared
                .receipts
                .as_ref()
                .ok_or_else(|| publication_input_error("prepared receipts have retired"))?,
            state: &prepared.state,
            text: Some(text),
            lease: Some(&prepared.reader),
            policy,
            model_work: prepared.model_work.as_ref(),
            model_update: prepared.model_update.as_ref(),
            training_selection: prepared
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.training_selection.as_ref()),
        })
    }

    #[cfg(feature = "semantic-policy")]
    fn prepared_drain_kernel_io(
        &self,
        step: &SemanticPreparedStep,
    ) -> Result<TransitionKernelIo<'_>, SemanticTransitionError> {
        let prepared = self
            .steps
            .get(&step.token)
            .and_then(|owner| owner.prepared.as_ref())
            .ok_or_else(|| publication_input_error("prepared drain has no original storage"))?;
        let logits = if let Some(policy) = prepared.policy.as_ref() {
            &policy.text_logits
        } else {
            &prepared
                .policy_buffers
                .as_ref()
                .ok_or_else(|| {
                    publication_input_error("prepared drain has no retained cold buffers")
                })?
                .text_logits
        };
        Ok(TransitionKernelIo {
            logits,
            support: prepared
                .support
                .as_ref()
                .ok_or_else(|| publication_input_error("prepared support has retired"))?,
            receipts: prepared
                .receipts
                .as_ref()
                .ok_or_else(|| publication_input_error("prepared receipts have retired"))?,
            state: &prepared.state,
            text: None,
            lease: Some(&prepared.reader),
            policy: None,
            model_work: None,
            model_update: None,
            training_selection: None,
        })
    }

    fn descriptor(&self) -> Descriptor {
        self.descriptor_with(&self.transition_kernel_io())
    }

    fn descriptor_with(&self, io: &TransitionKernelIo<'_>) -> Descriptor {
        #[cfg(not(feature = "semantic-policy"))]
        let policy = PolicyDescriptor::default();
        #[cfg(feature = "semantic-policy")]
        let policy = io.policy.map_or(
            PolicyDescriptor::default(),
            Self::retained_policy_descriptor,
        );
        Descriptor {
            logits: 0,
            support: 0,
            scratch: self.scratch.device_ptr_value(),
            receipts: 0,
            state: 0,
            components: self.device_components.device_ptr_value(),
            codebooks: self.device_codebooks.device_ptr_value(),
            arena: self.graph.transition_arena(),
            policy: PolicyDescriptor::default(),
            backward: PolicyBackward::default(),
            task: self
                .task
                .as_ref()
                .map_or(0, |(_, device)| device.device_ptr_value()),
            publication: PublicationCommand {
                control: self
                    .publication
                    .as_ref()
                    .map_or(0, |storage| storage.control.device_ptr_value()),
                lease: 0,
                operation: 0,
            },
            text: TextBinding::default(),
            model_work: ModelWorkInput::default(),
        }
        .with_transition(TransitionKernelPointers {
            logits: io.logits.device_ptr_value(),
            support: io.support.device_ptr_value(),
            receipts: io.receipts.device_ptr_value(),
            state: io.state.device_ptr_value(),
            lease: io.lease.map_or(0, TrackedCudaSlice::device_ptr_value),
            text: io
                .text
                .map_or(TextBinding::default(), TextBindingStorage::descriptor),
            policy,
            model_work: io
                .model_work
                .map_or(ModelWorkInput::default(), PreparedModelWork::descriptor),
        })
    }

    #[cfg(feature = "semantic-policy")]
    fn retained_policy_descriptor(policy: &PolicyStorage) -> PolicyDescriptor {
        let parameter = |offset: usize| policy.parameters.device_ptr_value() + (offset * 4) as u64;
        PolicyDescriptor {
            z: parameter(policy.layout.z.start),
            recurrence: parameter(policy.layout.recurrence.start),
            positions: parameter(policy.layout.positions.start),
            hidden: policy.hidden.device_ptr_value(),
            scores: policy.scores.device_ptr_value(),
            recurrent: policy.recurrent.device_ptr_value(),
            fields: std::array::from_fn(|i| {
                let field = &policy.layout.fields[i];
                PolicyField {
                    embeddings: if field.embeddings.is_empty() {
                        0
                    } else {
                        parameter(field.embeddings.start)
                    },
                    biases: if field.biases.is_empty() {
                        0
                    } else {
                        parameter(field.biases.start)
                    },
                    cardinality: field.cardinality as u32,
                    null_category: field.null_category.map_or(-1, |index| index as i32),
                }
            }),
        }
    }

    fn validate_ranges(&self) -> Result<(), SemanticTransitionError> {
        self.validate_ranges_with(&self.transition_kernel_io())
    }

    fn validate_ranges_with(
        &self,
        io: &TransitionKernelIo<'_>,
    ) -> Result<(), SemanticTransitionError> {
        let d = self.descriptor_with(io);
        let mut ranges = vec![
            (d.logits, (io.logits.len() * 4) as u64),
            (d.support, io.support.len() as u64),
            (d.scratch, SCRATCH_BYTES as u64),
            (
                d.receipts,
                (io.receipts.len() * size_of::<SemanticTransitionReceipt>()) as u64,
            ),
            (d.state, size_of::<DeviceState>() as u64),
            (
                d.components,
                (COMPONENT_COUNT * size_of::<SemanticComponent>()) as u64,
            ),
            (d.codebooks, (self.codebooks.words.len() * 8) as u64),
            (d.arena[0], d.arena[6] * 8),
        ];
        if let Some((_, device)) = &self.task {
            ranges.push((device.device_ptr_value(), (device.len() * 8) as u64));
        }
        #[cfg(feature = "semantic-policy")]
        {
            if let Some(policy) = io.policy {
                for buffer in [
                    &policy.parameters,
                    &policy.hidden,
                    &policy.scores,
                    &policy.recurrent,
                ] {
                    if !buffer.is_empty() {
                        ranges.push((buffer.device_ptr_value(), (buffer.len() * 4) as u64));
                    }
                }
            }
        }
        if let Some(text) = io.text {
            for input in &text.inputs {
                if let Some(source) = &input.source {
                    ranges.push((input.data, source.len() as u64));
                }
            }
        }
        if let Some(selection) = io.training_selection {
            ranges.push((
                *selection.device_ptr(),
                size_of::<SemanticTrainingViewSelection>() as u64,
            ));
        }
        if let Some(update) = io.model_update {
            ranges.push((
                update.bindings.device_ptr_value(),
                (update.bindings.len() * size_of::<ModelUpdateBinding>()) as u64,
            ));
            if let Some(output) = &update.output {
                for allocation in &output.allocations {
                    if let Some(source) = &allocation.source {
                        ranges.push((allocation.data, source.len() as u64));
                    }
                }
            }
        }
        disjoint_ranges(&ranges)
    }

    fn kernel_recorder(&self) -> LaunchRecorder {
        self.kernel_recorder_with(&self.transition_kernel_io())
    }

    fn kernel_recorder_with(&self, io: &TransitionKernelIo<'_>) -> LaunchRecorder {
        let mut recorder = self.domain.new_strict_recorder();
        recorder.read(io.logits);
        recorder.read(io.support);
        if let Some(work) = io.model_work {
            recorder.read(&work.device);
            recorder.read(&work.actual);
        }
        if let Some(update) = io.model_update {
            update.record(&mut recorder);
        }
        recorder.write(&self.scratch);
        recorder.write(io.receipts);
        recorder.read_write(io.state);
        recorder.read(&self.device_codebooks);
        recorder.read(&self.device_components);
        if let Some((_, device)) = &self.task {
            recorder.read(device);
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(policy) = io.policy {
            recorder.read(&policy.parameters);
            recorder.write(&policy.hidden);
            recorder.write(&policy.scores);
            recorder.write(&policy.recurrent);
        }
        self.graph.record_transition(&mut recorder);
        if let Some(publication) = &self.publication {
            publication.record(&mut recorder);
        }
        if let Some(lease) = io.lease {
            recorder.read(lease);
        }
        if let Some(text) = io.text {
            text.record(&mut recorder);
        }
        if let Some(selection) = io.training_selection {
            recorder.read(selection);
        }
        recorder
    }
}

fn enqueue_recorded<E: fmt::Display>(
    domain: &ResidentExecutionDomain,
    poisoned: &mut bool,
    recorder: LaunchRecorder,
    operation: impl FnOnce(&CudaEnqueue<'_>) -> Result<(), E>,
) -> Result<(), SemanticTransitionError> {
    // SAFETY: all call sites register exactly the allocations used on the supplied
    // stream. Poison is armed before the may-enqueue boundary, including unwinding;
    // only a successful consuming commit permits reuse.
    let result = unsafe {
        domain.enqueue(recorder, |stream| {
            *poisoned = true;
            operation(stream)
        })
    };
    let enqueued = result.map_err(|e| {
        if matches!(
            &e,
            LaunchEnqueueError::Operation(_) | LaunchEnqueueError::OperationAndCleanup { .. }
        ) {
            *poisoned = true;
        }
        runtime_error("enqueue", e)
    })?;
    enqueued
        .commit()
        .map_err(|e| runtime_error("enqueue commit", e))?;
    *poisoned = false;
    Ok(())
}

fn wait_on_stream<E: fmt::Display>(
    stream: &CudaStream,
    poisoned: &mut bool,
    waits: &mut u64,
    operation: &'static str,
    wait: impl FnOnce(&CudaStream) -> Result<(), E>,
) -> Result<(), SemanticTransitionError> {
    *waits += 1;
    wait(stream).map_err(|e| {
        *poisoned = true;
        runtime_error(operation, e)
    })
}

fn validate_bound_inputs(
    components: &[SemanticComponent],
    rng: SemanticRngBinding,
    inputs: &[SemanticComponentInput],
) -> Result<(), SemanticTransitionError> {
    let invalid = |detail: &str| SemanticTransitionError::InvalidInput {
        detail: detail.into(),
    };
    if rng.stream_serial >= 1u64 << 56 {
        return Err(invalid("stream serial exceeds 56 bits"));
    }
    if inputs.len() != COMPONENT_COUNT {
        return Err(invalid("input roster must contain exactly 136 components"));
    }
    for (component, input) in components.iter().zip(inputs) {
        if input.logits.len() != component.cardinality as usize
            || input.product_support.len() != input.logits.len()
        {
            return Err(invalid("logits/support shape differs from the catalogue"));
        }
        if input.product_support.iter().any(|&m| m > 1) {
            return Err(invalid("product support must be binary"));
        }
        debug_assert!(component.is_text() || component.kind == COMPONENT_KIND_EDIT as u32);
        let has_support = input.product_support.contains(&1);
        if !has_support {
            return Err(invalid("empty product support"));
        }
        if input
            .logits
            .iter()
            .zip(&input.product_support)
            .any(|(&z, &m)| m == 1 && !z.is_finite())
        {
            return Err(invalid("legal logits must be finite FP32"));
        }
    }
    Ok(())
}

fn disjoint_ranges(ranges: &[(u64, u64)]) -> Result<(), SemanticTransitionError> {
    for (i, &(start, len)) in ranges.iter().enumerate() {
        let end = start
            .checked_add(len)
            .filter(|_| start != 0 && len != 0)
            .ok_or(SemanticTransitionError::OverlappingInputs)?;
        for &(other, other_len) in &ranges[..i] {
            let other_end = other
                .checked_add(other_len)
                .ok_or(SemanticTransitionError::OverlappingInputs)?;
            if start < other_end && other < end {
                return Err(SemanticTransitionError::OverlappingInputs);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn raw_prefix_export_retains_the_original_input_seal() {
        let range = PublicationRange {
            role: 1,
            index: 0,
            generation: 1,
            length_bytes: 64,
            logical_begin: 0,
            logical_end: 1,
            ..PublicationRange::default()
        };
        let input = StepInputPlan {
            role: 1,
            index: 0,
            layout: committed_prefix_layout(&range, 4).unwrap(),
            banks: [
                StepInputBankPlan {
                    storage_slot: 0,
                    span: 0..256,
                },
                StepInputBankPlan {
                    storage_slot: 0,
                    span: 0..256,
                },
            ],
        };
        let raw = publication_record_layout(&range);
        let exported = publication_export_span(&range, &raw, 256).unwrap();
        assert_eq!(exported, 0..64);
        assert!(step_input_alias_matches(
            &input,
            &range,
            4096,
            256,
            (raw, 0, 0, 4096, exported.len())
        ));
        assert!(!step_input_alias_matches(
            &input,
            &range,
            4096,
            256,
            (raw, 0, 0, 4097, exported.len())
        ));
        assert!(!step_input_alias_matches(
            &input,
            &range,
            4096,
            256,
            (raw, 0, 1, 4096, exported.len())
        ));
        assert!(!step_input_alias_matches(
            &input,
            &range,
            4096,
            256,
            (raw, 0, 0, 4096, 128)
        ));
        assert!(step_input_alias_matches(
            &input,
            &range,
            4096,
            256,
            (input.layout, 0, 1, 4096, 256)
        ));
        let empty = PublicationRange {
            length_bytes: 0,
            logical_end: 0,
            ..range
        };
        let raw_empty = publication_record_layout(&empty);
        assert_eq!(
            publication_export_span(&empty, &raw_empty, 256).unwrap(),
            0..0
        );
        assert!(step_input_alias_matches(
            &input,
            &empty,
            4096,
            256,
            (raw_empty, 0, 0, 4096, 0)
        ));
    }

    #[test]
    fn transition_descriptors_preserve_each_original_policy_and_outcome() {
        use super::*;
        let shared = Descriptor {
            logits: 0,
            support: 0,
            scratch: 8,
            receipts: 0,
            state: 0,
            components: 16,
            codebooks: 24,
            arena: [32; 7],
            policy: PolicyDescriptor::default(),
            backward: PolicyBackward::default(),
            task: 40,
            publication: PublicationCommand {
                control: 48,
                lease: 0,
                operation: 0,
            },
            text: TextBinding::default(),
            model_work: ModelWorkInput::default(),
        };
        let original = |base| TransitionKernelPointers {
            logits: base,
            support: base + 8,
            receipts: base + 16,
            state: base + 24,
            lease: base + 32,
            text: TextBinding {
                rows: base + 40,
                count: base + 48,
                selected: base + 56,
            },
            policy: PolicyDescriptor {
                z: base + 64,
                recurrent: base + 72,
                ..PolicyDescriptor::default()
            },
            model_work: ModelWorkInput {
                events: base + 80,
                count: base + 88,
                bound: base + 96,
            },
        };
        let first = shared.with_transition(original(1024));
        let second = shared.with_transition(original(2048));
        for (descriptor, base) in [(first, 1024), (second, 2048)] {
            assert_eq!(
                (
                    descriptor.logits,
                    descriptor.support,
                    descriptor.receipts,
                    descriptor.state
                ),
                (base, base + 8, base + 16, base + 24)
            );
            assert_eq!(
                (
                    descriptor.policy.z,
                    descriptor.policy.recurrent,
                    descriptor.publication.lease
                ),
                (base + 64, base + 72, base + 32)
            );
            assert_eq!(
                (
                    descriptor.text.rows,
                    descriptor.text.count,
                    descriptor.text.selected
                ),
                (base + 40, base + 48, base + 56)
            );
            assert_eq!(
                (
                    descriptor.model_work.events,
                    descriptor.model_work.count,
                    descriptor.model_work.bound
                ),
                (base + 80, base + 88, base + 96)
            );
            assert_eq!(
                (
                    descriptor.scratch,
                    descriptor.components,
                    descriptor.codebooks,
                    descriptor.task
                ),
                (8, 16, 24, 40)
            );
            assert_eq!(descriptor.publication.control, 48);
        }
        assert_ne!(first.policy.recurrent, second.policy.recurrent);
        assert_ne!(first.receipts, second.receipts);
    }

    #[test]
    fn prepared_schedule_preserves_each_requested_mode_until_recording() {
        let issuer = Arc::new(());
        let schedule = [
            SemanticTransitionKind::Proposal,
            SemanticTransitionKind::Recompute,
            SemanticTransitionKind::Proposal,
        ];
        let mut build =
            PreparedSegmentState::new(Arc::clone(&issuer), vec![4, 5, 6], schedule.into_iter())
                .unwrap();
        let steps = build.handles().unwrap();
        assert!(build.enter(&steps[1], &issuer).is_err());
        for (step, expected) in steps.iter().zip(schedule) {
            assert_eq!(build.requested_kind(step, &issuer).unwrap(), expected);
            assert_eq!(build.enter(step, &issuer).unwrap(), expected);
            build.leave(step, &issuer).unwrap();
        }
        build.finish().unwrap();
        build.submit().unwrap();
        assert!(build.submit().is_err());
        assert!(build.requested_kind(&steps[0], &Arc::new(())).is_err());
        assert!(
            PreparedSegmentState::new(Arc::clone(&issuer), vec![9], schedule.into_iter()).is_err()
        );
        assert!(PreparedSegmentState::new(
            issuer,
            vec![9],
            std::iter::once(SemanticTransitionKind::Drain)
        )
        .is_err());
    }

    #[test]
    fn prepared_step_scope_rejects_foreign_owners_and_out_of_order_recording() {
        let issuer = Arc::new(());
        let mut build = PreparedSegmentState::new(
            Arc::clone(&issuer),
            vec![4, 5],
            [
                SemanticTransitionKind::Proposal,
                SemanticTransitionKind::Proposal,
            ]
            .into_iter(),
        )
        .unwrap();
        let steps = build.handles().unwrap();
        assert!(build.check(&steps[0], &issuer, false).is_ok());
        assert!(build.check(&steps[0], &Arc::new(()), false).is_err());
        assert!(build.enter(&steps[1], &issuer).is_err());
        build.enter(&steps[0], &issuer).unwrap();
        assert!(build.check(&steps[0], &issuer, true).is_ok());
        assert!(build.check(&steps[1], &issuer, true).is_err());
        assert!(build.finish().is_err());
        build.leave(&steps[0], &issuer).unwrap();
        assert!(build.enter(&steps[0], &issuer).is_err());
        build.enter(&steps[1], &issuer).unwrap();
        build.leave(&steps[1], &issuer).unwrap();
        build.finish().unwrap();
        assert!(build.check(&steps[0], &issuer, false).is_err());
        assert!(PreparedSegmentState::new(issuer, Vec::new(), std::iter::empty()).is_err());
    }
    use super::*;
    use xlog_core::MemoryBudget;

    #[test]
    fn publication_role_roster_preserves_every_required_owner() {
        let mut counts = [1; 55];
        counts[15] = 2;
        assert!(validate_publication_counts(&counts).is_ok());
        for code in [1usize, 14, 17, 38, 47, 55] {
            let mut omitted = counts;
            omitted[code - 1] = 0;
            assert!(validate_publication_counts(&omitted).is_err());
        }
        counts[18] = 0;
        counts[19] = 0;
        counts[20] = 0;
        counts[23] = 0;
        counts[24] = 0;
        assert!(validate_publication_counts(&counts).is_ok());
        counts[50] = 2;
        assert!(validate_publication_counts(&counts).is_err());
        assert!(SemanticStateRole::from_code(0).is_none());
        assert!(SemanticStateRole::from_code(56).is_none());
        assert!(SemanticStateRole::from_code(u64::MAX).is_none());
        for code in 1..=55 {
            assert_eq!(SemanticStateRole::from_code(code).unwrap() as u64, code);
        }
    }

    #[test]
    fn prefix_snapshot_copy_respects_owned_capacity_strides() {
        let source = SemanticTensorLayout {
            role: 4,
            index: 0,
            element_bytes: 2,
            scalar_type: 4,
            rank: 4,
            logical_axis: 2,
            dimensions: [1, 2, 3, 4],
            strides_bytes: [48, 24, 8, 2],
        };
        let destination = prefix_capacity_layout(source, 8).unwrap();
        assert_eq!(destination.dimensions, [1, 2, 8, 4]);
        assert_eq!(destination.strides_bytes, [128, 64, 8, 2]);
        assert_eq!(
            tensor_copy_plan(&source, &destination).unwrap(),
            [(0, 0, 24), (24, 64, 24)]
        );
        let mut empty = source;
        empty.dimensions[2] = 0;
        assert!(tensor_copy_plan(&empty, &destination).unwrap().is_empty());
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn policy_score_cotangent_admission_requires_exact_fp64_metadata() {
        use crate::dlpack::{DLDataType, DLDevice, DLTensor, K_DLCUDA, K_DLFLOAT, K_DLINT};

        let mut shape = [COMPONENT_COUNT as i64];
        let mut stride = [1i64];
        // This exercises the production metadata boundary only: the address is
        // never dereferenced and no CUDA provider or managed owner is created.
        let mut tensor = DLTensor {
            data: 4096usize as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: K_DLCUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: K_DLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: the local shape/stride arrays outlive each metadata-only call.
        let admit = |tensor: &DLTensor| unsafe { policy_score_cotangent_metadata(tensor, 0) };
        assert_eq!(
            admit(&tensor).unwrap(),
            (4096, COMPONENT_COUNT * size_of::<f64>())
        );
        tensor.strides = stride.as_mut_ptr();
        tensor.byte_offset = 8;
        assert_eq!(
            admit(&tensor).unwrap(),
            (4104, COMPONENT_COUNT * size_of::<f64>())
        );
        tensor.byte_offset = 0;
        tensor.dtype.code = K_DLINT;
        assert!(
            admit(&tensor).is_err(),
            "Int64 cells must not be bitcast as FP64 cotangents"
        );
        tensor.dtype.code = K_DLFLOAT;
        tensor.dtype.bits = 32;
        assert!(admit(&tensor).is_err());
        tensor.dtype.bits = 64;
        tensor.dtype.lanes = 2;
        assert!(admit(&tensor).is_err());
        tensor.dtype.lanes = 1;
        for extent in [0, COMPONENT_COUNT as i64 - 1, COMPONENT_COUNT as i64 + 1] {
            shape[0] = extent;
            assert!(admit(&tensor).is_err());
        }
        shape[0] = COMPONENT_COUNT as i64;
        tensor.shape = shape.as_mut_ptr();
        stride[0] = 2;
        assert!(matches!(
            admit(&tensor),
            Err(SemanticTransitionError::InvalidInput { .. })
        ));
        stride[0] = 1;
        tensor.strides = stride.as_mut_ptr();
        tensor.ndim = 2;
        assert!(admit(&tensor).is_err());
        tensor.ndim = 1;
        tensor.device.device_id = 1;
        assert!(admit(&tensor).is_err());
        tensor.device.device_id = 0;
        tensor.byte_offset = 4;
        assert!(admit(&tensor).is_err());
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    #[ignore = "requires an explicitly authorized CUDA runtime"]
    fn policy_dlpack_binding_snapshots_actual_producer_allocations() {
        policy_dlpack_binding_fixture(PolicyFixtureCompletion::Backward);
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    #[ignore = "requires an explicitly authorized CUDA runtime"]
    fn policy_inference_finish_retires_published_and_refused_original_owners() {
        for completion in [
            PolicyFixtureCompletion::Inference,
            PolicyFixtureCompletion::NumericalRefusal,
        ] {
            policy_dlpack_binding_fixture(completion);
        }
    }

    #[cfg(feature = "semantic-policy")]
    #[derive(Clone, Copy)]
    enum PolicyFixtureCompletion {
        Backward,
        Inference,
        NumericalRefusal,
    }

    #[cfg(feature = "semantic-policy")]
    fn policy_dlpack_binding_fixture(completion: PolicyFixtureCompletion) {
        use crate::memory::CudaColumn;
        use xlog_core::{ScalarType, Schema};

        let provider = Arc::new(
            crate::CudaProviderBuilder::new(0, MemoryBudget::with_limit(512 * 1024 * 1024))
                .with_stream_capacity(1)
                .build()
                .unwrap(),
        );
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, id, stream)
            .unwrap();
        let mut session =
            super::text_parent_tests::publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(super::text_parent_tests::publication_test_parent(&provider))
            .unwrap();
        let mut parent = session.acquire().unwrap();
        session
            .admit_transition(&parent, SemanticTransitionKind::Proposal)
            .unwrap();
        let (continuation, witness) =
            super::text_parent_tests::publication_test_continuation_with_numerical_admissibility(
                &mut session,
                &parent,
                &provider,
                u8::from(!matches!(
                    completion,
                    PolicyFixtureCompletion::NumericalRefusal
                )),
            );
        session
            .bind_continuation(&parent, continuation, &witness, 1)
            .unwrap();
        let witness = if matches!(completion, PolicyFixtureCompletion::Backward) {
            drop(witness);
            None
        } else {
            Some(witness)
        };
        let layout = session.policy_layout().unwrap();
        let mut logits = provider
            .memory()
            .alloc::<f32>(32 * TEXT_CARDINALITY)
            .unwrap();
        let mut support = provider
            .memory()
            .alloc::<u8>(session.codebooks.input_cells)
            .unwrap();
        let mut parameters = provider
            .memory()
            .alloc::<f32>(layout.parameter_cells)
            .unwrap();
        let mut component_baselines = provider.memory().alloc::<f32>(COMPONENT_COUNT).unwrap();
        provider
            .htod_sync_copy_into_tracked(&vec![2.0; logits.len()], &mut logits)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(&vec![1u8; support.len()], &mut support)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(&vec![3.0; parameters.len()], &mut parameters)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(
                &vec![0.5; component_baselines.len()],
                &mut component_baselines,
            )
            .unwrap();
        let mut original_logit = logits.view().slice(0..1);
        let mut original_support = support.view().slice(0..1);
        let mut original_parameter = parameters.view().slice(0..1);
        let export = |slice: TrackedCudaSlice<u8>, scalar: ScalarType, cells: usize| {
            let buffer = provider
                .buffer_from_columns(
                    vec![CudaColumn::Owned(slice)],
                    cells as u64,
                    Schema::new(vec![("input".into(), scalar)]),
                )
                .unwrap();
            provider.to_dlpack_table(buffer).column(0).unwrap()
        };
        let logits = crate::dlpack::export_slice_managed_tensor(
            Arc::new(logits.into_bytes()),
            provider.device().ordinal() as i32,
            crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            32,
            TEXT_CARDINALITY,
        )
        .unwrap();
        let support = export(support, ScalarType::Bool, session.codebooks.input_cells);
        let parameters = export(
            parameters.into_bytes(),
            ScalarType::F32,
            layout.parameter_cells,
        );
        let component_baselines = crate::dlpack::export_slice_managed_tensor(
            Arc::new(component_baselines.into_bytes()),
            provider.device().ordinal() as i32,
            crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            1,
            COMPONENT_COUNT,
        )
        .unwrap();
        let rng = session.continuation_rng(&parent).unwrap();
        let original_components = session.device_components.device_ptr_value();
        let original_codebooks = session.device_codebooks.device_ptr_value();
        let before_policy_binding = session.host_io_stats();
        session
            .bind_policy_dlpack(
                session.binding(),
                rng,
                logits,
                support,
                parameters,
                component_baselines,
            )
            .unwrap();
        assert_eq!(
            session.host_io_stats(),
            before_policy_binding,
            "policy binding must not upload state or immutable catalogue metadata"
        );
        provider
            .htod_sync_copy_into_tracked(&[9.0], &mut original_logit)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(&[0u8], &mut original_support)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(&[9.0], &mut original_parameter)
            .unwrap();
        let read = |view: crate::memory::DeviceMemoryView<f32>| {
            provider.device().inner().dtoh_sync_copy(&view).unwrap()[0]
        };
        assert_eq!(
            read(
                session
                    .policy
                    .as_ref()
                    .unwrap()
                    .text_logits
                    .view()
                    .slice(0..1)
            ),
            2.0
        );
        assert_eq!(
            read(
                session
                    .policy
                    .as_ref()
                    .unwrap()
                    .parameters
                    .view()
                    .slice(0..1)
            ),
            3.0
        );
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(&session.support.view().slice(0..1))
                .unwrap(),
            [1u8]
        );
        assert_eq!(session.rng, Some(rng));
        // An unconsumed forward retains its real continuation witness.
        if matches!(completion, PolicyFixtureCompletion::Backward) {
            assert!(session.release(&mut parent, &[1]).is_err());
        }
        session.capture().unwrap();
        session.launch().unwrap();
        let outcome = session.observe(0).unwrap();
        assert_eq!(
            session.device_components.device_ptr_value(),
            original_components
        );
        assert_eq!(
            session.device_codebooks.device_ptr_value(),
            original_codebooks
        );
        assert_eq!(
            *session.policy_tapes[0].components.device_ptr(),
            original_components
        );
        assert_eq!(
            *session.policy_tapes[0].codebooks.device_ptr(),
            original_codebooks
        );
        match (&completion, &outcome) {
            (
                PolicyFixtureCompletion::NumericalRefusal,
                SemanticTransitionOutcome::Refused(
                    SemanticTransitionRefusal::NonFinitePolicyInput { completed_draws: 0 },
                ),
            ) => {}
            (
                PolicyFixtureCompletion::Backward | PolicyFixtureCompletion::Inference,
                SemanticTransitionOutcome::Published(_),
            ) => {}
            _ => panic!("policy fixture returned the wrong native outcome: {outcome:?}"),
        }
        let parent_identity = session.retained_step_identity(&parent).unwrap();
        session.release_published_reader(&mut parent, &[1]).unwrap();
        assert!(!parent.active && !session.readers.contains_key(&parent.token));
        assert_eq!(
            session.retained_step_identity(&parent).unwrap(),
            parent_identity
        );
        assert!(session.published_identity(&parent).is_err());
        if !matches!(completion, PolicyFixtureCompletion::Backward) {
            let mut foreign_session =
                super::text_parent_tests::publication_test_session(Arc::clone(&provider), &domain);
            foreign_session
                .bind_parent(super::text_parent_tests::publication_test_parent(&provider))
                .unwrap();
            let mut foreign_parent = foreign_session.acquire().unwrap();
            let parent_identity = session.retained_step_identity(&parent).unwrap();
            let witness_owners = Arc::strong_count(&session.steps[&parent.token].content_witnesses);
            assert!(witness_owners > 1);
            assert_eq!(session.policy_tapes.len(), 1);
            assert_eq!(session.policy_tapes[0].invocation, rng);
            let before_final_use = session.host_io_stats();
            for invalid_stream in [0, 2] {
                assert!(session
                    .finish_policy_invocation(&parent, rng, invalid_stream)
                    .is_err());
                assert_eq!(session.policy_tapes.len(), 1);
                assert_eq!(session.policy_tapes[0].invocation, rng);
                assert_eq!(
                    Arc::strong_count(&session.steps[&parent.token].content_witnesses),
                    witness_owners
                );
            }
            assert!(session
                .finish_policy_invocation(&foreign_parent, rng, 1)
                .is_err());
            assert_eq!(session.policy_tapes.len(), 1);
            assert_eq!(session.policy_tapes[0].invocation, rng);
            assert_eq!(
                Arc::strong_count(&session.steps[&parent.token].content_witnesses),
                witness_owners
            );
            assert_eq!(
                session.retained_step_identity(&parent).unwrap(),
                parent_identity
            );
            session.finish_policy_invocation(&parent, rng, 1).unwrap();
            assert!(session.policy_tapes.is_empty());
            assert!(session.steps[&parent.token].consumer_streams.contains(&1));
            assert_eq!(
                session.host_io_stats(),
                before_final_use,
                "inference final use must use original device guards and stream events only"
            );
            let retained_witness_owners =
                Arc::strong_count(&session.steps[&parent.token].content_witnesses);
            assert!(retained_witness_owners > 1 && !parent.active);
            assert!(session.finish_policy_invocation(&parent, rng, 1).is_err());
            assert_eq!(
                Arc::strong_count(&session.steps[&parent.token].content_witnesses),
                retained_witness_owners
            );
            assert_eq!(
                session.retained_step_identity(&parent).unwrap(),
                parent_identity
            );
            drop(witness);
            drop((original_logit, original_support, original_parameter));
            drop(outcome);
            assert_eq!(
                Arc::strong_count(&session.steps[&parent.token].content_witnesses),
                1
            );
            session.release(&mut parent, &[1]).unwrap();
            assert!(!parent.active && !session.readers.contains_key(&parent.token));
            assert!(!session.is_poisoned());
            foreign_session.release(&mut foreign_parent, &[1]).unwrap();
            return;
        }
        let mut coefficients = provider.memory().alloc::<f64>(COMPONENT_COUNT).unwrap();
        provider
            .htod_sync_copy_into_tracked(&vec![1.0f64; COMPONENT_COUNT], &mut coefficients)
            .unwrap();
        let buffer = provider
            .buffer_from_columns(
                vec![CudaColumn::Owned(coefficients.into_bytes())],
                COMPONENT_COUNT as u64,
                Schema::new(vec![("cotangent".into(), ScalarType::F64)]),
            )
            .unwrap();
        let cotangents = provider.to_dlpack_table(buffer).column(0).unwrap();
        let before_backward = session.host_io_stats();
        let gradients = session.backward_policy_dlpack(rng, cotangents, 1).unwrap();
        let parameter_pointer = gradients.parameters.device_ptr_value();
        let text_pointer = gradients.text_logits.device_ptr_value();
        let [parameters, text] = gradients.into_dlpack().unwrap();
        assert_eq!(
            session.host_io_stats(),
            before_backward,
            "backward and original gradient exports must use device guards and stream events only"
        );
        // SAFETY: both managed owners retain these complete immutable DLPack
        // descriptors and their real backward allocations during inspection.
        unsafe {
            let parameter_tensor = &(*parameters.as_ptr()).dl_tensor;
            let text_tensor = &(*text.as_ptr()).dl_tensor;
            assert_eq!((parameter_tensor.ndim, text_tensor.ndim), (1, 2));
            assert!(
                !parameter_tensor.shape.is_null()
                    && !text_tensor.shape.is_null()
                    && !text_tensor.strides.is_null()
            );
            assert_eq!(
                std::slice::from_raw_parts(parameter_tensor.shape, 1),
                [layout.parameter_cells as i64]
            );
            assert_eq!(
                std::slice::from_raw_parts(text_tensor.shape, 2),
                [32, TEXT_CARDINALITY as i64]
            );
            assert_eq!(
                std::slice::from_raw_parts(text_tensor.strides, 2),
                [TEXT_CARDINALITY as i64, 1]
            );
            assert_eq!(parameter_tensor.data as usize as u64, parameter_pointer);
            assert_eq!(text_tensor.data as usize as u64, text_pointer);
            assert_eq!(
                (
                    text_tensor.dtype.code,
                    text_tensor.dtype.bits,
                    text_tensor.dtype.lanes
                ),
                (2, 32, 1)
            );
            assert_eq!(
                text_tensor.device.device_id,
                provider.device().ordinal() as i32
            );
        }
        assert!(session.steps[&parent.token].consumer_streams.contains(&1));
        assert!(
            session.release(&mut parent, &[1]).is_err(),
            "live gradient owners retain their original parent"
        );
        drop(parameters);
        drop(text);
        drop(witness);
        drop((original_logit, original_support, original_parameter));
        drop(outcome);
        session.release(&mut parent, &[1]).unwrap();
        assert!(session.retained_step_identity(&parent).is_err());
        drop(session);
    }

    #[test]
    fn semantic_work_accepts_refused_prefixes_and_rejects_impossible_effects() {
        for (edit_commands, added_supports, defined_truth_changes, valid) in [
            (0, 0, 0, true), // NO_EDIT only
            (1, 0, 0, true), // duplicate or refusal without mutation
            (2, 1, 0, true), // one attachment, then duplicate or refusal
            (2, 2, 1, true), // NEITHER -> TRUE -> BOTH
            (2, 2, 2, true), // edits to two already defined statements
            (3, 0, 0, false),
            (0, 1, 0, false),
            (1, 2, 0, false),
            (1, 1, 2, false),
            (u64::MAX, 0, 0, false),
        ] {
            let work = SemanticTransitionWork {
                edit_commands,
                added_supports,
                defined_truth_changes,
            };
            assert_eq!(work.is_valid(), valid, "{work:?}");
        }
    }

    #[test]
    fn policy_storage_follows_native_codebooks_without_allocating_null_embeddings() {
        let layout = SemanticPolicyLayout::from_components(COMPONENTS).unwrap();
        assert_eq!(layout.z, 0..4 * 128);
        assert_eq!(layout.recurrence, layout.z.end..layout.z.end + 128 * 128);
        assert_eq!(
            layout.positions,
            layout.recurrence.end..layout.recurrence.end + 18 * 128
        );
        let mut cursor = layout.positions.end;
        for (field, component) in layout.fields.iter().zip(&COMPONENTS[32..50]) {
            let cardinality = component.cardinality as usize;
            let null = component.field != 0;
            assert_eq!(field.null_category, if null { Some(0) } else { None });
            assert_eq!(field.cardinality, cardinality);
            assert_eq!(
                field.embeddings,
                cursor..cursor + (cardinality - usize::from(null)) * 128
            );
            cursor = field.embeddings.end;
            let bias_count = if null && cardinality == 1 {
                0
            } else {
                cardinality
            };
            assert_eq!(field.biases, cursor..cursor + bias_count);
            cursor = field.biases.end;
        }
        assert_eq!(layout.parameter_cells, cursor);
        assert_eq!(layout.recurrent_cells(), 2 * 37 * 128);
        assert_eq!(
            layout.score_cells(),
            COMPONENTS[32..50]
                .iter()
                .map(|c| c.cardinality as usize)
                .max()
                .unwrap()
        );
    }

    #[test]
    fn resident_ranges_reject_partial_overlap_overflow_and_empty_storage() {
        assert!(disjoint_ranges(&[(8, 16), (24, 8)]).is_ok());
        for ranges in [
            [(8, 16), (16, 16)],
            [(8, 16), (8, 8)],
            [(8, 16), (u64::MAX, 8)],
            [(8, 0), (16, 8)],
        ] {
            assert_eq!(
                disjoint_ranges(&ranges),
                Err(SemanticTransitionError::OverlappingInputs)
            );
        }
    }

    #[test]
    fn policy_layout_tracks_admitted_cardinality_and_rejects_inconsistent_slots() {
        let mut components = COMPONENTS.to_vec();
        for lane in 0..2 {
            for slot in 0..2 {
                components[lane * 68 + 32 + slot * 18 + 2].cardinality = 7;
            }
        }
        let layout = SemanticPolicyLayout::from_components(&components).unwrap();
        assert_eq!(layout.fields[2].embeddings.len(), 6 * 128);
        assert_eq!(layout.fields[2].biases.len(), 7);
        assert_eq!(layout.fields[2].null_category, Some(0));
        assert_eq!(layout.fields[0].null_category, None);
        assert_eq!(
            layout.fields[0].embeddings.len(),
            layout.fields[0].cardinality * 128
        );
        components[32 + 18 + 2].cardinality = 6;
        assert!(SemanticPolicyLayout::from_components(&components).is_err());
        assert!(SemanticPolicyLayout::from_components(&COMPONENTS[..135]).is_err());
        let mut malformed = COMPONENTS.to_vec();
        malformed[32].lane = 0;
        assert!(SemanticPolicyLayout::from_components(&malformed).is_err());
        malformed[32] = COMPONENTS[32];
        malformed[32].cardinality = 0;
        assert!(SemanticPolicyLayout::from_components(&malformed).is_err());
    }

    #[test]
    fn decoded_support_corruption_rolls_back_first_lane_and_poison_closes_session() {
        use crate::{
            SemanticPolarity, SemanticPredicateRecord, SemanticRecordRole, SemanticSupportRecord,
            SemanticTypedRecord,
        };
        use cudarc::nvrtc::Ptx;
        use xlog_core::{RelId, Schema};
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        #[derive(Clone, Copy)]
        enum IntegrityFault {
            SecondEdit,
            AfterSealedLane,
            RefusedEditThenInvalidLogit,
        }
        for fault_case in [
            IntegrityFault::SecondEdit,
            IntegrityFault::AfterSealedLane,
            IntegrityFault::RefusedEditThenInvalidLogit,
        ] {
            let late_failure = matches!(fault_case, IntegrityFault::AfterSealedLane);
            let refused_then_invalid =
                matches!(fault_case, IntegrityFault::RefusedEditThenInvalidLogit);
            let global_failure = late_failure || refused_then_invalid;
            let provider =
                crate::CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
                    .with_stream_capacity(1)
                    .build()
                    .unwrap();
            let provider = Arc::new(provider);
            let runtime = Arc::clone(provider.memory().runtime().unwrap());
            let id = runtime.stream_pool().acquire().unwrap();
            let stream = runtime.stream_pool().resolve(id).unwrap();
            let domain = provider
                .bind_resident_execution_domain(runtime, id, stream)
                .unwrap();
            let mut graph = provider
                .allocate_semantic_hypergraph(
                    &domain,
                    SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
                )
                .unwrap();
            let base = graph.empty_root();
            let roles = [
                SemanticRecordRole::Statement,
                SemanticRecordRole::Provenance,
                SemanticRecordRole::Source,
                SemanticRecordRole::Context,
                SemanticRecordRole::Scope,
            ];
            graph
                .admit_records(
                    base,
                    SemanticAdmissionRecords {
                        predicates: roles
                            .into_iter()
                            .enumerate()
                            .map(|(i, role)| SemanticPredicateRecord {
                                predicate: RelId(i as u32),
                                role,
                                schema: Schema::new(vec![]),
                            })
                            .collect(),
                        records: (0..5)
                            .map(|i| SemanticTypedRecord {
                                predicate: RelId(i),
                                arguments: vec![],
                                qualifiers: vec![],
                            })
                            .collect(),
                        supports: vec![SemanticSupportRecord {
                            statement: 0,
                            polarity: SemanticPolarity::Pro,
                            provenance: 1,
                            source: 2,
                            context: 3,
                            scope: 4,
                        }],
                    },
                    SemanticAdmissionLimits {
                        max_records: 12,
                        max_terms: 0,
                        max_references: 5,
                        max_utf8_bytes: 0,
                    },
                )
                .unwrap();
            let mut session = SemanticTransitionSession::from_hypergraph(graph).unwrap();

            // Immutable typed inputs cannot produce an invalid polarity. Inject
            // faults in this compilation of the production kernel; actual sampler,
            // mutation and cleanup remain unchanged. These are integrity failures,
            // not semantic policy refusals.
            let source = include_str!("../kernels/semantic_transition.cu");
            let anchor = if late_failure {
                "lane_live=candidate.words[41]!=0;"
            } else {
                "decode_action(books,choices,&statement,&event);"
            };
            assert_eq!(source.matches(anchor).count(), 1);
            let fault = if late_failure {
                "if (lane==1) {failed=1;state->status=3;}"
            } else if refused_then_invalid {
                "if (lane==0 && component.slot==0) event.polarity=0;"
            } else {
                "if (lane==0 && component.slot==1) event.polarity=0;"
            };
            let source = source.replacen(anchor, &format!("{anchor}\n{fault}"), 1);
            // Use the same ahead-of-time compiler as the normal kernel build. NVRTC
            // does not supply the standard headers included by this canonical source.
            struct CompilationFiles(std::path::PathBuf);
            impl Drop for CompilationFiles {
                fn drop(&mut self) {
                    for file in ["transition.cu", "transition.ptx"] {
                        if let Err(error) = std::fs::remove_file(self.0.join(file)) {
                            if error.kind() != std::io::ErrorKind::NotFound {
                                eprintln!("failed to remove test compiler output: {error}");
                            }
                        }
                    }
                    if let Err(error) = std::fs::remove_dir(&self.0) {
                        eprintln!("failed to remove test compiler directory: {error}");
                    }
                }
            }
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "xlog-decoded-support-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            let files = CompilationFiles(path);
            std::fs::write(files.0.join("transition.cu"), source).unwrap();
            let output = std::process::Command::new(
                std::env::var_os("NVCC_PATH").unwrap_or_else(|| "nvcc".into()),
            )
            .args([
                "--ptx",
                "-arch=sm_75",
                "-O3",
                "-std=c++17",
                "-I",
                env!("OUT_DIR"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/kernels"),
            ])
            .arg(files.0.join("transition.cu"))
            .arg("-o")
            .arg(files.0.join("transition.ptx"))
            .output()
            .expect("compile canonical transition with decoded support corruption");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let ptx =
                Ptx::from_src(std::fs::read_to_string(files.0.join("transition.ptx")).unwrap());
            drop(files);
            let device = provider.device().inner();
            let module = "semantic_transition_decoded_support_corruption";
            device
                .load_ptx(ptx, module, &["semantic_transition_execute"])
                .unwrap();
            session.execute = device
                .get_func(module, "semantic_transition_execute")
                .unwrap();
            let inputs: Vec<_> = session
                .components()
                .iter()
                .map(|component| {
                    let mut product_support = vec![0; component.cardinality as usize];
                    let choice = usize::from(
                        !component.is_text() && matches!(component.field, 0 | 1 | 2 | 11 | 17),
                    );
                    product_support[choice] = 1;
                    SemanticComponentInput {
                        logits: vec![0.; product_support.len()],
                        product_support,
                    }
                })
                .collect();
            session
                .bind_inputs(
                    session.binding(),
                    SemanticRngBinding {
                        model_generation: 1,
                        stream_serial: 1,
                        family_id: 1,
                        proposal: 0,
                    },
                    &inputs,
                )
                .unwrap();
            if refused_then_invalid {
                // Corrupt the retained bank only after valid cold binding. The
                // second edit's first conditional factor then fails through the
                // production nonfinite-logit check, after a genuine edit refusal
                // has retained its candidate acquisition for cleanup.
                let mut corrupted_logits: Vec<_> = inputs
                    .iter()
                    .flat_map(|input| input.logits.iter().copied())
                    .collect();
                let component = session.components()[50];
                assert_eq!((component.lane, component.slot, component.field), (1, 1, 0));
                corrupted_logits[component.offset as usize + 1] = f32::NAN;
                provider
                    .htod_sync_copy_into_tracked(&corrupted_logits, &mut session.logits)
                    .unwrap();
            }
            let [arena_ptr, _, root_capacity, statement_capacity, _, _, arena_words] =
                session.graph.transition_arena();
            assert_eq!(
                (root_capacity, statement_capacity, arena_words),
                (3, 4, 240)
            );
            assert_eq!(base.slot(), 0);
            let readback_stream = Arc::clone(&session.stream);
            let read_resident_arena = || {
                readback_stream.synchronize().unwrap();
                let mut words = vec![0u64; arena_words as usize];
                // SAFETY: the session still owns this entire allocation; its stream
                // is idle and the destination has exactly the advertised capacity.
                assert_eq!(
                    unsafe {
                        sys::cuMemcpyDtoH_v2(
                            words.as_mut_ptr().cast(),
                            arena_ptr,
                            std::mem::size_of_val(words.as_slice()),
                        )
                    },
                    sys::cudaError_enum::CUDA_SUCCESS
                );
                words
            };
            // Diagnostic I/O stays outside capture and the measured launch interval.
            // Read the device allocation itself, never the cached base_snapshot.
            let original_arena = read_resident_arena();
            // Canonical layout: sixteen control words, then sixteen words per root;
            // root statement heads precede the final candidate statement-head array.
            let base_record = 16..32;
            let heads_start = (arena_words - (root_capacity + 1) * statement_capacity) as usize;
            let base_heads = heads_start..heads_start + statement_capacity as usize;
            assert_eq!(&original_arena[19..22], &[0, 0, 0]);
            assert!(original_arena[base_heads.clone()]
                .iter()
                .all(|head| *head == 0));
            session.capture().unwrap();
            assert_eq!(session.captured.as_ref().unwrap().node_count().unwrap(), 1);
            let before = session.host_io_stats();
            session.launch().unwrap();
            assert_eq!(session.host_io_stats(), before);
            let error = session.observe(0).unwrap_err();
            if refused_then_invalid {
                assert!(matches!(
                    error,
                    SemanticTransitionError::NonFinitePolicyInput {
                        completed_draws: 50
                    }
                ));
            } else if late_failure {
                assert!(matches!(
                    error,
                    SemanticTransitionError::ObservationMismatch
                ));
            } else {
                assert!(matches!(error, SemanticTransitionError::Semantic(_)));
            }
            assert!(session.is_poisoned());
            assert_eq!(
                session.next_proposal, 0,
                "integrity error must not publish RNG successor"
            );
            assert!(matches!(
                session.launch(),
                Err(SemanticTransitionError::Poisoned)
            ));
            assert!(matches!(
                session.observe(0),
                Err(SemanticTransitionError::Poisoned)
            ));

            // SAFETY: observe completed both copies and stream waits before decoding
            // the integrity failure. Inspect that same terminal bank, with no new I/O.
            let (state, components) = unsafe { session.pinned.read() }.unwrap();
            if global_failure {
                assert_eq!(
                    (state.status, state.blocks, state.next_proposal),
                    if late_failure { (3, 68, 0) } else { (5, 50, 0) }
                );
                if late_failure {
                    assert_eq!(state.semantic_receipts[3][1], 1, "first lane really sealed");
                    assert_eq!(state.retired_roots_mask, 1);
                    assert_eq!(state.cleanup_receipts[1][0], 0, "sealed root reclaimed");
                } else {
                    assert_eq!(
                        state.semantic_receipts[1][0], 12,
                        "original edit refusal retained"
                    );
                    assert_eq!(&state.semantic_receipts[1][40..42], &[0, 1]);
                    assert_eq!(state.retired_roots_mask, 0, "no root was sealed");
                }
                assert_eq!(state.cleanup_receipts[0][0], 0, "live candidate discarded");
                assert_eq!(&state.cleanup_receipts[0][40..42], &[0, 0]);
                let terminal_arena = read_resident_arena();
                assert_eq!(
                    &terminal_arena[base_record.clone()],
                    &original_arena[base_record]
                );
                assert_eq!(
                    &terminal_arena[base_heads.clone()],
                    &original_arena[base_heads]
                );
                for slot in [32, 48, 64] {
                    assert_eq!(terminal_arena[slot], 0, "no pending root or candidate");
                }
                for (start, width) in [(80, 8), (112, 12), (160, 16)] {
                    for slot in 0..4 {
                        assert_eq!(terminal_arena[start + slot * width], 0);
                    }
                }
                drop(session);
                provider.memory().reap_pending_deallocations().unwrap();
                assert_eq!(provider.memory().allocated_bytes(), 0);
                assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
                continue;
            }
            assert_eq!((state.status, state.blocks), (0, 136));
            for (row, component) in components.iter().zip(session.components()) {
                assert_eq!(row.ordinal, component.ordinal);
                assert_eq!(row.admission_binding, session.binding().digest);
                assert_eq!(
                    row.choice,
                    u32::from(
                        !component.is_text() && matches!(component.field, 0 | 1 | 2 | 11 | 17)
                    )
                );
                assert_eq!(
                    (row.cdf_start, row.cdf_end, row.mass),
                    (0, 1 << 63, 1 << 63)
                );
                assert!(row.draw < row.cdf_end);
            }
            let receipts = state.semantic_receipts;
            assert_eq!(receipts[0][0], 0); // Acquired-base preflight succeeded.
            assert_eq!(&receipts[0][13..16], &[0, 0, 0]);
            assert_eq!(&receipts[1][0..3], &[0, 1, 1]); // First actual insertion.
            assert_eq!(receipts[2][0], 12); // Invalid second decoded support.
            assert_eq!(receipts[3][0], 12); // Refusal survives discard/seal.
            assert_eq!(&receipts[3][40..42], &[0, 1]); // Original refusal diagnostic retained.
            assert_eq!(&receipts[4][0..3], &[0, 1, 1]); // Next lane executes.
            assert_eq!(&receipts[5][0..3], &[0, 2, 1]); // Then exact duplicate.
            assert_eq!(&receipts[4][40..42], &[0, 2]); // Discarded candidate reused safely.
            assert_eq!(&receipts[5][40..42], &[0, 2]);
            assert_eq!(receipts[6][0], 0);
            assert_eq!(&receipts[6][13..16], &[1, 1, 1]);
            assert_eq!(receipts[6][6], 3); // Successful terminal retires the candidate again.
            assert_eq!(&receipts[6][3..5], &[1, 1]); // No root published by the failed lane.
            for slot in [7, 9, 11] {
                assert_eq!(receipts[4][slot], receipts[1][slot]);
                assert!(receipts[4][slot + 1] > receipts[1][slot + 1]);
            }
            // Observe has completed the failed lane's cleanup and the successful
            // next lane. Re-read resident state even though public observation is
            // now poison-closed. For this empty base, zero heads exhaust reachable
            // content; other roots and globally reused records may legitimately change.
            let terminal_arena = read_resident_arena();
            assert_eq!(&terminal_arena[..16], &original_arena[..16]);
            assert_eq!(
                &terminal_arena[base_record.clone()],
                &original_arena[base_record]
            );
            assert_eq!(
                &terminal_arena[base_heads.clone()],
                &original_arena[base_heads]
            );
            drop(session);
            provider.memory().reap_pending_deallocations().unwrap();
            assert_eq!(provider.memory().allocated_bytes(), 0);
            assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
        }
    }

    #[test]
    fn actual_enqueued_work_and_terminal_wait_faults_poison_public_session() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let provider =
            crate::CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
                .with_stream_capacity(1)
                .build()
                .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, id, stream)
            .unwrap();
        for failure in 0..12 {
            let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
            let inputs: Vec<_> = session
                .components()
                .iter()
                .map(|c| {
                    let mut mask = vec![0; c.cardinality as usize];
                    mask[0] = 1;
                    SemanticComponentInput {
                        logits: vec![0.; mask.len()],
                        product_support: mask,
                    }
                })
                .collect();
            session
                .bind_inputs(
                    session.binding(),
                    SemanticRngBinding {
                        model_generation: 1,
                        stream_serial: 2,
                        family_id: 3,
                        proposal: 4,
                    },
                    &inputs,
                )
                .unwrap();
            session.capture().unwrap();
            assert_eq!(session.captured.as_ref().unwrap().node_count().unwrap(), 1);
            if failure == 0 {
                let recorder = session.kernel_recorder();
                let graph = session.captured.as_ref().unwrap();
                let result = enqueue_recorded(&domain, &mut session.poisoned, recorder, |stream| {
                    graph.launch_in(stream)?;
                    Err::<(), _>(XlogError::Kernel(
                        "injected error after actual graph enqueue".into(),
                    ))
                });
                assert!(result.is_err());
            } else if failure == 1 {
                session.launch().unwrap();
                let result = wait_on_stream(
                    &session.stream,
                    &mut session.poisoned,
                    &mut session.stream_waits,
                    "terminal wait",
                    |stream| {
                        stream.synchronize().unwrap();
                        Err::<(), _>(XlogError::Kernel(
                            "injected terminal wait error after actual completion".into(),
                        ))
                    },
                );
                assert!(result.is_err());
            } else if failure == 2 {
                // Exercise the actual device-side rejection and typed observation:
                // corrupt a private cold state binding after public preflight.
                let invalid = DeviceState {
                    next_proposal: 4,
                    ..DeviceState::default()
                };
                provider
                    .htod_sync_copy_into_tracked(&[invalid], &mut session.state)
                    .unwrap();
                session.launch().unwrap();
                assert!(matches!(
                    session.observe(4),
                    Err(SemanticTransitionError::ObservationMismatch)
                ));
            } else if failure == 3 {
                // A captured-but-never-launched session must release its storage.
                assert!(!session.is_poisoned());
            } else if failure == 4 {
                // Repeated capture and a normal terminal observation also retire
                // all read dependencies without carrying capture-only events.
                session.capture().unwrap();
                session.launch().unwrap();
                session.observe(4).unwrap();
            } else if failure == 5 {
                // Internal integrity diagnostics must close the enclosing owner,
                // not merely appear as a nested lane error permitting replay.
                let mut words = [0; 42];
                words[0] = 11;
                assert!(session.graph.observe_transition_edit(words).is_err());
            } else if failure == 6 {
                let mut words = [0; 42];
                words[3] = u64::MAX;
                assert!(session.graph.observe_transition_root(words).is_err());
            } else {
                let mut words = [0; 42];
                words[1] = if failure == 11 { 2 } else { 1 };
                words[8] = 1;
                words[10] = 1;
                words[12] = 1;
                words[39] = session.graph.transition_arena()[1];
                match failure {
                    7 => words[39] = 0,
                    8 => words[8] = 0,
                    9 => words[10] = 0,
                    _ => words[12] = 0,
                }
                assert!(session.graph.observe_transition_edit(words).is_err());
            }
            if !(3..5).contains(&failure) {
                assert!(session.is_poisoned());
                assert!(matches!(
                    session.launch(),
                    Err(SemanticTransitionError::Poisoned)
                ));
                assert!(matches!(
                    session.observe(4),
                    Err(SemanticTransitionError::Poisoned)
                ));
            }
            drop(session);
            provider.memory().reap_pending_deallocations().unwrap();
            assert_eq!(provider.memory().allocated_bytes(), 0);
            assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
        }
    }
}

#[cfg(test)]
mod text_parent_tests {
    use super::*;

    #[test]
    fn model_content_snapshot_copies_original_seals_and_exact_contract_interval() {
        let directory = [
            PublicationRange {
                role: 18,
                generation: 1,
                length_bytes: 4,
                ..PublicationRange::default()
            },
            PublicationRange {
                role: 44,
                generation: 1,
                storage_slot: 2,
                offset_bytes: 7,
                length_bytes: 11,
                ..PublicationRange::default()
            },
            PublicationRange {
                role: 20,
                generation: 1,
                length_bytes: 4,
                ..PublicationRange::default()
            },
        ];
        let plan = model_content_copy_plan(&directory, &[2, 0], &[4, 4, 19]).unwrap();
        assert_eq!(
            plan.directory_offsets,
            [
                2 * size_of::<PublicationRange>(),
                0,
                size_of::<PublicationRange>()
            ]
        );
        assert_eq!(plan.contract_storage_slot, 2);
        assert_eq!(plan.contract_span, 7..18);
        let mut changed = directory;
        changed[1].digest = Identity256::from_bytes([91; 32]);
        let changed_plan = model_content_copy_plan(&changed, &[2, 0], &[4, 4, 19]).unwrap();
        assert_eq!(
            changed_plan.directory_offsets, plan.directory_offsets,
            "copy geometry must not upload a host-cached expected digest"
        );
    }

    #[test]
    fn model_content_snapshot_refuses_incomplete_or_unowned_ranges() {
        let directory = [
            PublicationRange {
                role: 18,
                generation: 1,
                length_bytes: 4,
                ..PublicationRange::default()
            },
            PublicationRange {
                role: 44,
                generation: 1,
                storage_slot: 1,
                offset_bytes: 3,
                length_bytes: 5,
                ..PublicationRange::default()
            },
        ];
        let plan = |rows: &[PublicationRange], indices: &[usize], allocations: &[usize]| {
            model_content_copy_plan(rows, indices, allocations)
        };
        assert!(plan(&directory, &[], &[4, 8]).is_err());
        assert!(plan(&directory, &[2], &[4, 8]).is_err());
        assert!(plan(&directory, &[1], &[4, 8]).is_err());
        assert!(plan(&directory, &[0, 0], &[4, 8]).is_err());
        assert!(plan(&directory[..1], &[0], &[4, 8]).is_err());
        assert!(plan(&directory, &[0], &[4]).is_err());
        assert!(plan(&directory, &[0], &[4, 7]).is_err());
        let mut changed = directory;
        changed[1].offset_bytes = u64::MAX;
        assert!(plan(&changed, &[0], &[4, usize::MAX]).is_err());
        changed = directory;
        changed[1].length_bytes = 0;
        assert!(plan(&changed, &[0], &[4, 8]).is_err());
        changed = directory;
        changed[1].generation = 2;
        assert!(plan(&changed, &[0], &[4, 8]).is_err());
        changed = directory;
        changed[1].index = 1;
        assert!(plan(&changed, &[0], &[4, 8]).is_err());
        let duplicate = [directory[0], directory[1], directory[1]];
        assert!(plan(&duplicate, &[0], &[4, 8]).is_err());
    }

    #[test]
    fn model_content_binding_requires_the_complete_original_roster() {
        let layout = |role, index| SemanticTensorLayout {
            role,
            index,
            scalar_type: 6,
            element_bytes: 4,
            rank: 2,
            logical_axis: u64::MAX,
            dimensions: [2, 3, 0, 0],
            strides_bytes: [12, 4, 0, 0],
        };
        let layouts: BTreeMap<_, _> = [(18, 0), (18, 1), (19, 0), (20, 0)]
            .into_iter()
            .map(|key| (key, layout(key.0, key.1)))
            .collect();
        let mut directory: Vec<_> = layouts
            .values()
            .map(|layout| PublicationRange {
                role: layout.role,
                index: layout.index,
                generation: 1,
                length_bytes: 24,
                ..PublicationRange::default()
            })
            .collect();
        directory.push(PublicationRange {
            role: 44,
            generation: 1,
            length_bytes: 8,
            ..PublicationRange::default()
        });
        let original: Vec<_> = layouts.values().map(|&layout| (layout, 0, 0, 24)).collect();
        let bind = |rows: &[(SemanticTensorLayout, u64, u64, usize)]| {
            model_content_ranges(&directory, &layouts, rows.iter().copied())
        };
        assert_eq!(bind(&original).unwrap(), vec![0, 1, 2, 3]);
        assert!(bind(&original[..3]).is_err());
        assert!(bind(&[]).is_err());
        let mut duplicate = original.clone();
        duplicate[3] = original[2];
        assert!(bind(&duplicate).is_err());
        let mut transient = original.clone();
        transient[0].0.role = 0;
        assert!(bind(&transient).is_err());
        let mut extra = original.clone();
        extra.push((layout(18, 2), 0, 0, 24));
        assert!(bind(&extra).is_err());
        let mut reordered = original.clone();
        reordered.reverse();
        assert_eq!(bind(&reordered).unwrap(), vec![3, 2, 1, 0]);
        let without_adapter: Vec<_> = directory
            .iter()
            .copied()
            .filter(|row| row.role != 19)
            .collect();
        let absent_adapter: Vec<_> = original
            .iter()
            .copied()
            .filter(|row| row.0.role != 19)
            .collect();
        assert!(
            model_content_ranges(&without_adapter, &layouts, absent_adapter.iter().copied())
                .is_ok()
        );
        assert!(
            model_content_ranges(&without_adapter, &layouts, original.iter().copied()).is_err()
        );
        assert!(model_content_ranges(&directory[..4], &layouts, original.iter().copied()).is_err());
    }

    #[test]
    fn model_content_binding_accepts_logical_equality_not_changed_shape_or_interval() {
        let expected = SemanticTensorLayout {
            role: 18,
            index: 0,
            scalar_type: 6,
            element_bytes: 4,
            rank: 2,
            logical_axis: 0,
            dimensions: [2, 3, 0, 0],
            strides_bytes: [12, 4, 0, 0],
        };
        let layouts = BTreeMap::from([((18, 0), expected)]);
        let directory = [
            PublicationRange {
                role: 18,
                generation: 1,
                length_bytes: 24,
                logical_begin: 5,
                logical_end: 7,
                ..PublicationRange::default()
            },
            PublicationRange {
                role: 44,
                generation: 1,
                length_bytes: 8,
                ..PublicationRange::default()
            },
        ];
        let bind = |layout, begin, end, bytes| {
            model_content_ranges(
                &directory,
                &layouts,
                std::iter::once((layout, begin, end, bytes)),
            )
        };
        let mut actual = expected;
        actual.strides_bytes = [4, 8, 0, 0];
        assert_eq!(bind(actual, 5, 7, 24).unwrap(), vec![0]);
        actual.strides_bytes = [32, 4, 0, 0];
        assert_eq!(bind(actual, 5, 7, 44).unwrap(), vec![0]);
        assert!(bind(actual, 5, 7, 24).is_err());
        assert!(bind(actual, 6, 8, 44).is_err());
        actual.strides_bytes = [4, 4, 0, 0];
        assert!(bind(actual, 5, 7, 16).is_err());
        actual = expected;
        actual.dimensions = [3, 2, 0, 0];
        actual.strides_bytes = [8, 4, 0, 0];
        assert!(bind(actual, 5, 8, 24).is_err());
        actual = expected;
        actual.scalar_type = 2;
        assert!(bind(actual, 5, 7, 24).is_err());
    }

    #[test]
    fn tensor_content_range_accepts_private_operands_without_publication_roles() {
        for (scalar_type, element_bytes, dtype) in [
            (
                7,
                8,
                crate::dlpack::DLDataType {
                    code: 0,
                    bits: 64,
                    lanes: 1,
                },
            ),
            (
                8,
                1,
                crate::dlpack::DLDataType {
                    code: 6,
                    bits: 8,
                    lanes: 1,
                },
            ),
            (
                6,
                4,
                crate::dlpack::DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
            ),
        ] {
            let mut layout = SemanticTensorLayout {
                role: 0,
                index: 0,
                scalar_type,
                element_bytes,
                rank: 2,
                logical_axis: u64::MAX,
                dimensions: [2, 3, 0, 0],
                strides_bytes: [element_bytes, 2 * element_bytes, 0, 0],
            };
            let metadata = TensorMetadata {
                device_type: 2,
                device_id: 0,
                dtype,
                shape: &[2, 3],
                strides: Some(&[1, 2]),
                data: 4096,
                byte_offset: 0,
            };
            let bytes = 6 * element_bytes as usize;
            assert_eq!(
                validate_tensor_metadata(&layout, &metadata, 0).unwrap(),
                (4096, bytes)
            );
            assert_eq!(tensor_content_range(&layout, 0, 0, bytes).unwrap().role, 0);
            assert!(SemanticStateRole::from_code(0).is_none());
            layout.role = 56;
            assert!(validate_tensor_metadata(&layout, &metadata, 0).is_err());
            assert!(tensor_content_range(&layout, 0, 0, bytes).is_err());
            layout.role = 0;
            layout.dimensions[0] = 0;
            assert_eq!(
                tensor_content_range(&layout, 0, 0, 0).unwrap().length_bytes,
                0
            );
        }
    }

    #[test]
    fn content_coordinates_keep_private_occurrences_in_original_roster_order() {
        assert!(validate_content_coordinates([(0, 0), (0, 1), (0, 2)].into_iter()).is_ok());
        assert!(validate_content_coordinates([(4, 1), (0, 1), (5, 1)].into_iter()).is_ok());
        for coordinates in [
            vec![(0, 1)],
            vec![(0, 0), (0, 0)],
            vec![(0, 0), (0, 2)],
            vec![(4, 0), (4, 0)],
            vec![(56, 0)],
        ] {
            assert!(validate_content_coordinates(coordinates.into_iter()).is_err());
        }
    }

    #[test]
    fn tensor_content_range_preserves_signed_tokens_and_boolean_masks() {
        for (scalar_type, element_bytes, dtype) in [
            (
                7,
                8,
                crate::dlpack::DLDataType {
                    code: 0,
                    bits: 64,
                    lanes: 1,
                },
            ),
            (
                8,
                1,
                crate::dlpack::DLDataType {
                    code: 6,
                    bits: 8,
                    lanes: 1,
                },
            ),
        ] {
            let layout = SemanticTensorLayout {
                role: 1,
                index: 0,
                scalar_type,
                element_bytes,
                rank: 2,
                logical_axis: 1,
                dimensions: [1, 3, 0, 0],
                strides_bytes: [3 * element_bytes, element_bytes, 0, 0],
            };
            let metadata = TensorMetadata {
                device_type: 2,
                device_id: 0,
                dtype,
                shape: &[1, 3],
                strides: Some(&[3, 1]),
                data: 4096,
                byte_offset: 0,
            };
            assert_eq!(
                validate_tensor_metadata(&layout, &metadata, 0).unwrap(),
                (4096, 3 * element_bytes as usize)
            );
            let range = tensor_content_range(&layout, 11, 14, 3 * element_bytes as usize).unwrap();
            assert_eq!(
                (
                    range.role,
                    range.index,
                    range.logical_begin,
                    range.logical_end
                ),
                (1, 0, 11, 14)
            );
            assert!(tensor_content_range(&layout, 11, 13, 3 * element_bytes as usize).is_err());
            let mut wrong = metadata;
            wrong.dtype.code = 1;
            assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        }
    }

    #[test]
    fn tensor_content_range_checks_transposed_empty_and_exact_byte_extents() {
        let mut layout = SemanticTensorLayout {
            role: 4,
            index: 2,
            scalar_type: 6,
            element_bytes: 4,
            rank: 2,
            logical_axis: u64::MAX,
            dimensions: [2, 3, 0, 0],
            strides_bytes: [4, 8, 0, 0],
        };
        assert_eq!(
            tensor_content_range(&layout, 0, 0, 24)
                .unwrap()
                .length_bytes,
            24
        );
        assert!(tensor_content_range(&layout, 0, 1, 24).is_err());
        assert!(tensor_content_range(&layout, 0, 0, 23).is_err());
        assert!(tensor_content_range(&layout, 0, 0, 25).is_err());
        layout.logical_axis = 0;
        layout.dimensions[0] = 0;
        assert_eq!(
            tensor_content_range(&layout, 9, 9, 0).unwrap().length_bytes,
            0
        );
        assert!(tensor_content_range(&layout, 9, 8, 0).is_err());
    }

    fn publication_test_tensor(
        provider: &CudaKernelProvider,
        role: u64,
        value: f32,
    ) -> SemanticTensorInput {
        let bytes = value.to_le_bytes();
        let owner = allocate_publication::<u8>(provider, bytes.len()).unwrap();
        upload_publication(provider, &bytes, &owner).unwrap();
        let tensor = crate::dlpack::export_slice_managed_tensor(
            Arc::new(owner),
            provider.device().ordinal() as i32,
            crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            1,
            1,
        )
        .unwrap();
        SemanticTensorInput {
            layout: SemanticTensorLayout {
                role,
                index: 0,
                scalar_type: 6,
                element_bytes: 4,
                rank: 2,
                logical_axis: u64::MAX,
                dimensions: [1, 1, 0, 0],
                strides_bytes: [4, 4, 0, 0],
            },
            logical_begin: 0,
            logical_end: 0,
            tensor,
            native_allocation: None,
        }
    }

    pub(super) fn publication_test_session(
        provider: Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
    ) -> SemanticTransitionSession {
        use crate::{SemanticArgument, SemanticPredicateRecord, SemanticTypedRecord};
        use xlog_core::{RelId, ScalarType, Schema};
        let mut graph = provider
            .allocate_semantic_hypergraph(
                domain,
                crate::SemanticHypergraphCapacities::try_new(4, 8, 8, 8).unwrap(),
            )
            .unwrap();
        let records = SemanticAdmissionRecords {
            predicates: vec![SemanticPredicateRecord {
                predicate: RelId(1),
                role: SemanticRecordRole::Statement,
                schema: Schema::new(vec![
                    ("left".into(), ScalarType::U32),
                    ("right".into(), ScalarType::U32),
                ])
                .with_sort_labels(vec!["bit".into(), "bit".into()])
                .unwrap(),
            }],
            records: [[1, 1], [0, 1]]
                .into_iter()
                .map(|values| SemanticTypedRecord {
                    predicate: RelId(1),
                    arguments: values.map(SemanticArgument::U32).to_vec(),
                    qualifiers: vec![],
                })
                .collect(),
            supports: vec![],
        };
        graph
            .admit_records(
                graph.empty_root(),
                records,
                SemanticAdmissionLimits {
                    max_records: 8,
                    max_terms: 32,
                    max_references: 8,
                    max_utf8_bytes: 256,
                },
            )
            .unwrap();
        let mut session = SemanticTransitionSession::from_hypergraph(graph).unwrap();
        session
            .bind_task_evaluation(task_binding_tests::task_spec([0, 1, 1], vec![]))
            .unwrap();
        session
    }

    fn publication_test_provider() -> (Arc<CudaKernelProvider>, ResidentExecutionDomain) {
        let provider = Arc::new(
            crate::CudaProviderBuilder::new(
                0,
                xlog_core::MemoryBudget::with_limit(512 * 1024 * 1024),
            )
            .with_stream_capacity(1)
            .build()
            .unwrap(),
        );
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, stream)
            .unwrap();
        (provider, domain)
    }

    pub(super) fn publication_test_parent(provider: &CudaKernelProvider) -> SemanticParentBinding {
        let mut counts = [1u64; 55];
        counts[15] = 2;
        for role in [4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 51, 52, 53, 54] {
            counts[role - 1] = 0;
        }
        let mut records = Vec::new();
        let mut model_contract_layout = SemanticModelContractLayout::default();
        for role in 1..=55 {
            if is_tensor_role(role)
                || matches!(role, 1..=3 | 14..=16 | 30 | 31 | 33 | 43 | 47 | 48 | 55)
            {
                continue;
            }
            let mut bytes = format!("application record {role}").into_bytes();
            if role == 44 {
                model_contract_layout = SemanticModelContractLayout {
                    schema_begin: 104,
                    schema_bytes: bytes.len() as u64,
                    schema_digest_offset: 0,
                    generation_offset: 32,
                    numerical_digest_offset: 40,
                    identity_offset: 72,
                };
                let mut record = vec![0; 104];
                record.extend_from_slice(&bytes);
                bytes = record;
            }
            records.push(SemanticStateRecord {
                role: SemanticStateRole::from_code(role).unwrap(),
                index: 0,
                capacity_bytes: 128,
                bytes,
            });
        }
        let slot = |token, logical_position, committed| SemanticTextSlot {
            token,
            logical_position,
            kind: 1,
            provenance: 1,
            valid: 1,
            committed,
            recomputed: 1,
            provenance_record: 0,
        };
        let mut source = [SemanticTextSlot::default(); 32];
        source[7] = slot(8, 1, 0);
        let model_owner = Arc::new(allocate_publication::<u8>(provider, 4).unwrap());
        upload_publication(provider, &18f32.to_le_bytes(), &model_owner).unwrap();
        let tensor = crate::dlpack::export_slice_managed_tensor(
            Arc::clone(&model_owner),
            provider.device().ordinal() as i32,
            crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            1,
            1,
        )
        .unwrap();
        let backing = crate::dlpack::export_slice_managed_tensor(
            Arc::clone(&model_owner),
            provider.device().ordinal() as i32,
            crate::dlpack::DLDataType {
                code: 1,
                bits: 8,
                lanes: 1,
            },
            1,
            4,
        )
        .unwrap();
        // SAFETY: this newly owned export has not been shared. Dropping its
        // singleton dimension preserves all four bytes and the original owner;
        // the retained shape allocation still owns the selected second entry.
        unsafe {
            let metadata = &mut (*backing.as_ptr()).dl_tensor;
            metadata.ndim = 1;
            metadata.shape = metadata.shape.add(1);
        }
        let model_layout = SemanticTensorLayout {
            role: 18,
            index: 0,
            scalar_type: 6,
            element_bytes: 4,
            rank: 2,
            logical_axis: u64::MAX,
            dimensions: [1, 1, 0, 0],
            strides_bytes: [4, 4, 0, 0],
        };
        let mut tensors = (8..=13)
            .map(|role| publication_test_tensor(provider, role, role as f32))
            .collect::<Vec<_>>();
        tensors.push(SemanticTensorInput {
            layout: model_layout,
            logical_begin: 0,
            logical_end: 0,
            tensor,
            native_allocation: None,
        });
        let mut parent = SemanticParentBinding {
            recovered_instance: None,
            source,
            prefix: vec![slot(7, 0, 1)],
            ring_head: 7,
            provenance_records: 0,
            prefix_capacity: 64,
            feedback_capacity: 2,
            max_position: 262144,
            pad_token: 0,
            terminal_tokens: vec![100],
            final_intent_payload_bytes: 8,
            intent_effect: b"record result".to_vec(),
            intent_entry_capacity: 4,
            intent_payload_capacity_bytes: 256,
            model_generation: 1,
            policy_generation: 1,
            neural_generation: 1,
            cache_generation: 1,
            model_numerical_mode: b"native test numerical mode".to_vec(),
            model_contract_layout,
            authority_generation: 1,
            training_cursor: 0,
            training_rng: [1, 2, 3, 4],
            fuel: 8,
            rng: SemanticRngBinding {
                model_generation: 1,
                stream_serial: 5,
                family_id: 2,
                proposal: 0,
            },
            topology_identity: Identity256::from_bytes([4; 32]),
            table_identity: Identity256::from_bytes([5; 32]),
            role_counts: counts,
            records,
            tensors,
            model_memory: SemanticModelMemory {
                allocations: vec![SemanticTensorInput {
                    layout: SemanticTensorLayout {
                        role: 0,
                        index: 0,
                        scalar_type: 1,
                        element_bytes: 1,
                        rank: 1,
                        logical_axis: u64::MAX,
                        dimensions: [4, 0, 0, 0],
                        strides_bytes: [1, 0, 0, 0],
                    },
                    logical_begin: 0,
                    logical_end: 0,
                    tensor: backing,
                    native_allocation: None,
                }],
                storages: vec![SemanticModelStorage {
                    allocation: 0,
                    byte_offset: 0,
                    span_bytes: 4,
                }],
                views: vec![SemanticModelView {
                    role: 18,
                    index: 0,
                    storage: 0,
                    byte_offset: 0,
                }],
            },
            active_layouts: vec![],
        };
        parent
            .bind_initial_sources(
                &[SemanticObservedSource {
                    identity: "observed-input".into(),
                    tokens: vec![7, 8],
                    origin: b"complete observed fixture input".to_vec(),
                }],
                &(0..2)
                    .map(|position| SemanticSourceMapping {
                        source: "observed-input".into(),
                        offset: position,
                        logical_position: position,
                    })
                    .collect::<Vec<_>>(),
                b"current fixture authority",
                64,
            )
            .unwrap();
        parent
    }

    fn publication_test_tensor_pair(
        provider: &CudaKernelProvider,
        layout: SemanticTensorLayout,
        bytes: &[u8],
    ) -> (SemanticTensorInput, SemanticTensorInput) {
        use xlog_core::{ScalarType, Schema};
        let owner = allocate_publication::<u8>(provider, bytes.len()).unwrap();
        upload_publication(provider, bytes, &owner).unwrap();
        let scalar = match layout.scalar_type {
            3 => ScalarType::U64,
            6 => ScalarType::F32,
            8 => ScalarType::Bool,
            _ => panic!("fixture tensor has an unsupported scalar type"),
        };
        let (original, captured) = if layout.rank == 1 {
            let buffer = provider
                .buffer_from_columns(
                    vec![crate::CudaColumn::Owned(owner)],
                    layout.dimensions[0],
                    Schema::new(vec![("value".into(), scalar)]),
                )
                .unwrap();
            let table = provider.to_dlpack_table(buffer);
            (table.column(0).unwrap(), table.column(0).unwrap())
        } else {
            assert_eq!(layout.rank, 2);
            let dtype = match scalar {
                ScalarType::U64 => crate::dlpack::DLDataType {
                    code: 1,
                    bits: 64,
                    lanes: 1,
                },
                ScalarType::F32 => crate::dlpack::DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                _ => unreachable!("fixture matrices use floating point or unsigned integers"),
            };
            let owner = Arc::new(owner);
            let export = || {
                crate::dlpack::export_slice_managed_tensor(
                    Arc::clone(&owner),
                    provider.device().ordinal() as i32,
                    dtype,
                    layout.dimensions[0] as usize,
                    layout.dimensions[1] as usize,
                )
                .unwrap()
            };
            (export(), export())
        };
        let input = |tensor| SemanticTensorInput {
            layout,
            logical_begin: 0,
            logical_end: 0,
            tensor,
            native_allocation: None,
        };
        (input(original), input(captured))
    }

    pub(super) fn publication_test_continuation(
        session: &mut SemanticTransitionSession,
        old: &SemanticPublishedLease,
        provider: &CudaKernelProvider,
    ) -> (SemanticContinuationInput, SemanticTensorContentWitness) {
        publication_test_continuation_with_numerical_admissibility(session, old, provider, 1)
    }

    pub(super) fn publication_test_continuation_with_numerical_admissibility(
        session: &mut SemanticTransitionSession,
        old: &SemanticPublishedLease,
        provider: &CudaKernelProvider,
        numerical_admissibility: u8,
    ) -> (SemanticContinuationInput, SemanticTensorContentWitness) {
        assert!(numerical_admissibility <= 1);
        let contract = session.publication.as_ref().unwrap().contract_value;
        let feedback = session
            .read_published_record_bytes(old, SemanticStateRole::RawFeedback)
            .unwrap();
        assert_eq!(
            feedback.len(),
            contract.feedback_capacity as usize * size_of::<RawFeedbackRecord>()
        );
        let mut filled: Vec<_> = old
            .source
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.valid == 1 && slot.kind == 1)
            .collect();
        let mut masks: Vec<_> = old
            .source
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.valid == 1 && slot.kind == 2)
            .collect();
        filled.sort_by_key(|(_, slot)| slot.logical_position);
        masks.sort_by_key(|(_, slot)| slot.logical_position);
        let mut text_rows = vec![0u64; 32 * 2];
        for (row, &(slot, source)) in masks.iter().enumerate() {
            text_rows[row * 2] = slot as u64;
            text_rows[row * 2 + 1] = source.logical_position;
        }
        let capacity = 32 + contract.feedback_capacity;
        let mut active_rows = vec![0u64; capacity as usize * 4];
        let mut active_count = 0u64;
        let mut append = |physical, slot, position, kind| {
            let offset = active_count as usize * 4;
            active_rows[offset..offset + 4].copy_from_slice(&[physical, slot, position, kind]);
            active_count += 1;
        };
        for (physical, &(slot, source)) in filled.iter().enumerate() {
            append(physical as u64, slot as u64, source.logical_position, 1);
        }
        for (slot, record) in feedback
            .chunks_exact(size_of::<RawFeedbackRecord>())
            .enumerate()
        {
            let valid = u64::from_ne_bytes(record[..8].try_into().unwrap());
            assert!(valid <= 1);
            if valid == 1 {
                append(
                    (filled.len() + slot) as u64,
                    32 + slot as u64,
                    contract.prefix_capacity + 32 + slot as u64,
                    3,
                );
            }
        }
        for (offset, &(slot, source)) in masks.iter().enumerate() {
            append(
                filled.len() as u64 + contract.feedback_capacity + offset as u64,
                slot as u64,
                source.logical_position,
                2,
            );
        }
        let mut tensors = Vec::new();
        let mut capture = Vec::new();
        for role in 8..=13 {
            let layout = SemanticTensorLayout {
                role,
                index: 0,
                scalar_type: 6,
                element_bytes: 4,
                rank: 2,
                logical_axis: u64::MAX,
                dimensions: [1, 1, 0, 0],
                strides_bytes: [4, 4, 0, 0],
            };
            let (original, witnessed) =
                publication_test_tensor_pair(provider, layout, &(role as f32 + 1.0).to_ne_bytes());
            tensors.push(original);
            capture.push(witnessed);
        }
        let service_bytes = [
            publication_abi_bytes(&text_rows),
            publication_abi_bytes(&[masks.len() as u64]),
            vec![0u8; 32],
            publication_abi_bytes(&active_rows),
            publication_abi_bytes(&[active_count]),
            vec![numerical_admissibility],
        ];
        let layouts = continuation_service_layouts(tensors.len(), capacity).unwrap();
        let mut services = Vec::new();
        for (layout, bytes) in layouts.into_iter().zip(service_bytes) {
            let (original, witnessed) = publication_test_tensor_pair(provider, layout, &bytes);
            services.push(original.tensor);
            capture.push(witnessed);
        }
        let witness = session.capture_tensor_content(old, capture, 1).unwrap();
        let [text_rows, text_row_count, selected_text, active_rows, active_row_count, numerical_admissibility]:
            [DlpackManagedTensor; 6] = services.try_into().ok().unwrap();
        (
            SemanticContinuationInput {
                tensors,
                authority_decisions: b"current fixture continuation".to_vec(),
                text_rows,
                text_row_count,
                selected_text,
                active_rows,
                active_row_count,
                numerical_admissibility,
            },
            witness,
        )
    }

    fn publication_test_recompute(
        session: &mut SemanticTransitionSession,
        old: &SemanticPublishedLease,
        provider: &crate::CudaKernelProvider,
    ) {
        session
            .admit_transition(old, SemanticTransitionKind::Recompute)
            .unwrap();
        let (continuation, witness) = publication_test_continuation(session, old, provider);
        session
            .bind_continuation(old, continuation, &witness, 1)
            .unwrap();
        drop(witness);
        session.capture().unwrap();
        session.launch().unwrap();
        session.observe(0).unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn publication_material_native_restore_preserves_acquired_prefix_feedback_and_history() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut old = session.acquire().unwrap();
        let old_prefix = session
            .read_published_record_bytes(&old, SemanticStateRole::PrefixSource)
            .unwrap();
        let old_mode = session.published_model_numerical_mode(&old).unwrap();
        assert_eq!(old_mode, b"native test numerical mode");
        let old_material = session.published_state_material(&old).unwrap();
        let (_, alias) = session.published_prefix(&old, 1).unwrap();
        publication_test_recompute(&mut session, &old, &provider);
        let mut current = session.acquire().unwrap();
        assert_eq!(current.header.prefix_extent, 2);
        assert_eq!(
            session
                .read_published_record_bytes(&old, SemanticStateRole::PrefixSource)
                .unwrap(),
            old_prefix
        );
        assert_eq!(
            session.published_state_material(&old).unwrap(),
            old_material
        );
        assert!(session.release(&mut old, &[1]).is_err());
        drop(alias);
        session.release(&mut old, &[1]).unwrap();
        let bytes = session.published_state_material(&current).unwrap();
        let original = PublicationMaterial::decode(&bytes).unwrap();
        session.release(&mut current, &[]).unwrap();
        drop(session);
        let mut restored = publication_test_session(Arc::clone(&provider), &domain);
        let identity = restored.restore_state_material(&bytes).unwrap();
        assert_ne!(identity.instance, original.bank.header.instance);
        assert_eq!(identity.logical_digest, original.bank.header.logical_digest);
        assert_eq!(identity.state_digest, original.bank.header.state_digest);
        let mut lease = restored.acquire().unwrap();
        assert_eq!(
            restored.published_model_numerical_mode(&lease).unwrap(),
            old_mode
        );
        let reread =
            PublicationMaterial::decode(&restored.published_state_material(&lease).unwrap())
                .unwrap();
        for role in [1, 2, 14, 15, 27, 28, 30, 31, 32, 33, 43] {
            assert_eq!(
                reread
                    .ranges
                    .iter()
                    .find(|item| item.range.role == role)
                    .unwrap()
                    .bytes,
                original
                    .ranges
                    .iter()
                    .find(|item| item.range.role == role)
                    .unwrap()
                    .bytes,
                "role {role}"
            );
        }
        assert_eq!(reread.graph, original.graph);
        restored.release(&mut lease, &[]).unwrap();
        drop(restored);
        provider.memory().reap_pending_deallocations().unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn publication_material_runtime_reader_rejects_completed_layout_alias_writes() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut lease = session.acquire().unwrap();
        let mode = session.published_model_numerical_mode(&lease).unwrap();
        let original = session
            .read_published_record_bytes(&lease, SemanticStateRole::TensorLayout)
            .unwrap();
        let alias = session
            .published_record(&lease, SemanticStateRole::TensorLayout, 0, 1)
            .unwrap();
        // SAFETY: the live managed alias retains its descriptor and the exact
        // published allocation for the duration of both consumer writes.
        let tensor = unsafe { &(*alias.as_ptr()).dl_tensor };
        let pointer = tensor.data as usize as u64 + tensor.byte_offset;
        // Mutate a layout value and then only the active-computed marker. The
        // latter leaves cold layouts intact, so the original seal is necessary.
        for offset in [
            size_of::<TensorLayoutTableHeader>() + 2 * size_of::<u64>(),
            2 * size_of::<u64>(),
        ] {
            assert!(offset < original.len());
            // SAFETY: one in-bounds byte of this retained alias, submitted on
            // its registered legacy stream; no further writer is submitted
            // until the getter has completed and checked that stream prefix.
            assert_eq!(
                unsafe {
                    sys::cuMemsetD8Async(
                        pointer + offset as u64,
                        original[offset] ^ 1,
                        1,
                        std::ptr::null_mut(),
                    )
                },
                sys::cudaError_enum::CUDA_SUCCESS
            );
            assert!(matches!(
                session.published_model_numerical_mode(&lease),
                Err(SemanticTransitionError::ObservationMismatch)
            ));
            // SAFETY: restore this test's byte on the same retained alias and
            // registered stream so ordinary retirement sees the original state.
            assert_eq!(
                unsafe {
                    sys::cuMemsetD8Async(
                        pointer + offset as u64,
                        original[offset],
                        1,
                        std::ptr::null_mut(),
                    )
                },
                sys::cudaError_enum::CUDA_SUCCESS
            );
            assert_eq!(
                session.published_model_numerical_mode(&lease).unwrap(),
                mode
            );
        }
        drop(alias);
        session.release(&mut lease, &[1]).unwrap();
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run; fatal cases use fresh processes"]
    fn replay_boundaries_reject_mutated_shared_predecessor_weights() {
        const CHILD_KIND: &str = "XLOG_REPLAY_CONTENT_TEST_CHILD";
        const COMPLETED: &str =
            "shared replay content mutation produced a fatal CUDA completion error";
        let Some(kind) = std::env::var_os(CHILD_KIND) else {
            let qualified = concat!(
                module_path!(),
                "::replay_boundaries_reject_mutated_shared_predecessor_weights"
            );
            let test_name = qualified.split_once("::").unwrap().1;
            for kind in ["verify", "evidence", "provenance"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        test_name,
                        "--ignored",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(CHILD_KIND, kind)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{kind} child did not verify a CUDA trap:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains(COMPLETED),
                    "{kind} child did not reach the typed CUDA completion check"
                );
            }
            return;
        };
        assert!(kind == "verify" || kind == "evidence" || kind == "provenance");
        let (provider, domain) = publication_test_provider();
        let mut original = publication_test_session(Arc::clone(&provider), &domain);
        original
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut old = original.acquire().unwrap();
        let predecessor_bytes = original.published_state_material(&old).unwrap();
        publication_test_recompute(&mut original, &old, &provider);
        let mut current = original.acquire().unwrap();
        let evidence = original.published_replay_evidence(&old, &current).unwrap();
        assert!(
            original.release_events.is_empty(),
            "completed export events must not wait for reader release"
        );
        let material = SemanticReplayMaterial::decode(
            &predecessor_bytes,
            &evidence,
            old.identity.logical_digest,
            current.identity.logical_digest,
        )
        .unwrap();
        original.release(&mut old, &[1]).unwrap();
        original.release(&mut current, &[1]).unwrap();
        drop(original);

        let mut replay = publication_test_session(Arc::clone(&provider), &domain);
        replay.restore_replay_material(&material).unwrap();
        let predecessor = replay.acquire().unwrap();
        publication_test_recompute(&mut replay, &predecessor, &provider);
        let successor = replay.acquire().unwrap();
        let alias = replay
            .published_tensor(&predecessor, SemanticStateRole::SlowWeights, 0, 1)
            .unwrap();
        let successor_alias = replay
            .published_tensor(&successor, SemanticStateRole::SlowWeights, 0, 1)
            .unwrap();
        // SAFETY: each managed export retains its own actual acquired reader and
        // valid DLPack descriptor; the pointers prove these banks share weights.
        let tensor = unsafe { &(*alias.as_ptr()).dl_tensor };
        let successor_tensor = unsafe { &(*successor_alias.as_ptr()).dl_tensor };
        assert_eq!(tensor.byte_offset, 0);
        assert_eq!(tensor.data, successor_tensor.data);
        let pointer = tensor.data as usize as u64;
        drop(successor_alias);
        replay
            .verify_replay_publication(&material, &predecessor, &successor)
            .unwrap();
        replay
            .published_replay_evidence(&predecessor, &successor)
            .unwrap();
        replay
            .published_replay_provenance(&predecessor, &successor)
            .unwrap();
        // SAFETY: the fixture's live alias retains four FP32 bytes, and the
        // external write uses the actual registered legacy consumer stream.
        assert_eq!(
            unsafe { sys::cuMemsetD8Async(pointer, 0, 4, std::ptr::null_mut()) },
            sys::cudaError_enum::CUDA_SUCCESS
        );
        if kind == "verify" {
            drop(alias);
            replay.quiesce_published_reader(&predecessor, &[1]).unwrap();
            assert!(replay
                .verify_replay_publication(&material, &predecessor, &successor)
                .is_err());
        } else if kind == "evidence" {
            assert!(replay
                .published_replay_evidence(&predecessor, &successor)
                .is_err());
        } else {
            assert!(replay
                .published_replay_provenance(&predecessor, &successor)
                .is_err());
        }
        // SAFETY: only this disposable child's context has been poisoned. A
        // typed fatal CUDA error, not a Rust panic, establishes guard refusal.
        let completion = unsafe { sys::cuStreamSynchronize(std::ptr::null_mut()) };
        assert!(
            matches!(
                completion,
                sys::cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION
                    | sys::cudaError_enum::CUDA_ERROR_LAUNCH_FAILED
            ),
            "unexpected CUDA completion: {completion:?}"
        );
        println!("{COMPLETED}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::process::exit(0);
    }

    fn publication_test_alias(
        session: &mut SemanticTransitionSession,
        lease: &SemanticPublishedLease,
        role: SemanticStateRole,
        consumer_stream: u64,
    ) -> SemanticTensorInput {
        let layout = session.publication.as_ref().unwrap().layouts[&(role as u64, 0)];
        let range = *lease
            .directory
            .iter()
            .find(|range| range.role == role as u64 && range.index == 0)
            .unwrap();
        let (native_allocation, backing) = session
            .published_tensor_allocation(lease, role, 0, consumer_stream)
            .unwrap();
        drop(backing);
        SemanticTensorInput {
            layout,
            logical_begin: range.logical_begin,
            logical_end: range.logical_end,
            tensor: session
                .published_tensor(lease, role, 0, consumer_stream)
                .unwrap(),
            native_allocation: Some(native_allocation),
        }
    }

    struct ContentTestExport {
        managed: crate::dlpack::DLManagedTensor,
        _owner: DlpackManagedTensor,
        shape: [i64; 2],
        strides: [i64; 2],
    }

    unsafe extern "C" fn delete_content_test_export(managed: *mut crate::dlpack::DLManagedTensor) {
        if managed.is_null() {
            return;
        }
        // SAFETY: content_test_export installs this sole Box owner and deleter;
        // its canonical managed backing owns the real allocation until drop.
        let owner = unsafe { (*managed).manager_ctx.cast::<ContentTestExport>() };
        if !owner.is_null() {
            drop(unsafe { Box::from_raw(owner) });
        }
    }

    fn content_test_export(
        owner: Arc<TrackedCudaSlice<u8>>,
        device_id: i32,
        dtype: crate::dlpack::DLDataType,
        layout: SemanticTensorLayout,
        logical_interval: (u64, u64),
    ) -> SemanticTensorInput {
        assert_eq!(layout.rank, 2);
        let shape = [
            i64::try_from(layout.dimensions[0]).unwrap(),
            i64::try_from(layout.dimensions[1]).unwrap(),
        ];
        let strides = [
            i64::try_from(layout.strides_bytes[0] / layout.element_bytes).unwrap(),
            i64::try_from(layout.strides_bytes[1] / layout.element_bytes).unwrap(),
        ];
        let backing = crate::dlpack::export_slice_managed_tensor(
            owner,
            device_id,
            dtype,
            usize::try_from(shape[0]).unwrap(),
            usize::try_from(shape[1]).unwrap(),
        )
        .unwrap();
        // SAFETY: the canonical export owns valid, producer-ready metadata and
        // storage. The test producer supplies a genuine view of those bytes,
        // retaining both explicit strides and the original owner in one Box.
        let source = unsafe { &(*backing.as_ptr()).dl_tensor };
        let mut export = Box::new(ContentTestExport {
            managed: crate::dlpack::DLManagedTensor {
                dl_tensor: crate::dlpack::DLTensor {
                    data: source.data,
                    device: source.device,
                    ndim: 2,
                    dtype: source.dtype,
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: source.byte_offset,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(delete_content_test_export),
            },
            _owner: backing,
            shape,
            strides,
        });
        export.managed.dl_tensor.shape = export.shape.as_mut_ptr();
        export.managed.dl_tensor.strides = export.strides.as_mut_ptr();
        let owner = Box::into_raw(export);
        // SAFETY: the stable Box retains the actual managed CUDA allocation
        // and both metadata arrays; the assigned deleter consumes that Box once.
        let tensor = unsafe {
            (*owner).managed.manager_ctx = owner.cast();
            DlpackManagedTensor::from_raw(&mut (*owner).managed)
        };
        SemanticTensorInput {
            layout,
            logical_begin: logical_interval.0,
            logical_end: logical_interval.1,
            tensor,
            native_allocation: None,
        }
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn native_feedback_alias_capture_reuses_original_producer_seals() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut lease = session.acquire().unwrap();
        session
            .admit_transition(&lease, SemanticTransitionKind::Proposal)
            .unwrap();
        let feedback = session.published_feedback(&lease, 1).unwrap();
        let layouts = session.steps[&lease.token].feedback[0]
            .content
            .tensors
            .iter()
            .map(|tensor| tensor.layout)
            .collect::<Vec<_>>();
        let inputs = feedback
            .into_dlpack()
            .into_iter()
            .zip(layouts.iter().copied())
            .map(|(tensor, layout)| SemanticTensorInput {
                tensor,
                layout,
                logical_begin: 0,
                logical_end: 0,
                native_allocation: None,
            })
            .collect();
        let mut other = session.acquire().unwrap();
        assert!(
            session.capture_tensor_content(&other, inputs, 1).is_err(),
            "a second reader must not reseal the first reader's native outputs"
        );
        session.release(&mut other, &[1]).unwrap();
        let feedback = session.published_feedback(&lease, 1).unwrap();
        let inputs = feedback
            .into_dlpack()
            .into_iter()
            .zip(layouts)
            .map(|(tensor, layout)| SemanticTensorInput {
                tensor,
                layout,
                logical_begin: 0,
                logical_end: 0,
                native_allocation: None,
            })
            .collect();
        let witness = session.capture_tensor_content(&lease, inputs, 1).unwrap();
        let reader = &session.steps[&lease.token];
        let TensorContentSeals::Captured(original) = &reader.feedback[1].content.seals else {
            panic!("native producer seals");
        };
        let TensorContentSeals::Captured(consumer) = &reader.content[witness.index].seals else {
            panic!("original alias seals");
        };
        assert_eq!(original.len(), 6);
        for (original, consumer) in original.iter().zip(consumer) {
            let CapturedTensorDigest::Tensor {
                cells: original, ..
            } = original
            else {
                panic!("feedback producer digest");
            };
            let CapturedTensorDigest::Tensor {
                cells: consumer,
                producer_sealed,
                ..
            } = consumer
            else {
                panic!("feedback alias digest");
            };
            assert!(
                Arc::ptr_eq(original, consumer),
                "alias capture must not allocate a replacement digest"
            );
            assert!(
                *producer_sealed,
                "alias capture must verify, never overwrite the producer seal"
            );
        }
        drop(witness);
        session.release(&mut lease, &[1]).unwrap();
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn retained_model_content_survives_reader_retirement_and_bank_reuse() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut lease = session.acquire().unwrap();
        let identity = session.published_identity(&lease).unwrap();
        let layout = session.publication.as_ref().unwrap().layouts[&(18, 0)];
        let live = Arc::new(allocate_publication::<u8>(&provider, 4).unwrap());
        upload_publication(&provider, &18.0f32.to_le_bytes(), &live).unwrap();
        let producer = || {
            content_test_export(
                Arc::clone(&live),
                provider.device().ordinal() as i32,
                crate::dlpack::DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                layout,
                (0, 0),
            )
        };
        let witness = session
            .bind_model_content(&lease, vec![producer()], 1)
            .unwrap();
        session
            .admit_transition(&lease, SemanticTransitionKind::Recompute)
            .unwrap();
        let feedback = session.published_feedback(&lease, 1).unwrap();
        assert!(session.release_published_reader(&mut lease, &[1]).is_err());
        let (continuation, continuation_witness) =
            publication_test_continuation(&mut session, &lease, &provider);
        session
            .bind_continuation(&lease, continuation, &continuation_witness, 1)
            .unwrap();
        drop(continuation_witness);
        session.capture().unwrap();
        session.launch().unwrap();
        session.observe(0).unwrap();
        let (_, bank_alias) = session.published_prefix(&lease, 1).unwrap();
        assert!(session.release_published_reader(&mut lease, &[1]).is_err());
        assert!(lease.is_active());
        drop(bank_alias);
        session.release_published_reader(&mut lease, &[1]).unwrap();
        assert!(!lease.is_active());
        assert_eq!(session.retained_step_identity(&lease).unwrap(), identity);
        assert!(session.published_source(&lease, 1).is_err());
        assert!(session.quiesce_published_reader(&lease, &[1]).is_err());
        assert!(session.release(&mut lease, &[1]).is_err());
        for _ in 0..2 {
            let mut current = session.acquire().unwrap();
            publication_test_recompute(&mut session, &current, &provider);
            session.release(&mut current, &[1]).unwrap();
        }
        let mut reused = session.acquire().unwrap();
        assert_eq!((reused.identity.word >> 1) - (identity.word >> 1), 3);
        session.release(&mut reused, &[]).unwrap();
        let before = session.host_io_stats();
        session
            .verify_tensor_content(&lease, &witness, vec![producer()], 1)
            .unwrap();
        assert_eq!(session.host_io_stats(), before);
        drop(witness);
        assert!(
            session.release(&mut lease, &[1]).is_err(),
            "original feedback exports retain their step"
        );
        drop(feedback);
        session.release(&mut lease, &[1]).unwrap();
        assert!(session.retained_step_identity(&lease).is_err());
        drop(live);
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn model_content_binding_uses_original_seals_and_retires_each_reader() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        for _ in 0..2 {
            let mut lease = session.acquire().unwrap();
            let layout = session.publication.as_ref().unwrap().layouts[&(18, 0)];
            let live = Arc::new(allocate_publication::<u8>(&provider, 4).unwrap());
            upload_publication(&provider, &18.0f32.to_le_bytes(), &live).unwrap();
            let producer = || {
                content_test_export(
                    Arc::clone(&live),
                    provider.device().ordinal() as i32,
                    crate::dlpack::DLDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    layout,
                    (0, 0),
                )
            };
            let original = producer();
            let same = producer();
            let substituted =
                publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
            let before = session.host_io_stats();
            let witness = session
                .bind_model_content(&lease, vec![original], 1)
                .unwrap();
            assert!(
                matches!(
                    session.steps[&lease.token].content[witness.index].seals,
                    TensorContentSeals::Model(_)
                ),
                "model binding must not capture a replacement baseline"
            );
            session
                .verify_tensor_content(&lease, &witness, vec![same], 1)
                .unwrap();
            assert_eq!(session.host_io_stats(), before);
            assert!(
                session
                    .verify_tensor_content(&lease, &witness, vec![substituted], 1)
                    .is_err(),
                "equal publication bytes cannot replace this invocation's original live storage"
            );
            assert!(session.release(&mut lease, &[1]).is_err());
            drop(witness);
            session.release(&mut lease, &[1]).unwrap();
            assert!(
                session.readers.is_empty(),
                "the first reader is not a permanent model baseline"
            );
        }
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
        assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn publication_content_guards_preserve_zero_transfer_and_release_parent_alias_witnesses() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut original = session.acquire().unwrap();
        let material = session.published_state_material(&original).unwrap();
        session.release(&mut original, &[]).unwrap();
        drop(session);
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session.restore_state_material(&material).unwrap();
        let mut lease = session.acquire().unwrap();
        let before_exports = session.host_io_stats();
        let sealed_prefix = *lease
            .directory
            .iter()
            .find(|range| range.role == 1)
            .unwrap();
        let (_, prefix) = session.published_prefix(&lease, 1).unwrap();
        let source = session.published_source(&lease, 1).unwrap();
        let context = session.published_model_context(&lease, 1).unwrap();
        assert!(matches!(
            session.published_model_context(&lease, 2),
            Err(SemanticTransitionError::InvalidInput { .. })
        ));
        let outside_alias = session
            .published_tensor(&lease, SemanticStateRole::SlowWeights, 0, 1)
            .unwrap();
        assert_eq!(
            session.host_io_stats(),
            before_exports,
            "native capacity exports must not copy metadata or wait on the host"
        );
        // SAFETY: the genuine managed export retains its complete descriptor
        // and shape array until prefix is dropped below. No device bytes read.
        let prefix_tensor = unsafe { &(*prefix.as_ptr()).dl_tensor };
        assert_eq!(prefix_tensor.ndim, 2);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(prefix_tensor.shape, 2) },
            [64, 8]
        );
        assert_eq!(
            (prefix_tensor.dtype.code, prefix_tensor.dtype.bits),
            (1, 64)
        );
        let storage = session.publication.as_ref().unwrap();
        let bank = storage.banks[(lease.identity.word & 1) as usize].device_ptr_value();
        for (tensor, field) in [
            (&source, PublicationBankField::Source),
            (&context.source, PublicationBankField::Source),
            (&context.prefix.1, PublicationBankField::PrefixExtent),
            (&context.ring_head, PublicationBankField::RingHead),
        ] {
            let (range, shape, strides) = field.layout();
            // SAFETY: every descriptor and metadata array belongs to the live
            // genuine managed export retained above; device bytes are not read.
            let tensor = unsafe { &(*tensor.as_ptr()).dl_tensor };
            assert_eq!(
                tensor.data as usize as u64 + tensor.byte_offset,
                bank + range.start as u64
            );
            assert_eq!((tensor.dtype.code, tensor.dtype.bits), (1, 64));
            assert_eq!(tensor.ndim as usize, shape.len());
            assert_eq!(
                unsafe { std::slice::from_raw_parts(tensor.shape, shape.len()) },
                shape
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(tensor.strides, strides.len()) },
                strides
            );
        }
        let prefix_range = lease
            .directory
            .iter()
            .find(|range| range.role == 3)
            .unwrap();
        let identity_tensor = unsafe { &(*context.prefix.0.as_ptr()).dl_tensor };
        assert_eq!(
            identity_tensor.data as usize as u64 + identity_tensor.byte_offset,
            storage.allocations[prefix_range.storage_slot as usize].device_ptr_value()
                + prefix_range.offset_bytes
        );
        assert_eq!(
            (
                identity_tensor.ndim,
                identity_tensor.dtype.code,
                identity_tensor.dtype.bits
            ),
            (1, 1, 8)
        );
        assert_eq!(unsafe { *identity_tensor.shape }, 32);
        assert_eq!(context.parent, lease.identity);
        assert_eq!(context.descriptor_digest, lease.header.descriptor_digest);
        let after_prefix = lease
            .directory
            .iter()
            .find(|range| range.role == 1)
            .unwrap();
        assert_eq!(
            (
                after_prefix.logical_begin,
                after_prefix.logical_end,
                after_prefix.length_bytes,
                after_prefix.digest
            ),
            (
                sealed_prefix.logical_begin,
                sealed_prefix.logical_end,
                sealed_prefix.length_bytes,
                sealed_prefix.digest
            )
        );
        drop(prefix);
        drop(source);
        drop(context);
        // Generic producer fixture setup is cold; the original content-guard
        // measurement below starts after those fixture uploads and handoffs.
        let mut captured = vec![publication_test_alias(
            &mut session,
            &lease,
            SemanticStateRole::SlowWeights,
            1,
        )];
        let mut verified = vec![publication_test_alias(
            &mut session,
            &lease,
            SemanticStateRole::SlowWeights,
            1,
        )];
        let mut owners = Vec::new();
        let cases = [
            (
                SemanticTensorLayout {
                    role: 0,
                    index: 0,
                    scalar_type: 7,
                    element_bytes: 8,
                    rank: 2,
                    logical_axis: u64::MAX,
                    dimensions: [1, 3, 0, 0],
                    strides_bytes: [24, 8, 0, 0],
                },
                crate::dlpack::DLDataType {
                    code: crate::dlpack::K_DLINT,
                    bits: 64,
                    lanes: 1,
                },
                [i64::MIN, -7, i64::MAX]
                    .into_iter()
                    .flat_map(i64::to_le_bytes)
                    .collect::<Vec<_>>(),
                (0, 0),
            ),
            (
                SemanticTensorLayout {
                    role: 0,
                    index: 0,
                    scalar_type: 8,
                    element_bytes: 1,
                    rank: 2,
                    logical_axis: u64::MAX,
                    dimensions: [1, 3, 0, 0],
                    strides_bytes: [3, 1, 0, 0],
                },
                crate::dlpack::DLDataType {
                    code: crate::dlpack::K_DLBOOL,
                    bits: 8,
                    lanes: 1,
                },
                vec![0, 1, 1],
                (0, 0),
            ),
            (
                SemanticTensorLayout {
                    role: 0,
                    index: 0,
                    scalar_type: 6,
                    element_bytes: 4,
                    rank: 2,
                    logical_axis: u64::MAX,
                    dimensions: [2, 3, 0, 0],
                    strides_bytes: [4, 8, 0, 0],
                },
                crate::dlpack::DLDataType {
                    code: crate::dlpack::K_DLFLOAT,
                    bits: 32,
                    lanes: 1,
                },
                [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                (0, 0),
            ),
            (
                SemanticTensorLayout {
                    role: 0,
                    index: 0,
                    scalar_type: 6,
                    element_bytes: 4,
                    rank: 2,
                    logical_axis: u64::MAX,
                    dimensions: [0, 3, 0, 0],
                    strides_bytes: [12, 4, 0, 0],
                },
                crate::dlpack::DLDataType {
                    code: crate::dlpack::K_DLFLOAT,
                    bits: 32,
                    lanes: 1,
                },
                Vec::new(),
                (0, 0),
            ),
        ];
        for (mut layout, dtype, bytes, interval) in cases {
            layout.index = captured.len() as u64;
            assert_eq!(tensor_layout_bytes(&layout).unwrap(), bytes.len());
            let owner = Arc::new(allocate_publication::<u8>(&provider, bytes.len()).unwrap());
            if !bytes.is_empty() {
                upload_publication(&provider, &bytes, &owner).unwrap();
            }
            let device = provider.device().ordinal() as i32;
            let rejected = content_test_export(Arc::clone(&owner), device, dtype, layout, interval);
            let mut invalid_parent = publication_test_parent(&provider);
            invalid_parent.tensors.push(rejected);
            let mut admission = publication_test_session(Arc::clone(&provider), &domain);
            assert!(matches!(admission.bind_parent(invalid_parent),
                Err(SemanticTransitionError::InvalidInput { detail })
                    if detail == "initial parent tensor role is duplicate, uncomputed-active, or outside its model roster"));
            drop(admission);
            captured.push(content_test_export(
                Arc::clone(&owner),
                device,
                dtype,
                layout,
                interval,
            ));
            verified.push(content_test_export(
                Arc::clone(&owner),
                device,
                dtype,
                layout,
                interval,
            ));
            owners.push(owner);
        }
        let before = session.host_io_stats();
        session
            .guard_published_content(
                &lease,
                &[
                    (SemanticStateRole::PrefixSource, 0),
                    (SemanticStateRole::SlowWeights, 0),
                ],
                1,
            )
            .unwrap();
        let witness = session.capture_tensor_content(&lease, captured, 1).unwrap();
        session
            .verify_tensor_content(&lease, &witness, verified, 1)
            .unwrap();
        assert_eq!(
            session.host_io_stats(),
            before,
            "content guards must add no host transfers, metadata readback or session stream waits"
        );
        // SAFETY: protocol stream one is this context's legacy default stream;
        // native guard events order its consumers. This is terminal test work.
        assert_eq!(
            unsafe { sys::cuStreamSynchronize(std::ptr::null_mut()) },
            sys::cudaError_enum::CUDA_SUCCESS
        );
        assert!(
            session.release(&mut lease, &[1]).is_err(),
            "a live witness retains its reader"
        );
        assert!(session.quiesce_published_reader(&lease, &[1]).is_err());
        drop(witness);
        assert!(
            session.release(&mut lease, &[1]).is_err(),
            "an external alias still retains its reader"
        );
        assert!(session.quiesce_published_reader(&lease, &[1]).is_err());
        drop(outside_alias);
        let identity = session.published_identity(&lease).unwrap();
        let publication = Arc::clone(session.publication.as_ref().unwrap());
        let readers = session
            .publication_read(publication.control.view())
            .unwrap()[0]
            .reader_counts;
        session.quiesce_published_reader(&lease, &[1]).unwrap();
        assert!(
            session.release_events.is_empty(),
            "quiescence must retire completed consumer events"
        );
        assert_eq!(session.published_identity(&lease).unwrap(), identity);
        assert!(lease.active);
        assert_eq!(
            session
                .publication_read(publication.control.view())
                .unwrap()[0]
                .reader_counts,
            readers,
            "quiescence must retain the actual acquired native reader"
        );
        session.release(&mut lease, &[1]).unwrap();
        drop(session);
        drop(owners);
        provider.memory().reap_pending_deallocations().unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run; each fatal case uses a fresh process"]
    fn mutated_content_stops_the_dependent_cuda_consumer() {
        const CHILD_KIND: &str = "XLOG_CONTENT_GUARD_TEST_CHILD";
        const COMPLETED: &str = "content mutation produced a fatal CUDA completion error";
        let Some(kind) = std::env::var_os(CHILD_KIND) else {
            let qualified = concat!(
                module_path!(),
                "::mutated_content_stops_the_dependent_cuda_consumer"
            );
            let test_name = qualified.split_once("::").unwrap().1;
            for kind in ["publication", "witness"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        test_name,
                        "--ignored",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(CHILD_KIND, kind)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{kind} child did not verify a CUDA trap:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains(COMPLETED),
                    "{kind} child did not reach the typed CUDA completion check"
                );
            }
            return;
        };
        assert!(kind == "publication" || kind == "witness");
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let lease = session.acquire().unwrap();
        let alias = session
            .published_tensor(&lease, SemanticStateRole::SlowWeights, 0, 1)
            .unwrap();
        let captured =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let unchanged =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let changed =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let dependent = allocate_publication::<u8>(&provider, 4).unwrap();
        session
            .guard_published_content(&lease, &[(SemanticStateRole::SlowWeights, 0)], 1)
            .unwrap();
        let witness = session
            .capture_tensor_content(&lease, vec![captured], 1)
            .unwrap();
        session
            .verify_tensor_content(&lease, &witness, vec![unchanged], 1)
            .unwrap();
        // SAFETY: fixture aliases retain a real four-byte FP32 allocation in
        // this context; synchronization proves the unmodified guards completed.
        assert_eq!(
            unsafe { sys::cuStreamSynchronize(std::ptr::null_mut()) },
            sys::cudaError_enum::CUDA_SUCCESS
        );
        // SAFETY: this still-owned managed export retains its valid DLPack
        // descriptor and allocation until the disposable child exits.
        let tensor = unsafe { &(*alias.as_ptr()).dl_tensor };
        assert_eq!(tensor.byte_offset, 0);
        let pointer = tensor.data as usize as u64;
        // SAFETY: this is an external consumer writing its genuine managed
        // alias after capture, on the same stream it passes to verification.
        assert_eq!(
            unsafe { sys::cuMemsetD8Async(pointer, 0, 4, std::ptr::null_mut()) },
            sys::cudaError_enum::CUDA_SUCCESS
        );
        let guarded = if kind == "publication" {
            session.guard_published_content(&lease, &[(SemanticStateRole::SlowWeights, 0)], 1)
        } else {
            session.verify_tensor_content(&lease, &witness, vec![changed], 1)
        };
        // A fatal asynchronous launch can already surface at event handoff.
        // Only the typed CUDA completion below counts as expected rejection.
        if let Err(error) = guarded {
            eprintln!("content guard handoff: {error}");
        }
        // SAFETY: both retained buffers have four bytes in this context. The
        // native guard orders this dependent consumer on its declared stream.
        let copied = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                dependent.device_ptr_value(),
                pointer,
                4,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(
            copied,
            sys::cudaError_enum::CUDA_SUCCESS
                | sys::cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION
                | sys::cudaError_enum::CUDA_ERROR_LAUNCH_FAILED
        ));
        // SAFETY: terminal completion is deliberately outside the hot guard
        // path and belongs only to this disposable child process's CUDA context.
        let completion = unsafe { sys::cuStreamSynchronize(std::ptr::null_mut()) };
        assert!(
            matches!(
                completion,
                sys::cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION
                    | sys::cudaError_enum::CUDA_ERROR_LAUNCH_FAILED
            ),
            "unexpected CUDA completion: {completion:?}"
        );
        println!("{COMPLETED}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        // The context is poisoned by design. Exit only after checking the
        // actual CUDA error; the parent never accepts a panic as trap evidence.
        std::process::exit(0);
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run; aborted live owners use a fresh process"]
    fn aborted_session_retains_step_after_reader_retirement() {
        const CHILD: &str = "XLOG_ABORT_RETAINED_STEP_TEST_CHILD";
        const COMPLETED: &str =
            "aborted retained step kept its original owners without a bank reader";
        if std::env::var_os(CHILD).is_none() {
            let qualified = concat!(
                module_path!(),
                "::aborted_session_retains_step_after_reader_retirement"
            );
            let test_name = qualified.split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    test_name,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "retained-step abort child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains(COMPLETED));
            return;
        }
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut lease = session.acquire().unwrap();
        let layout = session.publication.as_ref().unwrap().layouts[&(18, 0)];
        let live = Arc::new(allocate_publication::<u8>(&provider, 4).unwrap());
        upload_publication(&provider, &18.0f32.to_le_bytes(), &live).unwrap();
        let producer = || {
            content_test_export(
                Arc::clone(&live),
                provider.device().ordinal() as i32,
                crate::dlpack::DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                layout,
                (0, 0),
            )
        };
        let witness = session
            .bind_model_content(&lease, vec![producer()], 1)
            .unwrap();
        session.release_published_reader(&mut lease, &[1]).unwrap();
        assert!(session.readers.is_empty());
        let before = session.host_io_stats();
        session.abort();
        assert!(matches!(
            session.retained_step_identity(&lease),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.verify_tensor_content(&lease, &witness, vec![producer()], 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.release(&mut lease, &[1]),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert_eq!(session.host_io_stats(), before);
        let prior = UNRELEASED_PUBLICATIONS.with(|owners| owners.borrow().len());
        drop(session);
        UNRELEASED_PUBLICATIONS.with(|owners| {
            let owners = owners.borrow();
            assert_eq!(owners.len(), prior + 1);
            let retained = owners.last().unwrap();
            assert!(retained._readers.is_empty());
            let step = &retained._steps[&lease.token];
            assert_eq!(step.identity, Some(lease.identity));
            assert!(matches!(
                step.content[witness.index].seals,
                TensorContentSeals::Model(_)
            ));
            assert_eq!(
                step.content[witness.index].tensors[0].data,
                live.device_ptr_value()
            );
        });
        println!("{COMPLETED}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::process::exit(0);
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run; aborted live owners use a fresh process"]
    fn aborted_session_refuses_existing_publication_and_witness_owners() {
        const CHILD: &str = "XLOG_ABORT_SESSION_TEST_CHILD";
        const COMPLETED: &str = "aborted session rejected retained publication and witness reuse";
        if std::env::var_os(CHILD).is_none() {
            let qualified = concat!(
                module_path!(),
                "::aborted_session_refuses_existing_publication_and_witness_owners"
            );
            let test_name = qualified.split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    test_name,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "abort child did not verify unusable retained owners:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains(COMPLETED),
                "abort child did not finish the native Poisoned checks"
            );
            return;
        }
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let lease = session.acquire().unwrap();
        let _outside_alias = session
            .published_tensor(&lease, SemanticStateRole::SlowWeights, 0, 1)
            .unwrap();
        let captured =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let unchanged =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let refused_capture =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let refused_verify =
            publication_test_alias(&mut session, &lease, SemanticStateRole::SlowWeights, 1);
        let witness = session
            .capture_tensor_content(&lease, vec![captured], 1)
            .unwrap();
        session
            .verify_tensor_content(&lease, &witness, vec![unchanged], 1)
            .unwrap();
        let identity = session.published_identity(&lease).unwrap();
        assert!(!session.published_range_keys(&lease).unwrap().is_empty());
        // SAFETY: native events order this context's declared default consumer;
        // complete the genuine pre-abort witness before testing invalidation.
        assert_eq!(
            unsafe { sys::cuStreamSynchronize(std::ptr::null_mut()) },
            sys::cudaError_enum::CUDA_SUCCESS
        );
        let before = session.host_io_stats();
        session.abort();
        session.abort();
        assert!(session.is_poisoned());
        assert_eq!(
            session.host_io_stats(),
            before,
            "abort must not synchronize or transfer retained content"
        );
        assert!(
            session.publication.is_some(),
            "abort must retain live publication storage"
        );
        assert!(
            session.readers.contains_key(&lease.token),
            "abort must retain existing reader owners"
        );
        assert_eq!(
            lease.identity, identity,
            "an aborted Session retains its lease's historical identity"
        );
        assert!(matches!(
            session.acquire(),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.published_identity(&lease),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.published_range_keys(&lease),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.published_tensor(&lease, SemanticStateRole::SlowWeights, 0, 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.published_prefix(&lease, 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.guard_published_content(&lease, &[(SemanticStateRole::SlowWeights, 0)], 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.capture_tensor_content(&lease, vec![refused_capture], 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.verify_tensor_content(&lease, &witness, vec![refused_verify], 1),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.capture(),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.launch(),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(matches!(
            session.bind_task_evaluation(task_binding_tests::task_spec([0, 1, 1], vec![])),
            Err(SemanticTransitionError::Poisoned)
        ));
        assert!(session.is_poisoned());
        println!("{COMPLETED}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        // Abort deliberately keeps historical owners unusable and retained.
        // Isolate their teardown from every other test; a panic is not success.
        std::process::exit(0);
    }

    // Codec-only sample: it intentionally has no native publication authority.
    // Actual hashes and replayable graph content are checked by restore on GPU.
    fn publication_material_sample() -> PublicationMaterial {
        let mut counts = [1u64; 55];
        counts[15] = 2;
        let layouts: BTreeMap<_, _> = (1..=55)
            .filter(|&role| is_tensor_role(role))
            .map(|role| {
                (
                    (role, 0),
                    SemanticTensorLayout {
                        role,
                        index: 0,
                        scalar_type: 6,
                        element_bytes: 4,
                        rank: 2,
                        logical_axis: u64::MAX,
                        dimensions: [1, 1, 0, 0],
                        strides_bytes: [4, 4, 0, 0],
                    },
                )
            })
            .collect();
        let mut ranges = Vec::new();
        for (ordinal, &count) in counts.iter().enumerate() {
            for index in 0..count {
                let role = ordinal as u64 + 1;
                let bytes = if role == 55 {
                    tensor_table_bytes(&layouts.values().copied().collect::<Vec<_>>(), false, &[])
                        .unwrap()
                } else if role == 44 {
                    let mut bytes = vec![0; 104];
                    bytes.extend_from_slice(&[44, index as u8, 0x80, 0xff]);
                    bytes
                } else {
                    vec![role as u8, index as u8, 0x80, 0xff]
                };
                ranges.push(PublicationMaterialRange {
                    range: PublicationRange {
                        role,
                        index,
                        generation: 1,
                        length_bytes: bytes.len() as u64,
                        storage_slot: ranges.len() as u64,
                        ..PublicationRange::default()
                    },
                    capacity: bytes.len() + 128,
                    bytes,
                });
            }
        }
        let header = PublicationHeader {
            abi: 1,
            model_generation: 7,
            authority_generation: 3,
            sealed_epoch: 8,
            publication_word: 16,
            range_count: ranges.len() as u64,
            state_digest: Identity256::from_bytes([29; 32]),
            ..PublicationHeader::default()
        };
        let mut model_memory = ModelMemoryGeometry::default();
        let mut model_allocations = Vec::new();
        for range in ranges
            .iter_mut()
            .filter(|range| matches!(range.range.role, 18..=25))
        {
            let ordinal = model_allocations.len() as u64;
            let mut bytes = vec![0; range.capacity];
            bytes[..range.bytes.len()].copy_from_slice(&range.bytes);
            range.range.backing_digest = Identity256::from_bytes(Sha256::digest(&bytes).into());
            model_memory.allocation_bytes.push(bytes.len() as u64);
            model_memory.storages.push(SemanticModelStorage {
                allocation: ordinal,
                byte_offset: 0,
                span_bytes: bytes.len() as u64,
            });
            model_memory.views.push(SemanticModelView {
                role: range.range.role,
                index: range.range.index,
                storage: ordinal,
                byte_offset: 0,
            });
            model_allocations.push(bytes);
        }
        let mut material = PublicationMaterial {
            bank: PublicationBank {
                header,
                source: [SemanticTextSlot::default(); 32],
                state: DeviceState::default(),
                receipts: [SemanticTransitionReceipt::default(); COMPONENT_COUNT],
            },
            contract: PublicationContract {
                abi: 1,
                model_generation: 7,
                authority_generation: 3,
                window_capacity: 32,
                role_count: 55,
                range_capacity: ranges.len() as u64,
                terminal_token_count: 1,
                model_contract_layout: SemanticModelContractLayout {
                    schema_begin: 104,
                    schema_bytes: 4,
                    schema_digest_offset: 0,
                    generation_offset: 32,
                    numerical_digest_offset: 40,
                    identity_offset: 72,
                },
                ..PublicationContract::default()
            },
            role_counts: counts,
            terminals: vec![3],
            layouts,
            model_memory,
            model_allocations,
            ranges,
            graph: SemanticRootMaterial {
                records: SemanticAdmissionRecords {
                    predicates: vec![],
                    records: vec![],
                    supports: vec![],
                },
                symbols: vec![],
                insertions: vec![],
                digest: [0; 32],
                extents: [0; 3],
                admission_base_digest: [0; 32],
                admission_base_extents: [0; 3],
            },
        };
        publication_material_sample_runtime(&mut material);
        for range in &mut material.ranges {
            if !is_tensor_role(range.range.role) {
                range.range.digest = range.original_record_digest();
            }
        }
        material.bank.header.descriptor_digest = material.original_descriptor_digest();
        material
    }

    // Build the fixture's cold contract only after its allocation geometry is
    // chosen. This helper is not a repair operation on imported material.
    fn publication_material_sample_runtime(material: &mut PublicationMaterial) {
        material.bank.header.model_geometry_digest =
            material.model_memory.digest(&material.layouts).unwrap();
        let runtime = initial_runtime_contract_record(
            material.contract,
            &material.role_counts,
            &material.terminals,
            &material.layouts,
            material
                .ranges
                .iter()
                .filter(|item| item.range.role != 43)
                .map(|item| (item.range.role, item.range.index, item.capacity))
                .collect(),
            b"codec sample numerical mode",
            &material.model_memory,
        )
        .unwrap();
        let record = material
            .ranges
            .iter_mut()
            .find(|item| item.range.role == 43)
            .unwrap();
        record.capacity = runtime.capacity_bytes;
        record.bytes = runtime.bytes;
        record.range.length_bytes = record.bytes.len() as u64;
        record.range.digest = record.original_record_digest();
    }

    #[test]
    fn publication_material_codec_preserves_all_ranges_and_rejects_incomplete_content() {
        let material = publication_material_sample();
        let bytes = material.encode().unwrap();
        let decoded = PublicationMaterial::decode(&bytes).unwrap();
        assert_eq!(decoded.encode().unwrap(), bytes);
        assert_eq!(decoded.ranges.len(), material.ranges.len());
        for (saved, actual) in material.ranges.iter().zip(&decoded.ranges) {
            assert_eq!(saved.bytes, actual.bytes);
            assert_eq!(saved.capacity, actual.capacity);
        }
        for boundary in [
            0,
            25,
            26,
            29,
            30,
            61,
            62,
            100,
            bytes.len() / 2,
            bytes.len() - 1,
        ] {
            assert!(PublicationMaterial::decode(&bytes[..boundary]).is_err());
        }
        let mut changed = bytes.clone();
        changed.push(0);
        assert!(PublicationMaterial::decode(&changed).is_err());
        changed = bytes.clone();
        changed[30] ^= 1;
        assert!(PublicationMaterial::decode(&changed).is_err());
        let mut incomplete = publication_material_sample();
        incomplete.ranges.pop();
        assert!(incomplete.encode().is_err());
        let mut reordered = publication_material_sample();
        reordered.ranges.swap(0, 1);
        assert!(reordered.encode().is_err());
        let mut address = publication_material_sample();
        address.contract.terminal_tokens = 0x1000;
        assert!(address.encode().is_err());
        let mut mismatched = publication_material_sample();
        mismatched.layouts.get_mut(&(4, 0)).unwrap().scalar_type = 2;
        assert!(mismatched.encode().is_err());
    }

    #[test]
    fn publication_material_runtime_preserves_mode_and_exact_self_capacity() {
        let material = publication_material_sample();
        let bytes = material.encode().unwrap();
        let restored = PublicationMaterial::decode(&bytes).unwrap();
        assert_eq!(
            restored.model_numerical_mode().unwrap(),
            b"codec sample numerical mode"
        );
        let record = restored
            .ranges
            .iter()
            .find(|item| item.range.role == 43)
            .unwrap();
        assert_eq!(record.capacity, record.bytes.len());
        assert_eq!(restored.encode().unwrap(), bytes);
        let capacities = material
            .ranges
            .iter()
            .map(|item| (item.range.role, item.range.index, item.capacity))
            .collect::<Vec<_>>();
        let mut relocated = material.contract;
        relocated.terminal_tokens = 0x1020304050607080;
        relocated.role_counts = 0x8070605040302010;
        relocated.semantic_owner = 99;
        let original = runtime_contract_bytes(
            relocated,
            &material.role_counts,
            &material.terminals,
            &material.layouts,
            &capacities,
            material.model_numerical_mode().unwrap(),
            &material.model_memory,
        )
        .unwrap();
        assert_eq!(
            original, record.bytes,
            "runtime addresses and owner handles are not portable identity"
        );
        assert_ne!(
            runtime_contract_bytes(
                relocated,
                &material.role_counts,
                &material.terminals,
                &material.layouts,
                &capacities,
                b"different complete mode",
                &material.model_memory
            )
            .unwrap(),
            original
        );
        assert!(runtime_contract_bytes(
            relocated,
            &material.role_counts,
            &material.terminals,
            &material.layouts,
            &capacities,
            b"",
            &material.model_memory
        )
        .is_err());
        assert!(
            initial_runtime_contract_record(
                relocated,
                &material.role_counts,
                &material.terminals,
                &material.layouts,
                capacities,
                b"mode",
                &material.model_memory
            )
            .is_err(),
            "caller runtime entry cannot be inserted twice"
        );
    }

    #[test]
    fn publication_material_runtime_binds_logical_contract_and_terminals() {
        let mut changed = publication_material_sample();
        changed.contract.max_position += 1;
        assert!(
            changed.encode().is_err(),
            "runtime record must bind the actual position contract"
        );
        let mut changed = publication_material_sample();
        changed.terminals[0] += 1;
        assert!(
            changed.encode().is_err(),
            "runtime record must retain actual terminal values"
        );
    }

    #[test]
    fn publication_material_runtime_binds_allocation_capacity() {
        let mut changed = publication_material_sample();
        changed
            .ranges
            .iter_mut()
            .find(|range| range.range.role == 2)
            .unwrap()
            .capacity += 1;
        assert!(
            changed.encode().is_err(),
            "runtime record must bind actual cold capacity"
        );
    }

    #[test]
    fn publication_material_runtime_rejects_changed_original_layout_table() {
        let material = publication_material_sample();
        assert_eq!(
            material.model_numerical_mode().unwrap(),
            b"codec sample numerical mode"
        );
        let table_index = material
            .ranges
            .iter()
            .position(|item| item.range.role == 55)
            .unwrap();
        for byte in 0..material.ranges[table_index].bytes.len() {
            let mut changed = publication_material_sample();
            changed.ranges[table_index].bytes[byte] ^= 1;
            assert!(
                changed.model_numerical_mode().is_err(),
                "changed table byte {byte}"
            );
        }
    }

    #[test]
    fn publication_material_runtime_rejects_resealed_layout_disagreement() {
        let mut changed = publication_material_sample();
        let table = changed
            .ranges
            .iter_mut()
            .find(|item| item.range.role == 55)
            .unwrap();
        let (_, mut layouts, active) = decode_tensor_table(&table.bytes).unwrap();
        layouts[0].scalar_type = 2;
        table.bytes = tensor_table_bytes(&layouts, active.computed, &active.rows).unwrap();
        table.range.digest = table.original_record_digest();
        assert!(changed.model_numerical_mode().is_err());
    }

    #[test]
    fn publication_material_runtime_allows_sealed_active_table_changes() {
        let mut changed = publication_material_sample();
        let mode = changed.model_numerical_mode().unwrap().to_vec();
        let table = changed
            .ranges
            .iter_mut()
            .find(|item| item.range.role == 55)
            .unwrap();
        let (_, layouts, _) = decode_tensor_table(&table.bytes).unwrap();
        table.bytes = tensor_table_bytes(&layouts, true, &[]).unwrap();
        table.range.digest = table.original_record_digest();
        assert_eq!(changed.model_numerical_mode().unwrap(), mode);
    }

    #[test]
    fn publication_material_runtime_refuses_opaque_replacement_even_if_resealed() {
        let mut changed = publication_material_sample();
        let runtime = changed
            .ranges
            .iter_mut()
            .find(|range| range.range.role == 43)
            .unwrap();
        runtime.bytes.fill(0);
        runtime.range.digest = runtime.original_record_digest();
        changed.bank.header.descriptor_digest = changed.original_descriptor_digest();
        assert!(
            changed.encode().is_err(),
            "opaque caller bytes are not the native runtime contract"
        );
    }

    #[test]
    fn capacity_prefix_export_preserves_extent_without_narrowing_shape() {
        let mut range = PublicationRange {
            role: 1,
            index: 0,
            generation: 1,
            length_bytes: 128,
            logical_begin: 0,
            logical_end: 2,
            ..PublicationRange::default()
        };
        let layout = committed_prefix_layout(&range, 8).unwrap();
        assert_eq!(layout.dimensions, [8, 8, 0, 0]);
        assert_eq!(tensor_layout_bytes(&layout).unwrap(), 512);
        assert_eq!(
            (range.logical_begin, range.logical_end, range.length_bytes),
            (0, 2, 128)
        );
        range.logical_end = 0;
        range.length_bytes = 0;
        assert_eq!(
            committed_prefix_layout(&range, 8).unwrap().dimensions,
            [8, 8, 0, 0]
        );
        assert!(committed_prefix_layout(
            &PublicationRange {
                logical_end: 9,
                length_bytes: 576,
                ..range
            },
            8
        )
        .is_err());
    }

    #[test]
    fn capacity_export_stream_rejects_ambiguous_default_handles() {
        assert!(dlpack_consumer_stream(0).is_err());
        assert!(dlpack_consumer_stream(2).is_err());
        assert!(dlpack_consumer_stream(1).unwrap().is_null());
        // Resolving a handle does not assert that the stream exists. The
        // driver checks the concrete handle during the ordered handoff.
        assert_eq!(dlpack_consumer_stream(4096).unwrap() as usize, 4096);
    }

    #[test]
    fn capacity_export_bounds_keep_seal_and_allocation_limits_distinct() {
        let range = PublicationRange {
            role: 1,
            index: 0,
            generation: 1,
            offset_bytes: 64,
            length_bytes: 128,
            logical_begin: 0,
            logical_end: 2,
            ..PublicationRange::default()
        };
        let layout = SemanticTensorLayout {
            role: 1,
            index: 0,
            scalar_type: 3,
            element_bytes: 8,
            rank: 2,
            logical_axis: 0,
            dimensions: [8, 8, 0, 0],
            strides_bytes: [64, 8, 0, 0],
        };
        assert_eq!(
            publication_export_span(&range, &layout, 576).unwrap(),
            64..576
        );
        assert!(publication_export_span(&range, &layout, 575).is_err());
        assert_eq!(
            (range.length_bytes, range.logical_begin, range.logical_end),
            (128, 0, 2)
        );
        for wrong in [
            PublicationRange {
                generation: 2,
                ..range
            },
            PublicationRange {
                offset_bytes: 65,
                ..range
            },
            PublicationRange {
                offset_bytes: u64::MAX - 7,
                ..range
            },
            PublicationRange {
                length_bytes: 513,
                ..range
            },
            PublicationRange {
                logical_end: 9,
                ..range
            },
            PublicationRange { index: 1, ..range },
        ] {
            assert!(publication_export_span(&wrong, &layout, 576).is_err());
        }
        let tensor = SemanticTensorLayout {
            role: 4,
            rank: 4,
            logical_axis: 2,
            dimensions: [1, 2, 8, 2],
            strides_bytes: [128, 64, 8, 4],
            scalar_type: 6,
            element_bytes: 4,
            ..layout
        };
        let tensor_range = PublicationRange {
            role: 4,
            offset_bytes: 0,
            length_bytes: 128,
            ..range
        };
        assert_eq!(
            publication_export_span(&tensor_range, &tensor, 128).unwrap(),
            0..128
        );
        let model = SemanticTensorLayout {
            role: 18,
            logical_axis: u64::MAX,
            ..tensor
        };
        assert_eq!(
            publication_export_span(
                &PublicationRange {
                    role: 18,
                    logical_begin: 0,
                    logical_end: 0,
                    ..tensor_range
                },
                &model,
                128
            )
            .unwrap(),
            0..128
        );
        assert!(publication_export_span(
            &PublicationRange {
                role: 18,
                length_bytes: 16,
                logical_begin: 0,
                logical_end: 0,
                ..tensor_range
            },
            &model,
            128
        )
        .is_err());
    }

    #[test]
    fn committed_prefix_export_layout_preserves_native_rows_and_empty_extent() {
        let range = PublicationRange {
            role: SemanticStateRole::PrefixSource as u64,
            index: 0,
            generation: 1,
            offset_bytes: 64,
            length_bytes: 128,
            logical_begin: 0,
            logical_end: 2,
            ..PublicationRange::default()
        };
        let layout = committed_prefix_layout(&range, 2).unwrap();
        assert_eq!(
            (layout.scalar_type, layout.element_bytes, layout.rank),
            (3, 8, 2)
        );
        assert_eq!(layout.dimensions, [2, 8, 0, 0]);
        assert_eq!(layout.strides_bytes, [64, 8, 0, 0]);
        assert_eq!(tensor_layout_bytes(&layout).unwrap(), 128);
        let row = SemanticTextSlot {
            token: 71,
            logical_position: 1,
            kind: 1,
            provenance: 2,
            valid: 1,
            committed: 1,
            recomputed: 1,
            provenance_record: 19,
        };
        let words = publication_abi_bytes(&[row])
            .chunks_exact(8)
            .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(words, [71, 1, 1, 2, 1, 1, 1, 19]);
        let empty = committed_prefix_layout(
            &PublicationRange {
                length_bytes: 0,
                logical_end: 0,
                ..range
            },
            0,
        )
        .unwrap();
        assert_eq!(empty.dimensions, [0, 8, 0, 0]);
        assert_eq!(empty.strides_bytes, [64, 8, 0, 0]);
        assert_eq!(tensor_layout_bytes(&empty).unwrap(), 0);
    }

    #[test]
    fn committed_prefix_export_rejects_inconsistent_or_unaligned_extent() {
        let valid = PublicationRange {
            role: 1,
            index: 0,
            generation: 1,
            length_bytes: 128,
            logical_begin: 0,
            logical_end: 2,
            ..PublicationRange::default()
        };
        for wrong in [
            PublicationRange { role: 2, ..valid },
            PublicationRange { index: 1, ..valid },
            PublicationRange {
                offset_bytes: 1,
                ..valid
            },
            PublicationRange {
                logical_begin: 1,
                ..valid
            },
            PublicationRange {
                logical_end: 3,
                ..valid
            },
            PublicationRange {
                length_bytes: 64,
                ..valid
            },
            PublicationRange {
                length_bytes: 192,
                ..valid
            },
        ] {
            assert!(committed_prefix_layout(&wrong, 2).is_err());
        }
        assert!(committed_prefix_layout(&valid, u64::MAX).is_err());
        assert!(committed_prefix_layout(&valid, 0).is_err());
    }

    #[test]
    fn publication_material_decode_checks_original_descriptor_before_relocation() {
        let material = publication_material_sample();
        let original = material.encode().unwrap();
        assert!(PublicationMaterial::decode(&original).is_ok());

        for field in 0..4 {
            let mut changed = publication_material_sample();
            match field {
                0 => changed.bank.header.instance = Identity256::from_bytes([71; 32]),
                1 => changed.bank.header.recovered_instance = Identity256::from_bytes([72; 32]),
                2 => changed.bank.header.semantic_owner ^= 1,
                3 => changed.ranges[0].range.storage_slot ^= 1,
                _ => unreachable!(),
            }
            assert!(
                PublicationMaterial::decode(&changed.encode().unwrap()).is_err(),
                "cold decoding accepted changed original descriptor field {field}"
            );
        }
    }

    #[test]
    fn publication_material_decode_checks_original_record_bytes() {
        // These records contain origin coordinates omitted by normalized state
        // hashing. A saved descriptor is insufficient without their raw seals.
        for role in [2, 14, 15, 30, 33, 48] {
            let mut changed = publication_material_sample();
            let record = changed
                .ranges
                .iter_mut()
                .find(|item| item.range.role == role)
                .unwrap();
            record.bytes[0] ^= 1;
            assert!(
                PublicationMaterial::decode(&changed.encode().unwrap()).is_err(),
                "cold decoding accepted altered original record {role}"
            );
        }
        for word in [1, 5, 6, 39] {
            let mut changed = publication_material_sample();
            let record = changed
                .ranges
                .iter_mut()
                .find(|item| item.range.role == 33)
                .unwrap();
            record.bytes = publication_abi_bytes(&[AttemptReceipt::default()]);
            record.range.length_bytes = record.bytes.len() as u64;
            record.capacity = record.bytes.len();
            record.range.digest = record.original_record_digest();
            publication_material_sample_runtime(&mut changed);
            changed.bank.header.descriptor_digest = changed.original_descriptor_digest();
            assert!(PublicationMaterial::decode(&changed.encode().unwrap()).is_ok());
            changed
                .ranges
                .iter_mut()
                .find(|item| item.range.role == 33)
                .unwrap()
                .bytes[word * 8] ^= 1;
            assert!(
                PublicationMaterial::decode(&changed.encode().unwrap()).is_err(),
                "cold decoding accepted altered normalized-out attempt word {word}"
            );
        }
    }

    // Fixed native ABI records exercise the cold evidence parser. They are not
    // claimed to be a device execution or historical publication authority.
    fn replay_material_sample() -> (PublicationMaterial, PublicationReplayEvidence) {
        let mut predecessor = publication_material_sample();
        predecessor.bank.header.instance = Identity256::from_bytes([11; 32]);
        predecessor.bank.header.logical_digest = Identity256::from_bytes([12; 32]);
        for (role, bytes) in [
            (
                30,
                publication_abi_bytes(&[IntentQueueHeader {
                    abi: 1,
                    ..IntentQueueHeader::default()
                }]),
            ),
            (33, publication_abi_bytes(&[AttemptReceipt::default()])),
        ] {
            let item = predecessor
                .ranges
                .iter_mut()
                .find(|item| item.range.role == role)
                .unwrap();
            item.range.length_bytes = bytes.len() as u64;
            item.capacity = bytes.len();
            item.bytes = bytes;
        }
        // The sample's final allocation capacities precede its cold contract,
        // exactly as in initial publication. Imported evidence is never resealed.
        publication_material_sample_runtime(&mut predecessor);
        for item in &mut predecessor.ranges {
            if !is_tensor_role(item.range.role) {
                item.range.digest = item.original_record_digest();
            }
        }
        predecessor.bank.header.descriptor_digest = predecessor.original_descriptor_digest();
        let mut evidence = PublicationReplayEvidence {
            successor: SemanticPublishedIdentity {
                instance: predecessor.bank.header.instance,
                word: 19,
                logical_digest: Identity256::from_bytes([13; 32]),
                state_digest: Identity256::default(),
            },
            ranges: predecessor
                .ranges
                .iter()
                .filter(|item| REPLAY_EVIDENCE_ROLES.contains(&item.range.role))
                .cloned()
                .collect(),
        };
        let attempt = AttemptReceipt {
            abi: 1,
            instance: evidence.successor.instance,
            base_word: predecessor.bank.header.publication_word,
            next_word: evidence.successor.word,
            logical_digest: evidence.successor.logical_digest,
            action_receipts_digest: Identity256::from_bytes(Sha256::digest([]).into()),
            semantic_receipts_digest: Identity256::from_bytes([14; 32]),
            coverage_digest: Identity256::from_bytes([15; 32]),
            replay_head_digest: evidence.range(27).unwrap().logical_record_digest().unwrap(),
            intent_head_digest: evidence.range(30).unwrap().logical_record_digest().unwrap(),
            acknowledgement_head_digest: evidence
                .range(32)
                .unwrap()
                .logical_record_digest()
                .unwrap(),
            previous_attempt_digest: predecessor
                .ranges
                .iter()
                .find(|item| item.range.role == 33)
                .unwrap()
                .logical_record_digest()
                .unwrap(),
            ..AttemptReceipt::default()
        };
        evidence.ranges.last_mut().unwrap().bytes = publication_abi_bytes(&[attempt]);
        seal_replay_sample(&mut evidence);
        (predecessor, evidence)
    }

    fn seal_replay_sample(evidence: &mut PublicationReplayEvidence) {
        let item = evidence.ranges.last_mut().unwrap();
        let mut attempt = item.attempt().unwrap();
        attempt.receipt_digest = item.logical_record_digest().unwrap();
        item.bytes = publication_abi_bytes(&[attempt]);
        for item in &mut evidence.ranges {
            item.range.digest = item.original_record_digest();
        }
        evidence.successor.state_digest = evidence.state_digest().unwrap();
    }

    #[test]
    fn semantic_replay_material_requires_complete_linked_evidence() {
        let (predecessor, evidence) = replay_material_sample();
        let bytes = predecessor.encode().unwrap();
        let parse = |evidence: &[u8]| {
            SemanticReplayMaterial::decode(
                &bytes,
                evidence,
                predecessor.bank.header.logical_digest,
                evidence_successor_logical(),
            )
        };
        let decoded = parse(&evidence.encode().unwrap()).unwrap();
        assert_eq!(
            decoded.predecessor_identity().instance,
            predecessor.bank.header.instance
        );
        assert_eq!(decoded.successor_identity(), evidence.successor);
        assert_eq!(decoded.transition_kind(), SemanticTransitionKind::Recompute);
        assert!(parse(&bytes).is_err());
        assert!(parse(&[]).is_err());
        let mut truncated = evidence.encode().unwrap();
        truncated.pop();
        assert!(parse(&truncated).is_err());
        let mut trailing = evidence.encode().unwrap();
        trailing.push(0);
        assert!(parse(&trailing).is_err());
        for change in 0..5 {
            let (_, mut changed) = replay_material_sample();
            match change {
                0 => changed.successor.instance = Identity256::from_bytes([90; 32]),
                1 => changed.successor.word += 4,
                2 => changed.successor.state_digest = Identity256::default(),
                3 => {
                    changed.ranges.remove(0);
                }
                4 => {
                    let item = changed.ranges.last_mut().unwrap();
                    let mut attempt = item.attempt().unwrap();
                    attempt.previous_attempt_digest = Identity256::from_bytes([91; 32]);
                    item.bytes = publication_abi_bytes(&[attempt]);
                    seal_replay_sample(&mut changed);
                }
                _ => unreachable!(),
            }
            assert!(
                parse(&changed.encode().unwrap()).is_err(),
                "accepted altered native evidence {change}"
            );
        }
    }

    fn evidence_successor_logical() -> Identity256 {
        Identity256::from_bytes([13; 32])
    }

    #[test]
    fn semantic_replay_provenance_preserves_original_authority_and_decision() {
        let (predecessor, evidence) = replay_material_sample();
        let replay = SemanticReplayMaterial::decode(
            &predecessor.encode().unwrap(),
            &evidence.encode().unwrap(),
            predecessor.bank.header.logical_digest,
            evidence.successor.logical_digest,
        )
        .unwrap();
        let authority = predecessor
            .ranges
            .iter()
            .find(|item| item.range.role == 38)
            .unwrap()
            .clone();
        let mut decision = predecessor
            .ranges
            .iter()
            .find(|item| item.range.role == 39)
            .unwrap()
            .clone();
        decision.bytes[0] ^= 1;
        decision.range.digest = decision.original_record_digest();
        let mut provenance = PublicationReplayProvenance {
            predecessor: replay.predecessor_identity(),
            authority,
            decision,
        };
        let bytes = provenance.encode().unwrap();
        assert_eq!(
            replay.verify_provenance(&bytes).unwrap(),
            provenance.decision.bytes
        );
        assert!(replay.verify_provenance(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(replay.verify_provenance(&trailing).is_err());
        assert!(replay
            .verify_provenance(&predecessor.encode().unwrap())
            .is_err());
        provenance.predecessor.word += 2;
        assert!(replay
            .verify_provenance(&provenance.encode().unwrap())
            .is_err());
        provenance.predecessor = replay.predecessor_identity();
        provenance.authority.bytes[0] ^= 1;
        provenance.authority.range.digest = provenance.authority.original_record_digest();
        assert!(replay
            .verify_provenance(&provenance.encode().unwrap())
            .is_err());
        provenance.authority = predecessor
            .ranges
            .iter()
            .find(|item| item.range.role == 38)
            .unwrap()
            .clone();
        provenance.decision.bytes[0] ^= 1;
        assert!(replay
            .verify_provenance(&provenance.encode().unwrap())
            .is_err());
    }

    #[test]
    fn semantic_replay_material_normalizes_only_native_audit_words() {
        let (_, evidence) = replay_material_sample();
        let mut attempt = evidence.range(33).unwrap().clone();
        let original = attempt.logical_record_digest().unwrap();
        for word in [1, 4, 5, 6, 39, 42] {
            attempt.bytes[word * 8] ^= 1;
        }
        assert_eq!(attempt.logical_record_digest().unwrap(), original);
        assert_ne!(
            attempt.original_record_digest(),
            evidence.range(33).unwrap().original_record_digest()
        );
        attempt.bytes[7 * 8] ^= 1;
        assert_ne!(attempt.logical_record_digest().unwrap(), original);
        for role in [27, 28, 31, 32] {
            let mut item = evidence.range(role).unwrap().clone();
            let original = item.logical_record_digest().unwrap();
            item.bytes[0] ^= 1;
            assert_ne!(
                item.logical_record_digest().unwrap(),
                original,
                "normalized opaque role {role}"
            );
        }
    }

    fn replay_codebook_sample() -> PublicationMaterialRange {
        let mut words: Vec<u64> = (0..28).collect();
        words[8] = 25 * 8;
        words[9] = 3 * 8;
        let bytes = publication_abi_bytes(&words);
        let mut item = PublicationMaterialRange {
            range: PublicationRange {
                role: 48,
                generation: 1,
                length_bytes: bytes.len() as u64,
                ..PublicationRange::default()
            },
            capacity: bytes.len(),
            bytes,
        };
        item.range.digest = item.original_record_digest();
        item
    }

    #[test]
    fn semantic_replay_receipt_hashes_preserve_every_nonruntime_field() {
        let codebook = replay_codebook_sample();
        let receipts = [SemanticTransitionReceipt {
            catalogue_generation: CATALOGUE_GENERATION,
            admission_binding: Identity256::from_bytes([17; 32]),
            choice: 5,
            ..SemanticTransitionReceipt::default()
        }];
        let expected = publication_action_receipts_digest(&codebook, &receipts).unwrap();
        for word in 0..codebook.bytes.len() / 8 {
            if matches!(word, 8 | 9) {
                continue;
            }
            let mut changed = codebook.clone();
            changed.bytes[word * 8] ^= 1;
            let equal =
                publication_action_receipts_digest(&changed, &receipts).unwrap() == expected;
            assert_eq!(
                equal,
                matches!(word, 14..=16 | 21..=24),
                "codebook word {word}"
            );
        }
        let mut changed = receipts;
        changed[0].admission_binding = Identity256::from_bytes([18; 32]);
        assert_eq!(
            publication_action_receipts_digest(&codebook, &changed).unwrap(),
            expected
        );
        changed[0].choice += 1;
        assert_ne!(
            publication_action_receipts_digest(&codebook, &changed).unwrap(),
            expected
        );
        let mut legacy = receipts;
        legacy[0].catalogue_generation = 0;
        let legacy_digest = publication_action_receipts_digest(&codebook, &legacy).unwrap();
        legacy[0].admission_binding = Identity256::from_bytes([18; 32]);
        assert_ne!(
            publication_action_receipts_digest(&codebook, &legacy).unwrap(),
            legacy_digest
        );
        assert_eq!(
            publication_action_receipts_digest(&codebook, &[]).unwrap(),
            Identity256::from_bytes(Sha256::digest([]).into())
        );
        let semantic = [[0u64; 42]; 7];
        let expected = publication_semantic_receipts_digest(&semantic);
        for ordinal in 0..semantic.len() {
            for word in 0..42 {
                let mut changed = semantic;
                changed[ordinal][word] = 1;
                assert_eq!(
                    publication_semantic_receipts_digest(&changed) == expected,
                    matches!(word, 3..=12 | 34..=41),
                    "semantic receipt {ordinal} word {word}"
                );
            }
        }
        let header = IntentQueueHeader {
            abi: 1,
            count: 2,
            capacity: 2,
            ..IntentQueueHeader::default()
        };
        let mut bytes = publication_abi_bytes(&[header]);
        bytes.extend(publication_abi_bytes(&[IntentEntry::default(); 2]));
        let intent = PublicationMaterialRange {
            range: PublicationRange {
                role: 30,
                logical_end: 2,
                length_bytes: bytes.len() as u64,
                ..PublicationRange::default()
            },
            capacity: bytes.len(),
            bytes,
        };
        let expected = intent.logical_record_digest().unwrap();
        for ordinal in 0..2 {
            for word in 0..43 {
                let mut changed = intent.clone();
                changed.bytes[size_of::<IntentQueueHeader>()
                    + ordinal * size_of::<IntentEntry>()
                    + word * 8] ^= 1;
                assert_eq!(
                    changed.logical_record_digest().unwrap() == expected,
                    matches!(word, 30..=34),
                    "intent entry {ordinal} word {word}"
                );
            }
        }
    }

    #[test]
    fn semantic_replay_observed_attempt_checks_actual_receipts_and_coverage() {
        let (predecessor, mut evidence) = replay_material_sample();
        let mut bank = predecessor.bank;
        bank.header.instance = evidence.successor.instance;
        bank.header.base_word = predecessor.bank.header.publication_word;
        bank.header.publication_word = evidence.successor.word;
        bank.header.sealed_epoch = evidence.successor.word >> 1;
        bank.header.logical_digest = evidence.successor.logical_digest;
        bank.header.proposal = predecessor.bank.header.proposal + 1;
        bank.state.proposal = predecessor.bank.header.proposal;
        bank.state.next_proposal = bank.header.proposal;
        bank.state.blocks = COMPONENT_COUNT as u64;
        bank.receipts[0].catalogue_generation = CATALOGUE_GENERATION;
        let codebook = replay_codebook_sample();
        let mut completion = CompletionCoverage {
            abi: 1,
            instance: bank.header.instance,
            base_word: bank.header.base_word,
            ..CompletionCoverage::default()
        };
        let bytes = publication_abi_bytes(&[completion]);
        let mut coverage = PublicationMaterialRange {
            range: PublicationRange {
                role: 14,
                generation: 1,
                length_bytes: bytes.len() as u64,
                ..PublicationRange::default()
            },
            capacity: bytes.len(),
            bytes,
        };
        completion.receipt_digest = coverage.logical_record_digest().unwrap();
        coverage.bytes = publication_abi_bytes(&[completion]);
        coverage.range.digest = coverage.original_record_digest();
        let item = evidence.ranges.last_mut().unwrap();
        let mut attempt = item.attempt().unwrap();
        attempt.action_receipts_digest =
            publication_action_receipts_digest(&codebook, &bank.receipts).unwrap();
        attempt.semantic_receipts_digest =
            publication_semantic_receipts_digest(&bank.state.semantic_receipts);
        attempt.coverage_digest = coverage.logical_record_digest().unwrap();
        item.bytes = publication_abi_bytes(&[attempt]);
        seal_replay_sample(&mut evidence);
        bank.header.state_digest = evidence.successor.state_digest;
        assert_eq!(
            evidence.validate(&predecessor).unwrap(),
            SemanticTransitionKind::Proposal
        );
        evidence
            .validate_observed(&predecessor, &bank, &coverage, &codebook)
            .unwrap();

        let mut changed = bank;
        changed.receipts[0].choice += 1;
        assert!(evidence
            .validate_observed(&predecessor, &changed, &coverage, &codebook)
            .is_err());
        changed = bank;
        changed.state.semantic_receipts[0][13] += 1;
        assert!(evidence
            .validate_observed(&predecessor, &changed, &coverage, &codebook)
            .is_err());
        changed = bank;
        changed.state.semantic_receipts[0][3] += 1;
        evidence
            .validate_observed(&predecessor, &changed, &coverage, &codebook)
            .unwrap();
        changed = bank;
        changed.state.blocks = 0;
        assert!(evidence
            .validate_observed(&predecessor, &changed, &coverage, &codebook)
            .is_err());
        changed = bank;
        changed.header.base_word += 2;
        assert!(evidence
            .validate_observed(&predecessor, &changed, &coverage, &codebook)
            .is_err());
        let mut changed_coverage = coverage.clone();
        changed_coverage.bytes[6 * 8] ^= 1;
        changed_coverage.range.digest = changed_coverage.original_record_digest();
        assert!(evidence
            .validate_observed(&predecessor, &bank, &changed_coverage, &codebook)
            .is_err());
        let mut changed_codebook = codebook.clone();
        changed_codebook.bytes[25 * 8] ^= 1;
        changed_codebook.range.digest = changed_codebook.original_record_digest();
        assert!(evidence
            .validate_observed(&predecessor, &bank, &coverage, &changed_codebook)
            .is_err());
        let item = evidence.ranges.last_mut().unwrap();
        let mut attempt = item.attempt().unwrap();
        attempt.semantic_receipts_digest = Identity256::from_bytes([92; 32]);
        item.bytes = publication_abi_bytes(&[attempt]);
        seal_replay_sample(&mut evidence);
        bank.header.state_digest = evidence.successor.state_digest;
        evidence.validate(&predecessor).unwrap();
        assert!(evidence
            .validate_observed(&predecessor, &bank, &coverage, &codebook)
            .is_err());
    }

    #[test]
    fn publication_material_codebooks_relocate_only_runtime_coordinates() {
        let original: Vec<u64> = (0..48).collect();
        let mut actual = original.clone();
        for word in (14..=16).chain(21..=24) {
            actual[word] += 500;
        }
        let saved = publication_abi_bytes(&original);
        assert_eq!(
            relocate_publication_codebooks(&saved, &actual).unwrap(),
            publication_abi_bytes(&actual)
        );
        for word in (0..14).chain(17..=20).chain(25..48) {
            let mut corrupt = actual.clone();
            corrupt[word] += 1;
            assert!(
                relocate_publication_codebooks(&saved, &corrupt).is_err(),
                "logical word {word}"
            );
        }
        assert!(relocate_publication_codebooks(&saved[..saved.len() - 1], &actual).is_err());
        assert!(relocate_publication_codebooks(&saved, &actual[..47]).is_err());
        assert!(relocate_publication_codebooks(&[], &[]).is_err());
    }

    type StepInputGeometryFixture = (
        [Vec<PublicationRange>; 2],
        BTreeMap<(u64, u64), SemanticTensorLayout>,
        Vec<usize>,
    );

    fn step_input_geometry_fixture() -> StepInputGeometryFixture {
        let mut layouts = BTreeMap::new();
        let mut banks = [Vec::new(), Vec::new()];
        let mut sizes = Vec::new();
        for (role, index) in std::iter::once(1)
            .chain(3..=13)
            .chain(15..=17)
            .chain(std::iter::once(44))
            .flat_map(|role| (0..if role == 16 { 3 } else { 1 }).map(move |index| (role, index)))
        {
            let layout = if role == 1 {
                SemanticTensorLayout {
                    role,
                    index: 0,
                    scalar_type: 3,
                    element_bytes: 8,
                    rank: 2,
                    logical_axis: 0,
                    dimensions: [64, 8, 0, 0],
                    strides_bytes: [64, 8, 0, 0],
                }
            } else {
                canonical_tensor_layout(SemanticTensorLayout {
                    role,
                    index: 0,
                    scalar_type: if role == 7 { 5 } else { 6 },
                    element_bytes: if role == 7 { 2 } else { 4 },
                    rank: 4,
                    logical_axis: if role <= 5 { 2 } else { u64::MAX },
                    dimensions: [1, 2, 64, 3],
                    strides_bytes: [0; 4],
                })
                .unwrap()
            };
            if (4..=13).contains(&role) {
                layouts.insert((role, 0), layout);
            }
            let capacity = if role == 3 {
                32
            } else if role == 44 {
                108
            } else if role == 15 {
                2 * size_of::<RawFeedbackRecord>()
            } else if matches!(role, 16 | 17) {
                40 + index as usize * 8
            } else {
                tensor_layout_bytes(&layout).unwrap()
            };
            let shared = matches!(role, 1 | 16 | 17);
            let first_slot = sizes.len();
            sizes.push(capacity);
            if !shared {
                sizes.push(capacity);
            }
            for (bank, directory) in banks.iter_mut().enumerate() {
                directory.push(PublicationRange {
                    role,
                    index,
                    storage_slot: (first_slot + usize::from(!shared) * bank) as u64,
                    generation: 1,
                    length_bytes: if role == 1 {
                        64
                    } else if role == 4 || role == 5 {
                        24
                    } else {
                        capacity as u64
                    },
                    logical_end: u64::from(matches!(role, 1 | 4 | 5 | 15)),
                    ..PublicationRange::default()
                });
            }
        }
        (banks, layouts, sizes)
    }

    #[test]
    fn step_input_geometry_preserves_exact_roster_and_full_shared_capacity() {
        let (banks, layouts, sizes) = step_input_geometry_fixture();
        let plan = plan_step_inputs(&banks, &layouts, 64, &sizes).unwrap();
        assert_eq!(
            plan.iter()
                .map(|input| (input.role, input.index))
                .collect::<Vec<_>>(),
            std::iter::once((1, 0))
                .chain((3..=13).map(|role| (role, 0)))
                .chain([(15, 0), (16, 0), (16, 1), (17, 0), (44, 0)])
                .collect::<Vec<_>>()
        );
        assert_eq!(plan[0].banks[0].span.len(), 64 * size_of::<SourceSlot>());
        assert_eq!(plan[1].banks[0].span.len(), 32);
        for input in plan.iter().filter(|input| matches!(input.role, 4 | 5)) {
            assert_eq!(input.banks[0].span.len(), 2 * 64 * 3 * 4);
        }
        assert_eq!(size_of::<PublicationStepInput>(), 160);
    }

    #[test]
    fn step_input_geometry_retains_raw_feedback_and_every_statement_record() {
        let (banks, layouts, sizes) = step_input_geometry_fixture();
        let plan = plan_step_inputs(&banks, &layouts, 64, &sizes).unwrap();
        let records = plan
            .iter()
            .filter(|input| matches!(input.role, 15..=17))
            .collect::<Vec<_>>();
        assert_eq!(
            records
                .iter()
                .map(|input| (input.role, input.index, input.banks[0].span.len()))
                .collect::<Vec<_>>(),
            vec![(15, 0, 768), (16, 0, 40), (16, 1, 48), (17, 0, 40)]
        );
        assert!(records
            .iter()
            .all(|input| input.layout.scalar_type == 1 && input.layout.element_bytes == 1));
        let mut missing = banks.clone();
        missing[1].retain(|range| (range.role, range.index) != (16, 1));
        assert!(plan_step_inputs(&missing, &layouts, 64, &sizes).is_err());
        let mut gapped = banks.clone();
        for bank in &mut gapped {
            bank.iter_mut()
                .find(|range| (range.role, range.index) == (16, 1))
                .unwrap()
                .index = 2;
        }
        assert!(plan_step_inputs(&gapped, &layouts, 64, &sizes).is_err());
        let mut shared_raw = banks;
        let slot = shared_raw[0]
            .iter()
            .find(|range| range.role == 15)
            .unwrap()
            .storage_slot;
        shared_raw[1]
            .iter_mut()
            .find(|range| range.role == 15)
            .unwrap()
            .storage_slot = slot;
        assert!(plan_step_inputs(&shared_raw, &layouts, 64, &sizes).is_err());
    }

    #[test]
    fn feedback_record_material_preserves_original_seal_and_rejects_changed_content() {
        let mut original = PublicationMaterialRange {
            range: PublicationRange {
                role: 15,
                generation: 1,
                length_bytes: 768,
                logical_end: 1,
                ..PublicationRange::default()
            },
            capacity: 768,
            bytes: vec![0; 768],
        };
        original.range.digest = original.original_record_digest();
        let material = original.clone().into_feedback_record().unwrap();
        assert_eq!(material.identity, original.range.digest);
        assert_eq!(
            (
                material.logical_begin,
                material.logical_end,
                material.capacity_bytes
            ),
            (0, 1, 768)
        );
        assert_eq!(material.bytes, original.bytes);
        let mut changed = original.clone();
        changed.bytes[0] ^= 1;
        assert!(changed.into_feedback_record().is_err());
        let mut changed = original.clone();
        changed.range.logical_end = 2;
        assert!(changed.into_feedback_record().is_err());
        let mut changed = original;
        changed.range.role = 18;
        assert!(changed.into_feedback_record().is_err());
    }

    #[test]
    fn step_input_geometry_rejects_missing_duplicate_and_noncanonical_coordinates() {
        let (banks, layouts, sizes) = step_input_geometry_fixture();
        let mut missing = banks.clone();
        missing[1].remove(4);
        assert!(plan_step_inputs(&missing, &layouts, 64, &sizes).is_err());
        let mut duplicate = banks.clone();
        let repeated = duplicate[0][4];
        duplicate[0].push(repeated);
        assert!(plan_step_inputs(&duplicate, &layouts, 64, &sizes).is_err());
        let mut wrong_index = banks;
        wrong_index[0][4].index = 1;
        assert!(plan_step_inputs(&wrong_index, &layouts, 64, &sizes).is_err());
    }

    #[test]
    fn step_input_geometry_requires_shared_prefix_and_isolated_key_value_caches() {
        let (banks, layouts, sizes) = step_input_geometry_fixture();
        {
            let role = 1;
            let mut changed = banks.clone();
            changed[1]
                .iter_mut()
                .find(|range| range.role == role)
                .unwrap()
                .storage_slot += 1;
            assert!(plan_step_inputs(&changed, &layouts, 64, &sizes).is_err());
        }
        for role in [4, 5].into_iter().chain(std::iter::once(3)).chain(6..=13) {
            let mut changed = banks.clone();
            let slot = changed[0]
                .iter()
                .find(|range| range.role == role)
                .unwrap()
                .storage_slot;
            changed[1]
                .iter_mut()
                .find(|range| range.role == role)
                .unwrap()
                .storage_slot = slot;
            assert!(plan_step_inputs(&changed, &layouts, 64, &sizes).is_err());
        }
    }

    #[test]
    fn step_input_geometry_checks_canonical_dtype_spans_and_allocation_bounds() {
        let (banks, layouts, sizes) = step_input_geometry_fixture();
        let plan = plan_step_inputs(&banks, &layouts, 64, &sizes).unwrap();
        let bf16 = plan.iter().find(|input| input.role == 7).unwrap();
        assert_eq!(
            (
                bf16.layout.scalar_type,
                bf16.layout.element_bytes,
                bf16.banks[0].span.len()
            ),
            (5, 2, 768)
        );
        let mut wrong = layouts.clone();
        wrong.get_mut(&(7, 0)).unwrap().strides_bytes[2] += 2;
        assert!(plan_step_inputs(&banks, &wrong, 64, &sizes).is_err());
        let mut short = sizes;
        short[banks[1]
            .iter()
            .find(|range| range.role == 7)
            .unwrap()
            .storage_slot as usize] -= 1;
        assert!(plan_step_inputs(&banks, &layouts, 64, &short).is_err());
        assert!(
            step_input_overlap(104, 0, 100, 8).unwrap(),
            "empty native aliases must still preserve their origin"
        );
        assert!(!step_input_overlap(108, 0, 100, 8).unwrap());
        assert!(step_input_overlap(96, 8, 100, 8).unwrap());
        assert!(step_input_overlap(u64::MAX, 2, 100, 8).is_err());
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn step_inputs_keep_private_values_after_reader_retirement_and_current_bank_reuse() {
        let (provider, domain) = publication_test_provider();
        let mut session = publication_test_session(Arc::clone(&provider), &domain);
        session
            .bind_parent(publication_test_parent(&provider))
            .unwrap();
        let mut lease = session.acquire().unwrap();
        let context = session.published_model_context(&lease, 1).unwrap();
        let cache = session
            .published_tensor(&lease, SemanticStateRole::from_code(8).unwrap(), 0, 1)
            .unwrap();
        let inputs = Arc::clone(session.steps[&lease.token].inputs.as_ref().unwrap());
        let mut feedback_before = Vec::new();
        for (&(role, index), view) in inputs
            .views
            .first()
            .expect("two publication input banks")
            .iter()
            .filter(|(key, _)| matches!(key.0, 15..=17))
        {
            feedback_before.push((
                (role, index),
                session.publication_read(view.clone()).unwrap(),
            ));
        }
        assert_eq!(feedback_before.len(), 4);
        drop(inputs);
        let source_pointer = context.source.as_ptr();
        let cache_pointer = cache.as_ptr();
        let source_address = unsafe { (*source_pointer).dl_tensor.data as u64 };
        let cache_address = unsafe { (*cache_pointer).dl_tensor.data as u64 };
        let storage = Arc::clone(session.publication.as_ref().unwrap());
        assert!(!storage
            .banks
            .iter()
            .any(|bank| source_address >= bank.device_ptr_value()
                && source_address < bank.device_ptr_value() + size_of::<PublicationBank>() as u64));
        assert!(!storage
            .allocations
            .iter()
            .any(|allocation| cache_address >= allocation.device_ptr_value()
                && cache_address < allocation.device_ptr_value() + allocation.len() as u64));
        let again = session.published_source(&lease, 1).unwrap();
        assert_eq!(
            unsafe { (*again.as_ptr()).dl_tensor.data as u64 },
            source_address
        );
        let source_layout = SemanticTensorLayout {
            role: 0,
            index: 0,
            scalar_type: 3,
            element_bytes: 8,
            rank: 2,
            logical_axis: u64::MAX,
            dimensions: [32, 8, 0, 0],
            strides_bytes: [64, 8, 0, 0],
        };
        let source_input = |tensor| SemanticTensorInput {
            layout: source_layout,
            logical_begin: 0,
            logical_end: 0,
            tensor,
            native_allocation: None,
        };
        let original_source = source_input(session.published_source(&lease, 1).unwrap());
        let original_cache = publication_test_alias(
            &mut session,
            &lease,
            SemanticStateRole::from_code(8).unwrap(),
            1,
        );
        let later_source = source_input(session.published_source(&lease, 1).unwrap());
        let later_cache = publication_test_alias(
            &mut session,
            &lease,
            SemanticStateRole::from_code(8).unwrap(),
            1,
        );
        let witness = session
            .capture_tensor_content(&lease, vec![original_source, original_cache], 1)
            .unwrap();
        let TensorContentSeals::Captured(origins) =
            &session.steps[&lease.token].content[witness.index].seals
        else {
            panic!("native input origins");
        };
        assert!(
            origins
                .iter()
                .all(|origin| matches!(origin, CapturedTensorDigest::Publication(_))),
            "capture must retain the original native input seals"
        );
        session
            .guard_published_content(&lease, &[(SemanticStateRole::from_code(8).unwrap(), 0)], 1)
            .unwrap();
        publication_test_recompute(&mut session, &lease, &provider);
        session.release_published_reader(&mut lease, &[1]).unwrap();
        for _ in 0..2 {
            let mut current = session.acquire().unwrap();
            let current_context = session.published_model_context(&current, 1).unwrap();
            let inputs = Arc::clone(session.steps[&current.token].inputs.as_ref().unwrap());
            let copied_header = session.publication_read(inputs.header.view()).unwrap()[0];
            assert_eq!(copied_header.publication_word, current.identity.word);
            assert_ne!(
                unsafe { (*current_context.source.as_ptr()).dl_tensor.data as u64 },
                source_address
            );
            drop(current_context);
            publication_test_recompute(&mut session, &current, &provider);
            session.release(&mut current, &[1]).unwrap();
        }
        let inputs = Arc::clone(session.steps[&lease.token].inputs.as_ref().unwrap());
        assert_eq!(
            session.publication_read(inputs.header.view()).unwrap()[0].publication_word,
            lease.identity.word
        );
        for (key, bytes) in feedback_before {
            assert_eq!(session.publication_read(inputs.views[0][&key].clone()).unwrap(), bytes,
                "raw feedback and its original statements/provenance must survive physical bank reuse");
        }
        let before = session.host_io_stats();
        session
            .verify_tensor_content(&lease, &witness, vec![later_source, later_cache], 1)
            .unwrap();
        assert_eq!(session.host_io_stats(), before);
        drop(witness);
        assert!(session.release(&mut lease, &[1]).is_err());
        drop(context);
        drop(cache);
        drop(again);
        drop(storage);
        drop(inputs);
        session.release(&mut lease, &[1]).unwrap();
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
    }

    #[test]
    fn model_context_joins_exact_acquired_header_prefix_and_ring() {
        let identity = |byte| Identity256::from_bytes([byte; 32]);
        let mut lease = SemanticPublishedLease {
            issuer: Arc::new(()),
            token: 7,
            active: true,
            identity: SemanticPublishedIdentity {
                instance: identity(1),
                word: 12,
                logical_digest: identity(2),
                state_digest: identity(3),
            },
            header: PublicationHeader {
                abi: 1,
                instance: identity(1),
                publication_word: 12,
                sealed_epoch: 6,
                logical_digest: identity(2),
                state_digest: identity(3),
                descriptor_digest: identity(4),
                semantic_owner: 5,
                semantic_slot: 6,
                semantic_generation: 7,
                semantic_digest: identity(8),
                semantic_extents: [9, 10, 11],
                prefix_extent: 2,
                ring_head: 17,
                model_generation: 12,
                neural_bank: 1,
                neural_generation: 13,
                cache_generation: 14,
                authority_generation: 15,
                ..PublicationHeader::default()
            },
            source: source_rows(),
            directory: Vec::new(),
        };
        let context = lease.model_context_header().unwrap();
        assert_eq!(context.descriptor_digest, identity(4));
        assert_eq!(
            (
                context.semantic_owner,
                context.semantic_slot,
                context.semantic_generation,
                context.semantic_digest,
                context.semantic_extents
            ),
            (5, 6, 7, identity(8), [9, 10, 11])
        );
        assert_eq!(context.ring_head, 17);
        assert_eq!(context.prefix_extent, 2);
        assert_eq!(
            (
                context.model_generation,
                context.neural_bank,
                context.neural_generation,
                context.cache_generation,
                context.authority_generation
            ),
            (12, 1, 13, 14, 15)
        );
        lease.header.state_digest = identity(19);
        assert!(lease.model_context_header().is_err());
    }

    #[test]
    fn publication_bank_views_use_original_fields_and_slot_order() {
        let header = PublicationHeader {
            prefix_extent: 23,
            ring_head: 17,
            ..PublicationHeader::default()
        };
        let source = source_rows();
        let mut bytes = publication_abi_bytes(&[header]);
        bytes.extend(publication_abi_bytes(&source));
        let (range, shape, strides) = PublicationBankField::Source.layout();
        assert_eq!(range.start, std::mem::offset_of!(PublicationBank, source));
        assert_eq!(range.len(), size_of::<[SourceSlot; 32]>());
        assert_eq!((shape, strides), (vec![32, 8], vec![8, 1]));
        assert_eq!(&bytes[range], publication_abi_bytes(&source));
        for (field, value) in [
            (PublicationBankField::PrefixExtent, 23u64),
            (PublicationBankField::RingHead, 17u64),
        ] {
            let (range, shape, strides) = field.layout();
            assert_eq!(range.start % size_of::<u64>(), 0);
            assert_eq!(range.len(), size_of::<u64>());
            assert_eq!((shape, strides), (vec![1], vec![1]));
            assert_eq!(&bytes[range], value.to_ne_bytes());
        }
    }

    #[test]
    fn active_row_table_requires_independent_publication_banks() {
        assert!(publication_mutable_role(
            SemanticStateRole::TensorLayout as u64
        ));
    }

    #[test]
    fn active_text_rows_include_the_reserved_window_before_feedback_positions() {
        for position in [63, 64, 95, 96] {
            let mut source = [SourceSlot::default(); 32];
            source[7] = SourceSlot {
                token: MASK_TOKEN as u64,
                logical_position: position,
                kind: 2,
                valid: 1,
                ..SourceSlot::default()
            };
            let admitted = validate_text_parent(&source, 0, 64, 2, 262144, 0, 0);
            assert_eq!(admitted.is_ok(), position < 96);
        }
    }

    #[test]
    fn active_table_keeps_computed_empty_distinct_from_uncomputed() {
        let layouts = [SemanticTensorLayout {
            role: 51,
            rank: 4,
            logical_axis: 2,
            element_bytes: 2,
            scalar_type: 4,
            dimensions: [1, 2, 35, 4],
            strides_bytes: [560, 280, 8, 2],
            ..SemanticTensorLayout::default()
        }];
        let cold = tensor_table_bytes(&layouts, false, &[]).unwrap();
        let empty = tensor_table_bytes(&layouts, true, &[]).unwrap();
        assert_ne!(cold, empty);
        assert_eq!(
            decode_tensor_table(&cold).unwrap().2,
            SemanticActiveRows {
                computed: false,
                rows: vec![]
            }
        );
        assert_eq!(
            decode_tensor_table(&empty).unwrap().2,
            SemanticActiveRows {
                computed: true,
                rows: vec![]
            }
        );
        assert_eq!(decode_tensor_table(&empty).unwrap().1, layouts);
        assert!(decode_tensor_table(&publication_abi_bytes(&layouts)).is_err());
        let mut invalid = empty.clone();
        invalid[16..24].copy_from_slice(&2u64.to_ne_bytes());
        assert!(decode_tensor_table(&invalid).is_err());
        invalid = empty.clone();
        invalid[24..32].copy_from_slice(&1u64.to_ne_bytes());
        assert!(decode_tensor_table(&invalid).is_err());
        assert!(decode_tensor_table(&empty[..empty.len() - 1]).is_err());
        let row = SemanticActiveRow {
            physical_row: 0,
            source_slot: 0,
            logical_position: 0,
            kind: 2,
        };
        assert!(tensor_table_bytes(&layouts, false, &[row]).is_err());
        assert!(tensor_table_capacity(usize::MAX, 35).is_err());
    }

    #[test]
    fn active_tensor_geometry_preserves_full_physical_capacity() {
        let producer = SemanticTensorLayout {
            role: 51,
            rank: 4,
            logical_axis: 2,
            element_bytes: 2,
            scalar_type: 4,
            dimensions: [1, 2, 35, 4],
            strides_bytes: [560, 2, 16, 4],
            ..SemanticTensorLayout::default()
        };
        let cold = canonical_tensor_layout(producer).unwrap();
        assert!(validate_active_capacity(&cold, 35).is_ok());
        assert!(validate_active_capacity(&cold, 34).is_err());
        let pending = continuation_tensor_layout(&producer, 0, 35, &cold).unwrap();
        assert_eq!(pending.dimensions, producer.dimensions);
        assert_eq!(pending.strides_bytes, [560, 280, 8, 2]);
        assert!(tensor_copy_plan(&producer, &pending).is_ok());
        assert!(continuation_tensor_layout(&producer, 8, 9, &cold).is_err());
        assert!(continuation_tensor_layout(&producer, 0, 3, &cold).is_err());
        let linear = SemanticTensorLayout {
            role: 53,
            rank: 3,
            logical_axis: 0,
            element_bytes: 4,
            scalar_type: 6,
            dimensions: [35, 4, 2, 0],
            strides_bytes: [32, 8, 4, 0],
            ..SemanticTensorLayout::default()
        };
        assert!(validate_active_capacity(&linear, 35).is_ok());
        assert!(validate_active_capacity(
            &SemanticTensorLayout {
                logical_axis: u64::MAX,
                ..linear
            },
            35
        )
        .is_err());
    }

    #[test]
    fn publication_rng_binding_uses_only_the_actual_header_coordinates() {
        let header = PublicationHeader {
            model_generation: 17,
            family_id: 29,
            stream_serial: (1u64 << 56) - 1,
            proposal: u64::from(u32::MAX),
            ..PublicationHeader::default()
        };
        assert_eq!(
            header.rng_binding().unwrap(),
            SemanticRngBinding {
                model_generation: 17,
                family_id: 29,
                stream_serial: (1u64 << 56) - 1,
                proposal: u32::MAX,
            }
        );
        for invalid in [
            PublicationHeader {
                model_generation: 1u64 << 32,
                ..header
            },
            PublicationHeader {
                family_id: 256,
                ..header
            },
            PublicationHeader {
                stream_serial: 1u64 << 56,
                ..header
            },
            PublicationHeader {
                proposal: 1u64 << 32,
                ..header
            },
        ] {
            assert!(invalid.rng_binding().is_err());
        }
    }

    #[test]
    fn cold_continuation_layouts_match_original_fixed_output_snapshots() {
        let prefix = canonical_tensor_layout(SemanticTensorLayout {
            role: 4,
            rank: 4,
            logical_axis: 2,
            element_bytes: 2,
            scalar_type: 4,
            dimensions: [1, 2, 128, 4],
            ..SemanticTensorLayout::default()
        })
        .unwrap();
        let active = canonical_tensor_layout(SemanticTensorLayout {
            role: 51,
            rank: 4,
            logical_axis: 2,
            element_bytes: 2,
            scalar_type: 4,
            dimensions: [1, 2, 35, 4],
            ..SemanticTensorLayout::default()
        })
        .unwrap();
        let state = canonical_tensor_layout(SemanticTensorLayout {
            role: 6,
            rank: 2,
            logical_axis: u64::MAX,
            element_bytes: 4,
            scalar_type: 6,
            dimensions: [4, 128, 0, 0],
            ..SemanticTensorLayout::default()
        })
        .unwrap();
        let mut layouts = Vec::new();
        for (model, extent) in [(prefix, 32), (active, 35), (state, 0)] {
            let cold = continuation_capacity_layout(model).unwrap();
            let mut producer = cold;
            if model.role != 6 {
                // The original can be transposed; only the owned destination
                // has the immutable canonical transport strides.
                producer.strides_bytes = [extent * 16, 2, 16, 4];
            }
            assert_eq!(
                continuation_tensor_layout(&producer, 0, extent, &model).unwrap(),
                cold
            );
            assert!(tensor_copy_plan(&producer, &cold).is_ok());
            layouts.push(cold);
        }
        assert_eq!(layouts[0].dimensions, [1, 2, 32, 4]);
        assert_eq!(layouts[1], active);
        assert_eq!(layouts[2], state);
        let table = tensor_table_bytes(&layouts, true, &[]).unwrap();
        let (_, decoded, rows) = decode_tensor_table(&table).unwrap();
        assert_eq!(decoded, layouts);
        assert_eq!(
            rows,
            SemanticActiveRows {
                computed: true,
                rows: vec![]
            }
        );
        assert!(continuation_capacity_layout(SemanticTensorLayout { role: 3, ..state }).is_err());
        assert!(continuation_capacity_layout(SemanticTensorLayout {
            logical_axis: 1,
            ..active
        })
        .is_err());
    }

    #[test]
    fn continuation_services_preserve_original_device_roster_and_capacity() {
        let layouts = continuation_service_layouts(9, 66).unwrap();
        let expected = [
            (3, 8, [32, 2, 0, 0]),
            (3, 8, [1, 0, 0, 0]),
            (8, 1, [32, 0, 0, 0]),
            (3, 8, [66, 4, 0, 0]),
            (3, 8, [1, 0, 0, 0]),
            (8, 1, [1, 0, 0, 0]),
        ];
        for (ordinal, (layout, (scalar_type, element_bytes, dimensions))) in
            layouts.iter().zip(expected).enumerate()
        {
            assert_eq!((layout.role, layout.index), (0, 9 + ordinal as u64));
            assert_eq!(
                (layout.scalar_type, layout.element_bytes, layout.dimensions),
                (scalar_type, element_bytes, dimensions)
            );
            assert_eq!(layout.logical_axis, u64::MAX);
            let bytes = tensor_layout_bytes(layout).unwrap();
            let range = tensor_content_range(layout, 0, 0, bytes).unwrap();
            assert_eq!(range.index, layout.index);
        }
        assert!(continuation_service_layouts(usize::MAX, 66).is_err());
        assert!(continuation_service_layouts(0, 0).is_err());
    }

    #[test]
    fn continuation_transport_does_not_rebase_original_suffix_or_nonpositional_outputs() {
        let suffix = canonical_tensor_layout(SemanticTensorLayout {
            role: 4,
            rank: 4,
            logical_axis: 2,
            element_bytes: 2,
            scalar_type: 4,
            dimensions: [1, 2, 32, 4],
            ..SemanticTensorLayout::default()
        })
        .unwrap();
        let cold = prefix_capacity_layout(
            SemanticTensorLayout {
                dimensions: [1, 2, 0, 4],
                ..suffix
            },
            16,
        )
        .unwrap();
        for (prefix, end) in [(0, 0), (3, 5), (16, 16)] {
            let pending = continuation_tensor_layout(&suffix, 0, 32, &cold).unwrap();
            assert_eq!(pending, suffix);
            assert!(continuation_tensor_layout(&suffix, prefix, end, &cold).is_err());
        }
        let state = canonical_tensor_layout(SemanticTensorLayout {
            role: 6,
            rank: 2,
            logical_axis: u64::MAX,
            element_bytes: 4,
            scalar_type: 6,
            dimensions: [4, 128, 0, 0],
            ..SemanticTensorLayout::default()
        })
        .unwrap();
        assert_eq!(
            continuation_tensor_layout(&state, 0, 0, &state).unwrap(),
            state
        );
        assert!(continuation_tensor_layout(&state, 3, 5, &state).is_err());
    }

    #[test]
    fn cold_intent_owner_constructs_an_empty_queue_with_the_actual_effect() {
        let effect = b"deliver the committed token sequence to the request output";
        let records = initial_intent_records(effect, 2, 4096, 1024).unwrap();
        assert_eq!(records[0].role, SemanticStateRole::IntentEntries);
        assert_eq!(records[0].bytes.len(), size_of::<IntentQueueHeader>());
        assert_eq!(
            records[0].capacity_bytes,
            size_of::<IntentQueueHeader>() + 2 * size_of::<IntentEntry>()
        );
        // SAFETY: the production builder returned a complete padding-free header.
        let header = unsafe {
            records[0]
                .bytes
                .as_ptr()
                .cast::<IntentQueueHeader>()
                .read_unaligned()
        };
        assert_eq!((header.abi, header.count, header.capacity), (1, 0, 2));
        assert_eq!(header.chain_head, Identity256::default());
        assert_eq!(
            (header.effect_offset_bytes, header.effect_length_bytes),
            (0, effect.len() as u64)
        );
        assert_eq!(header.payload_used_bytes, effect.len() as u64);
        assert_eq!(header.payload_capacity_bytes, 4096);
        assert_eq!(records[1].role, SemanticStateRole::IntentPayload);
        assert_eq!(records[1].bytes, effect);
        assert_eq!(records[1].capacity_bytes, 4096);
        assert!(initial_intent_records(&[], 2, 4096, 1024).is_err());
        assert!(initial_intent_records(effect, 0, 4096, 1024).is_err());
        assert!(initial_intent_records(effect, u64::MAX, 4096, 1024).is_err());
        assert!(initial_intent_records(effect, 2, effect.len() + 1023, 1024).is_err());
    }

    #[test]
    fn semantic_tensor_admission_preserves_original_type_strides_and_bounds() {
        let layout = SemanticTensorLayout {
            role: SemanticStateRole::AttentionKeys as u64,
            index: 0,
            element_bytes: 2,
            scalar_type: 4,
            rank: 4,
            logical_axis: 2,
            dimensions: [1, 2, 3, 4],
            strides_bytes: [48, 24, 8, 2],
        };
        let metadata = TensorMetadata {
            device_type: crate::dlpack::K_DLCUDA,
            device_id: 0,
            dtype: crate::dlpack::DLDataType {
                code: 2,
                bits: 16,
                lanes: 1,
            },
            shape: &[1, 2, 3, 4],
            strides: None,
            data: 0x1000,
            byte_offset: 2,
        };
        assert_eq!(
            validate_tensor_metadata(&layout, &metadata, 0).unwrap(),
            (0x1002, 48)
        );
        let mut wrong = metadata;
        wrong.dtype.code = 4; // BF16 is not F16 despite equal element width.
        assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        wrong = metadata;
        wrong.strides = Some(&[0, 0, 0, 0]);
        assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        wrong = metadata;
        wrong.device_id = 1;
        assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        wrong = metadata;
        wrong.data = u64::MAX - 1;
        assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        wrong = metadata;
        wrong.byte_offset = 1;
        assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
        let mut transposed = layout;
        transposed.strides_bytes = [48, 2, 16, 4];
        let mut strided = metadata;
        strided.strides = Some(&[24, 1, 8, 2]);
        assert_eq!(
            validate_tensor_metadata(&transposed, &strided, 0).unwrap(),
            (0x1002, 48)
        );
        let mut empty = layout;
        empty.dimensions[2] = 0;
        empty.strides_bytes = [0, 0, 8, 2];
        let mut empty_metadata = metadata;
        empty_metadata.shape = &[1, 2, 0, 4];
        empty_metadata.data = 0;
        empty_metadata.byte_offset = 0;
        assert_eq!(
            validate_tensor_metadata(&empty, &empty_metadata, 0).unwrap(),
            (0, 0)
        );
    }

    #[test]
    fn semantic_tensor_scalar_keeps_one_cell_and_original_rank() {
        for (scalar_type, element_bytes, code) in [
            (1, 1, 1),
            (2, 4, 1),
            (3, 8, 1),
            (4, 2, 2),
            (5, 2, 4),
            (6, 4, 2),
            (7, 8, 0),
            (8, 1, 6),
        ] {
            let layout = SemanticTensorLayout {
                role: 18,
                index: 0,
                scalar_type,
                element_bytes,
                rank: 0,
                logical_axis: u64::MAX,
                dimensions: [0; 4],
                strides_bytes: [0; 4],
            };
            let metadata = TensorMetadata {
                device_type: crate::dlpack::K_DLCUDA,
                device_id: 0,
                dtype: crate::dlpack::DLDataType {
                    code,
                    bits: (element_bytes * 8) as u8,
                    lanes: 1,
                },
                shape: &[],
                strides: None,
                data: 4096,
                byte_offset: element_bytes,
            };
            assert_eq!(
                validate_tensor_metadata(&layout, &metadata, 0).unwrap(),
                (4096 + element_bytes, element_bytes as usize)
            );
            assert_eq!(
                tensor_layout_bytes(&layout).unwrap(),
                element_bytes as usize
            );
            assert_eq!(canonical_tensor_layout(layout).unwrap(), layout);
            assert_eq!(
                tensor_copy_plan(&layout, &layout).unwrap(),
                vec![(0, 0, element_bytes as usize)]
            );
            let mut wrong = metadata;
            wrong.data = 0;
            assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
            wrong = metadata;
            wrong.shape = &[1];
            assert!(validate_tensor_metadata(&layout, &wrong, 0).is_err());
            let mut malformed = layout;
            malformed.logical_axis = 0;
            assert!(tensor_layout_bytes(&malformed).is_err());
            malformed = layout;
            malformed.dimensions[0] = 1;
            assert!(tensor_layout_bytes(&malformed).is_err());
            malformed = layout;
            malformed.strides_bytes[0] = element_bytes;
            assert!(tensor_layout_bytes(&malformed).is_err());
        }
    }

    #[test]
    fn publication_material_scalar_layout_survives_cold_codec() {
        let mut material = publication_material_sample();
        let layout = material.layouts.get_mut(&(18, 0)).unwrap();
        layout.rank = 0;
        layout.dimensions = [0; 4];
        layout.strides_bytes = [0; 4];
        let scalar = *layout;
        let table = material
            .ranges
            .iter_mut()
            .find(|item| item.range.role == 55)
            .unwrap();
        table.bytes = tensor_table_bytes(
            &material.layouts.values().copied().collect::<Vec<_>>(),
            false,
            &[],
        )
        .unwrap();
        table.range.digest = table.original_record_digest();
        publication_material_sample_runtime(&mut material);
        material.bank.header.descriptor_digest = material.original_descriptor_digest();
        let encoded = material.encode().unwrap();
        let restored = PublicationMaterial::decode(&encoded).unwrap();
        assert_eq!(restored.layouts[&(18, 0)], scalar);
        let original = material
            .ranges
            .iter()
            .find(|item| item.range.role == 18)
            .unwrap();
        let actual = restored
            .ranges
            .iter()
            .find(|item| item.range.role == 18)
            .unwrap();
        assert_eq!(actual.bytes, original.bytes);
        assert_eq!(actual.bytes.len(), scalar.element_bytes as usize);
        assert_eq!(restored.encode().unwrap(), encoded);
    }

    #[test]
    fn semantic_tensor_scalar_metadata_allows_null_zero_rank_arrays() {
        // Only the importer's metadata boundary runs here, not a device import.
        let mut value = 3.5f32;
        let mut descriptor = crate::dlpack::DLTensor {
            data: (&mut value as *mut f32).cast(),
            device: crate::dlpack::DLDevice {
                device_type: crate::dlpack::K_DLCUDA,
                device_id: 0,
            },
            ndim: 0,
            dtype: crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let layout = SemanticTensorLayout {
            role: 18,
            index: 0,
            element_bytes: 4,
            scalar_type: 6,
            rank: 0,
            logical_axis: u64::MAX,
            dimensions: [0; 4],
            strides_bytes: [0; 4],
        };
        // SAFETY: zero-rank metadata has no array entries to dereference.
        let actual = unsafe { semantic_tensor_metadata(&descriptor) }.unwrap();
        assert!(actual.shape.is_empty());
        assert!(actual.strides.is_none());
        assert_eq!(
            validate_tensor_metadata(&layout, &actual, 0).unwrap(),
            ((&value as *const f32) as usize as u64, size_of::<f32>())
        );
        for ndim in [-1, 1, 4, 5] {
            descriptor.ndim = ndim;
            // SAFETY: invalid rank/null shape is rejected before any array read.
            assert!(unsafe { semantic_tensor_metadata(&descriptor) }.is_err());
        }
    }

    #[test]
    fn publication_material_empty_model_buffer_keeps_zero_capacity() {
        for role in 18..=20 {
            let mut material = publication_material_sample();
            let layout = material.layouts.get_mut(&(role, 0)).unwrap();
            layout.rank = 3;
            layout.dimensions = [2, 0, 3, 0];
            layout.strides_bytes = [12, 12, 4, 0];
            let empty = *layout;
            let buffer = material
                .ranges
                .iter_mut()
                .find(|item| item.range.role == role)
                .unwrap();
            buffer.bytes.clear();
            buffer.capacity = 0;
            buffer.range.length_bytes = 0;
            let table = material
                .ranges
                .iter_mut()
                .find(|item| item.range.role == 55)
                .unwrap();
            table.bytes = tensor_table_bytes(
                &material.layouts.values().copied().collect::<Vec<_>>(),
                false,
                &[],
            )
            .unwrap();
            table.range.digest = table.original_record_digest();
            publication_material_sample_runtime(&mut material);
            material.bank.header.descriptor_digest = material.original_descriptor_digest();
            let encoded = material.encode().unwrap();
            let restored = PublicationMaterial::decode(&encoded).unwrap();
            assert_eq!(restored.layouts[&(role, 0)], empty);
            let actual = restored
                .ranges
                .iter()
                .find(|item| item.range.role == role)
                .unwrap();
            assert!(actual.bytes.is_empty());
            assert_eq!(actual.capacity, 0);
            assert_eq!(tensor_layout_bytes(&empty).unwrap(), 0);
            assert!(tensor_copy_plan(&empty, &empty).unwrap().is_empty());
            assert_eq!(restored.encode().unwrap(), encoded);
            for (capacity, begin, end) in [(1, 0, 0), (0, 1, 1), (0, 0, 1)] {
                let buffer = material
                    .ranges
                    .iter_mut()
                    .find(|item| item.range.role == role)
                    .unwrap();
                buffer.capacity = capacity;
                buffer.range.logical_begin = begin;
                buffer.range.logical_end = end;
                publication_material_sample_runtime(&mut material);
                material.bank.header.descriptor_digest = material.original_descriptor_digest();
                assert!(
                    material.encode().is_err(),
                    "empty buffer accepted a fictitious capacity or interval"
                );
            }
            let buffer = material
                .ranges
                .iter_mut()
                .find(|item| item.range.role == role)
                .unwrap();
            buffer.capacity = 0;
            buffer.range.logical_begin = 0;
            buffer.range.logical_end = 0;
            publication_material_sample_runtime(&mut material);
            material.bank.header.descriptor_digest = material.original_descriptor_digest();
            material.layouts.get_mut(&(role, 0)).unwrap().dimensions[1] = 1;
            assert!(
                material.encode().is_err(),
                "nonempty model cells acquired zero capacity"
            );
        }
    }

    #[test]
    fn feedback_payload_preserves_qualifiers_without_service_headers() {
        use crate::semantic_hypergraph::SemanticRecordEncoding;
        use crate::SemanticTypedRecord;
        use xlog_core::RelId;

        let atom = |header: u8, value: u32| {
            let mut bytes = vec![header; 55];
            bytes.extend_from_slice(&17u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            let start = bytes.len();
            bytes.push(1); // Canonical U32 argument tag.
            bytes.extend_from_slice(&value.to_le_bytes());
            let end = bytes.len();
            SemanticRecordEncoding {
                bytes,
                arguments: std::iter::once(start..end).collect(),
            }
        };
        let record = |qualifiers| SemanticTypedRecord {
            predicate: RelId(17),
            arguments: vec![crate::SemanticArgument::U32(1)],
            qualifiers,
        };
        let records = SemanticAdmissionRecords {
            predicates: vec![],
            records: vec![record(vec![1, 2]), record(vec![]), record(vec![])],
            supports: vec![],
        };
        let encoded = vec![atom(0xA5, 1), atom(0xA5, 2), atom(0xA5, 3)];
        let payload = feedback_statement_payload(&records, &encoded, 0).unwrap();
        let changed_headers = vec![atom(0x5A, 1), atom(0x5A, 2), atom(0x5A, 3)];
        assert_eq!(
            payload,
            feedback_statement_payload(&records, &changed_headers, 0).unwrap()
        );
        assert_eq!(&payload[..8], &13u64.to_le_bytes());
        assert_eq!(&payload[8..12], &17u32.to_le_bytes());
        assert_eq!(&payload[21..25], &2u32.to_le_bytes());
        let changed_values = vec![atom(0xA5, 1), atom(0xA5, 9), atom(0xA5, 3)];
        assert_ne!(
            payload,
            feedback_statement_payload(&records, &changed_values, 0).unwrap()
        );
        let mut reordered = records;
        reordered.records[0].qualifiers.reverse();
        assert_ne!(
            payload,
            feedback_statement_payload(&reordered, &encoded, 0).unwrap()
        );
    }

    #[test]
    fn publication_tensor_layout_keeps_dtype_distinct_from_byte_width() {
        assert_eq!(size_of::<SemanticTensorLayout>(), 112);
    }

    fn source_rows() -> [SourceSlot; 32] {
        let mut rows = [SourceSlot::default(); 32];
        // Source slots deliberately differ from logical and packed row order.
        rows[5] = SourceSlot {
            token: 40,
            logical_position: 0,
            kind: 1,
            provenance: 1,
            valid: 1,
            committed: 0,
            recomputed: 0,
            provenance_record: 0,
        };
        rows[1] = SourceSlot {
            token: MASK_TOKEN as u64,
            logical_position: 1,
            kind: 2,
            provenance: 0,
            valid: 1,
            committed: 0,
            recomputed: 0,
            provenance_record: 0,
        };
        rows[12] = SourceSlot {
            token: 42,
            logical_position: 2,
            kind: 1,
            provenance: 2,
            valid: 1,
            committed: 0,
            recomputed: 0,
            provenance_record: 2,
        };
        rows
    }

    #[test]
    fn initial_source_ledger_derives_tokens_and_private_provenance_without_changing_geometry() {
        let mut ring = [SourceSlot::default(); 32];
        ring[7] = SourceSlot {
            token: 41,
            logical_position: 3,
            kind: 1,
            provenance: 1,
            valid: 1,
            ..SourceSlot::default()
        };
        ring[2] = SourceSlot {
            token: MASK_TOKEN as u64,
            logical_position: 4,
            kind: 2,
            valid: 1,
            ..SourceSlot::default()
        };
        let prefix = vec![SourceSlot {
            token: 40,
            logical_position: 0,
            kind: 1,
            provenance: 1,
            valid: 1,
            committed: 1,
            ..SourceSlot::default()
        }];
        let sources = [SemanticObservedSource {
            identity: "observation".into(),
            tokens: vec![40, 41],
            origin: b"complete admitted observation and manifest".to_vec(),
        }];
        let mapping = [
            SemanticSourceMapping {
                source: "observation".into(),
                offset: 0,
                logical_position: 0,
            },
            SemanticSourceMapping {
                source: "observation".into(),
                offset: 1,
                logical_position: 3,
            },
        ];
        let (actual, committed, ledger) = initial_source_ledger(
            ring,
            prefix.clone(),
            &sources,
            &mapping,
            b"complete task authority",
            3,
            34,
        )
        .unwrap();
        assert_eq!(
            (
                actual[7].token,
                actual[7].logical_position,
                actual[7].provenance_record
            ),
            (41, 3, 1)
        );
        assert_eq!(committed[0].provenance_record, 0);
        assert_eq!(actual[2].provenance, 0);
        assert_eq!(ledger.bytes.len(), 2 * size_of::<TokenProvenance>());
        assert_eq!(ledger.capacity_bytes, 34 * size_of::<TokenProvenance>());
        // SAFETY: production emitted two complete initialized integral records.
        let record = unsafe {
            ledger
                .bytes
                .as_ptr()
                .add(size_of::<TokenProvenance>())
                .cast::<TokenProvenance>()
                .read_unaligned()
        };
        assert_eq!(
            (
                record.source_slot,
                record.logical_position,
                record.token,
                record.ordinal,
                record.authority_generation
            ),
            (7, 3, 41, 1, 3)
        );
        assert_eq!(record.action_receipt_digest, Identity256::default());
        assert_ne!(record.origin_logical_digest, Identity256::default());
        assert_ne!(record.authority_closure_digest, Identity256::default());
        assert_ne!(record.record_digest, Identity256::default());
        for field in 0..7 {
            let mut changed = ring;
            let mut links = mapping.to_vec();
            match field {
                0 => changed[7].token = 42,
                1 => changed[7].provenance = 2,
                2 => changed[7].provenance_record = 99,
                3 => links[1].offset = 2,
                4 => links[1].logical_position = 4,
                5 => links[1].source = "foreign".into(),
                _ => links[1].logical_position = 0,
            }
            assert!(
                initial_source_ledger(
                    changed,
                    prefix.clone(),
                    &sources,
                    &links,
                    b"complete task authority",
                    3,
                    34
                )
                .is_err(),
                "mutation {field}"
            );
        }
    }

    #[test]
    fn cold_text_parent_derives_only_contiguous_acquired_filled_coverage() {
        let rows = source_rows();
        assert_eq!(
            validate_text_parent(&rows, 0, 64, 2, 262144, 0, 3).unwrap(),
            1
        );
        assert_eq!(rows[1].kind, 2);
        assert_eq!(rows[12].logical_position, 2);
        assert_eq!(size_of::<SourceSlot>(), 64);
    }

    #[test]
    fn cold_mask_source_keeps_the_sentinel_without_fabricated_token_provenance() {
        let mut rows = [SourceSlot::default(); 32];
        rows[7] = SourceSlot {
            token: 248078,
            kind: 2,
            valid: 1,
            ..SourceSlot::default()
        };
        assert_eq!(
            validate_text_parent(&rows, 0, 64, 2, 262144, 0, 0).unwrap(),
            0
        );
        rows[7].token = 0;
        assert!(validate_text_parent(&rows, 0, 64, 2, 262144, 0, 0).is_err());
        rows[7].token = 248078;
        rows[7].provenance_record = 1;
        assert!(validate_text_parent(&rows, 0, 64, 2, 262144, 0, 2).is_err());
    }

    #[test]
    fn cold_text_parent_rejects_inconsistent_kind_and_source_metadata() {
        for field in 0..7 {
            let mut rows = source_rows();
            match field {
                0 => rows[1].provenance = 1,
                1 => rows[1].recomputed = 1,
                2 => rows[5].valid = 0,
                3 => rows[5].committed = 1,
                4 => rows[12].logical_position = 0,
                5 => rows[0].logical_position = 8,
                _ => rows[1].provenance_record = 3,
            }
            assert!(validate_text_parent(&rows, 0, 64, 2, 262144, 0, 3).is_err());
        }
    }

    #[test]
    fn cold_text_parent_checks_prefix_and_reserved_position_capacity() {
        let rows = source_rows();
        assert!(validate_text_parent(&rows, 0, 0, 2, 262144, 0, 3).is_err());
        assert!(validate_text_parent(&rows, 1, 64, 2, 262144, 0, 3).is_err());
        assert!(validate_text_parent(&rows, 0, 64, 2, 97, 0, 3).is_err());
        assert!(validate_text_parent(&rows, 0, 64, 2, 262144, 32, 3).is_err());
        assert!(validate_text_parent(&rows, 0, u64::MAX, 2, 262144, 0, 3).is_err());
    }
}

#[cfg(test)]
pub(crate) mod task_binding_tests {
    use super::*;
    use crate::{
        SemanticArgument, SemanticPolarity, SemanticPredicateRecord, SemanticSupportRecord,
        SemanticTypedRecord,
    };
    use xlog_core::{RelId, ScalarType, Schema};

    #[derive(Debug)]
    struct ArithmeticProgram;

    impl SemanticTaskProgram for ArithmeticProgram {
        fn observe(
            &self,
            _provider: Arc<CudaKernelProvider>,
        ) -> Result<SemanticTaskObservation, SemanticTransitionError> {
            Ok(arithmetic_observation())
        }
    }

    pub(crate) fn arithmetic_observation() -> SemanticTaskObservation {
        let inputs = [[1u32, 1u32], [0u32, 1u32], [1u32, 0u32]];
        let results = inputs.map(|[left, right]| (left + right) / 2);
        SemanticTaskObservation {
            program_source: b"fn observe(left: u32, right: u32) -> u32 { (left + right) / 2 }"
                .to_vec(),
            input_bytes: inputs
                .into_iter()
                .flatten()
                .flat_map(u32::to_le_bytes)
                .collect(),
            result_bytes: results.into_iter().flat_map(u32::to_le_bytes).collect(),
            expected_truth: results.map(|result| {
                if result == 1 {
                    crate::SemanticTruth::True
                } else {
                    crate::SemanticTruth::False
                }
            }),
        }
    }

    pub(crate) fn task_spec(
        statement_records: [u32; 3],
        allowed_support_records: Vec<u32>,
    ) -> SemanticTaskEvaluationSpec {
        SemanticTaskEvaluationSpec {
            statement_records,
            allowed_support_records,
            program: Arc::new(ArithmeticProgram),
            scoring: SemanticTaskScoring {
                correct_weight: 14,
                all_correct_weight: 7,
                work_weight: 1,
                improvement_weight: 45,
                refusal_weight: 15,
                spent_weight: 1,
            },
            admissible_truth_masks: [7, 7, 7],
        }
    }

    fn carry_records() -> SemanticAdmissionRecords {
        let mut records = SemanticAdmissionRecords {
            predicates: vec![SemanticPredicateRecord {
                predicate: RelId(17),
                role: SemanticRecordRole::Statement,
                schema: Schema::new(vec![
                    ("left".into(), ScalarType::U32),
                    ("right".into(), ScalarType::U32),
                ])
                .with_sort_labels(vec!["bit".into(), "bit".into()])
                .unwrap(),
            }],
            records: vec![
                SemanticTypedRecord {
                    predicate: RelId(17),
                    arguments: vec![SemanticArgument::U32(1), SemanticArgument::U32(1)],
                    qualifiers: vec![],
                },
                SemanticTypedRecord {
                    predicate: RelId(17),
                    arguments: vec![SemanticArgument::U32(0), SemanticArgument::U32(1)],
                    qualifiers: vec![],
                },
            ],
            supports: vec![SemanticSupportRecord {
                statement: 0,
                polarity: SemanticPolarity::Pro,
                provenance: 2,
                source: 3,
                context: 4,
                scope: 5,
            }],
        };
        for (index, role) in [
            SemanticRecordRole::Provenance,
            SemanticRecordRole::Source,
            SemanticRecordRole::Context,
            SemanticRecordRole::Scope,
        ]
        .into_iter()
        .enumerate()
        {
            let predicate = RelId(18 + index as u32);
            records.predicates.push(SemanticPredicateRecord {
                predicate,
                role,
                schema: Schema::new(vec![]),
            });
            records.records.push(SemanticTypedRecord {
                predicate,
                arguments: vec![],
                qualifiers: vec![],
            });
        }
        records
    }

    #[test]
    fn task_query_validation_does_not_interpret_argument_values() {
        let mut records = carry_records();
        records.records[0].arguments = vec![SemanticArgument::U32(17), SemanticArgument::U32(23)];
        task_spec([0, 1, 1], vec![0])
            .validate_records(&records)
            .unwrap();
    }

    #[test]
    fn application_observer_retains_input_and_output_bytes() {
        let observation = arithmetic_observation();
        assert_eq!(
            observation.input_bytes,
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]
        );
        assert_eq!(
            observation.result_bytes,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            observation.expected_truth,
            [
                crate::SemanticTruth::True,
                crate::SemanticTruth::False,
                crate::SemanticTruth::False,
            ]
        );
    }

    #[test]
    fn task_binding_identity_seals_observer_execution_and_configuration() {
        let admission = crate::semantic_hypergraph::tests::admit_material_records(carry_records());
        let identity = |spec, observation| {
            TaskEvaluationBinding::bind(&admission, spec, observation)
                .unwrap()
                .identity()
        };
        let original = identity(task_spec([0, 1, 1], vec![0]), arithmetic_observation());
        assert_eq!(
            original,
            identity(task_spec([0, 1, 1], vec![0]), arithmetic_observation()),
            "independent bindings with identical bytes must have the same identity"
        );
        for changed in 0..15 {
            let mut spec = task_spec([0, 1, 1], vec![0]);
            let mut observation = arithmetic_observation();
            let field = match changed {
                0 => {
                    observation.program_source.push(b' ');
                    "program source"
                }
                1 => {
                    observation.input_bytes.push(0);
                    "observer inputs"
                }
                2 => {
                    observation.result_bytes.push(0);
                    "observer results"
                }
                3 => {
                    observation.expected_truth[0] = crate::SemanticTruth::Both;
                    "first expected truth"
                }
                4 => {
                    observation.expected_truth[1] = crate::SemanticTruth::Neither;
                    "second expected truth"
                }
                5 => {
                    observation.expected_truth[2] = crate::SemanticTruth::Neither;
                    "third expected truth"
                }
                6 => {
                    spec.scoring.correct_weight += 1;
                    "correct weight"
                }
                7 => {
                    spec.scoring.all_correct_weight += 1;
                    "all-correct weight"
                }
                8 => {
                    spec.scoring.work_weight += 1;
                    "work weight"
                }
                9 => {
                    spec.scoring.improvement_weight += 1;
                    "improvement weight"
                }
                10 => {
                    spec.scoring.refusal_weight += 1;
                    "refusal weight"
                }
                11 => {
                    spec.scoring.spent_weight += 1;
                    "spent weight"
                }
                12 => {
                    spec.admissible_truth_masks[0] = 15;
                    "first truth mask"
                }
                13 => {
                    spec.admissible_truth_masks[1] = 15;
                    "second truth mask"
                }
                _ => {
                    spec.admissible_truth_masks[2] = 15;
                    "third truth mask"
                }
            };
            assert_ne!(
                original,
                identity(spec, observation),
                "identity omitted {field}"
            );
        }
    }

    #[test]
    fn task_truth_eligibility_requires_nonempty_four_valued_masks() {
        let admission = crate::semantic_hypergraph::tests::admit_material_records(carry_records());
        for slot in 0..3 {
            for mask in [0, 16, 128, 255] {
                let mut spec = task_spec([0, 1, 1], vec![0]);
                spec.admissible_truth_masks[slot] = mask;
                let result =
                    TaskEvaluationBinding::bind(&admission, spec, arithmetic_observation());
                assert!(
                    matches!(result, Err(SemanticTransitionError::InvalidInput { detail })
                    if detail.contains("nonempty four-valued mask")),
                    "slot {slot}, mask {mask}"
                );
            }
            for mask in 1..=15 {
                let mut spec = task_spec([0, 1, 1], vec![0]);
                spec.admissible_truth_masks[slot] = mask;
                TaskEvaluationBinding::bind(&admission, spec, arithmetic_observation()).unwrap();
            }
        }
    }

    #[test]
    fn action_leaf_categories_preserve_equal_original_supports() {
        use crate::semantic_hypergraph::tests::{admit_material_records, root_material_records};
        let mut records = root_material_records();
        records.supports.push(records.supports[0].clone());
        records.supports.push(records.supports[0].clone());
        records.supports.last_mut().unwrap().statement = 1;
        let original_count = records.supports.len();
        let admission = admit_material_records(records);
        let books = ActionCodebooks::derive(&admission, 1).unwrap();
        assert_eq!(books.leaves.len(), original_count);
        assert_eq!(books.words[7], original_count as u64 + 1);
        let mut seen = BTreeSet::new();
        for (category, meaning) in books.leaves.iter().enumerate() {
            let SemanticActionDescriptor::SupportEvent { source_record } = meaning else {
                panic!("wrong leaf meaning")
            };
            assert!(seen.insert(*source_record));
            assert_eq!(
                books.words[books.words[6] as usize + 18 * (category + 1) + 17],
                u64::from(*source_record)
            );
        }
        assert_eq!(seen, (0..original_count as u32).collect());
        let layout = SemanticPolicyLayout::from_components(&books.components).unwrap();
        assert_eq!(layout.fields[17].cardinality, original_count + 1);
        assert_eq!(layout.fields[17].embeddings.len(), original_count * 128);
    }

    #[test]
    #[cfg(feature = "semantic-policy")]
    fn policy_producer_geometry_keeps_full_mask_rows_and_boolean_support() {
        let layouts = policy_producer_layouts(97, 129).unwrap();
        let metadata = TensorMetadata {
            device_type: 2,
            device_id: 0,
            dtype: crate::dlpack::DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: &[32, TEXT_CARDINALITY as i64],
            strides: None,
            data: 4096,
            byte_offset: 4,
        };
        assert_eq!(
            validate_tensor_metadata(&layouts[0], &metadata, 0).unwrap(),
            (4100, 32 * TEXT_CARDINALITY * 4)
        );
        let mut malformed = metadata;
        malformed.shape = &[1, TEXT_CARDINALITY as i64];
        assert!(validate_tensor_metadata(&layouts[0], &malformed, 0).is_err());
        malformed.shape = &[32 * TEXT_CARDINALITY as i64];
        assert!(validate_tensor_metadata(&layouts[0], &malformed, 0).is_err());
        malformed.shape = &[32, TEXT_CARDINALITY as i64];
        malformed.strides = Some(&[TEXT_CARDINALITY as i64 + 1, 1]);
        assert!(validate_tensor_metadata(&layouts[0], &malformed, 0).is_err());
        let support = TensorMetadata {
            device_type: 2,
            device_id: 0,
            dtype: crate::dlpack::DLDataType {
                code: 6,
                bits: 8,
                lanes: 1,
            },
            shape: &[97],
            strides: Some(&[1]),
            data: 4096,
            byte_offset: 0,
        };
        assert_eq!(
            validate_tensor_metadata(&layouts[1], &support, 0).unwrap(),
            (4096, 97)
        );
        let mut wrong_dtype = support;
        wrong_dtype.dtype.code = 1;
        assert!(validate_tensor_metadata(&layouts[1], &wrong_dtype, 0).is_err());
        assert_eq!(layouts[2].dimensions, [129, 0, 0, 0]);
    }

    #[test]
    fn policy_support_reserves_every_original_category_span() {
        use crate::semantic_hypergraph::tests::{admit_material_records, root_material_records};
        let admission = admit_material_records(root_material_records());
        let books = ActionCodebooks::derive(&admission, 1).unwrap();
        let (spans, cells) = books.support_layout();
        assert_eq!(spans.len(), COMPONENT_COUNT);
        let mut cursor = 0;
        for (span, component) in spans.iter().zip(&books.components) {
            assert_eq!(span.start, cursor);
            assert_eq!(span.start, component.offset as usize);
            assert_eq!(span.len(), component.cardinality as usize);
            cursor = span.end;
        }
        assert_eq!(cells, cursor);
        assert_eq!(cells, books.input_cells);
        for lane in 0..2 {
            for slot in 0..32 {
                assert_eq!(spans[lane * 68 + slot].len(), TEXT_CARDINALITY);
            }
        }
    }

    #[test]
    fn action_codebooks_retain_typed_reconstruction_sources() {
        use crate::semantic_hypergraph::tests::{admit_material_records, root_material_records};
        let admission = admit_material_records(root_material_records());
        let books = ActionCodebooks::derive(&admission, 1).unwrap();
        assert_eq!(
            books.words[4] - books.words[2],
            6 * books.words[3],
            "operand bank lost original argument reconstruction references"
        );
        assert_eq!(
            books.words[6] - books.words[4],
            3 * books.words[5],
            "qualifier bank lost original bundle reconstruction references"
        );
        for (category, meaning) in books.operands.iter().enumerate() {
            let SemanticActionDescriptor::Operand { encoded_value, .. } = meaning else {
                panic!("wrong operand meaning")
            };
            let row = books.words[2] as usize + 6 * (category + 1);
            let source = &admission.encoded_records[books.words[row + 4] as usize];
            assert_eq!(
                &source.bytes[source.arguments[books.words[row + 5] as usize].clone()],
                encoded_value
            );
        }
        for (category, meaning) in books.qualifiers.iter().enumerate() {
            let SemanticActionDescriptor::QualifierBundle { records } = meaning else {
                panic!("wrong qualifier meaning")
            };
            let source = books.words[books.words[4] as usize + 3 * (category + 1) + 2];
            if records.is_empty() {
                assert_eq!(source, u64::from(u32::MAX));
            } else {
                assert_eq!(
                    &admission.records().records[source as usize].qualifiers,
                    records
                );
            }
        }
    }

    pub(crate) fn verify_native_decoded_reconstruction(
        executable: &std::path::Path,
        directory: &std::path::Path,
        run: &impl Fn(&mut std::process::Command),
    ) {
        use crate::semantic_hypergraph::tests::{
            admit_material_records, reconstructed_material_statement, root_material_records,
        };
        let mut records = root_material_records();
        records.supports.push(records.supports[0].clone());
        let admission = admit_material_records(records);
        let books = ActionCodebooks::derive(&admission, 1).unwrap();
        let leaf = books
            .leaves
            .iter()
            .position(|value| {
                matches!(
                    value,
                    SemanticActionDescriptor::SupportEvent { source_record: 3 }
                )
            })
            .unwrap() as u32
            + 1;
        for (ordinal, operand) in [1, books.operands.len() as u32].into_iter().enumerate() {
            let mut choices = [0u32; SEMANTIC_TRANSITION_COMPONENT_COUNT];
            choices[2] = 1;
            choices[3] = operand;
            choices[11] = 1;
            choices[17] = leaf;
            let mut input = format!("{}\n", books.words.len());
            for word in &books.words {
                input.push_str(&format!("{word} "));
            }
            for choice in choices {
                input.push_str(&format!("{choice} "));
            }
            let source = directory.join(format!("decoded-input-{ordinal}.txt"));
            let destination = directory.join(format!("decoded-output-{ordinal}.txt"));
            std::fs::write(&source, input).unwrap();
            run(std::process::Command::new(executable)
                .arg("--decode")
                .arg(&source)
                .arg(&destination));
            let words: Vec<u32> = std::fs::read_to_string(&destination)
                .unwrap()
                .split_whitespace()
                .map(|word| word.parse().unwrap())
                .collect();
            assert_eq!(words.len(), 20);
            assert_eq!(
                words[8],
                u32::MAX,
                "computed target invented an original occurrence"
            );
            assert_eq!(
                words[19], 3,
                "actual decoder selected a different equal support occurrence"
            );
            let reconstruction: [u32; 10] = words[9..19].try_into().unwrap();
            let key = reconstructed_material_statement(&admission, None, &reconstruction).unwrap();
            let digest: Vec<u8> = words[..8]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            assert_eq!(
                digest.as_slice(),
                key.identity().as_bytes(),
                "actual decode differs from canonical restore"
            );
            if ordinal == 0 {
                assert_eq!(
                    key.identity(),
                    admission.statement_key(0).unwrap().identity()
                );
            } else {
                assert_ne!(
                    key.identity(),
                    admission.statement_key(0).unwrap().identity()
                );
                assert_ne!(
                    key.identity(),
                    admission.statement_key(1).unwrap().identity()
                );
            }
            assert!(
                reconstructed_material_statement(&admission, Some(0), &reconstruction).is_err(),
                "derived reconstruction was accepted as an original occurrence"
            );
            std::fs::remove_file(source).unwrap();
            std::fs::remove_file(destination).unwrap();
        }
    }

    #[test]
    fn task_scoring_rejects_signed_return_overflow() {
        let mut spec = task_spec([0, 1, 1], vec![0]);
        spec.scoring.correct_weight = u32::MAX;
        spec.scoring.improvement_weight = u32::MAX;
        assert!(spec.validate_records(&carry_records()).is_err());
    }

    #[test]
    fn cold_task_binding_requires_admitted_statement_scope() {
        let spec = task_spec([0, 1, 1], vec![0]);
        spec.validate_records(&carry_records()).unwrap();
        for invalid in 0..4 {
            let mut records = carry_records();
            match invalid {
                0 => records.records[1].predicate = RelId(18),
                1 => records.records[1].predicate = RelId(999),
                2 => records.predicates[0].role = SemanticRecordRole::Qualifier,
                _ => records.supports[0].statement = 2,
            }
            assert!(spec.validate_records(&records).is_err(), "case {invalid}");
        }
        assert!(task_spec([0, 2, 2], vec![])
            .validate_records(&carry_records())
            .is_err());
        assert!(task_spec([0, 1, 1], vec![1])
            .validate_records(&carry_records())
            .is_err());
    }
}
