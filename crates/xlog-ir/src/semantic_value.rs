//! Shared semantic-value references and value-independent feedback coordinates.

use xlog_core::{Result, XlogError};

/// Generational reference qualified by a native owner and handle kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SemanticValueHandle {
    /// Native allocation owner, never part of a content identity.
    pub owner: u64,
    /// Native handle family.
    pub kind: u64,
    /// Slot within that family.
    pub slot: u64,
    /// Slot generation.
    pub generation: u64,
}

/// A statement's two events under one world, context, and measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SemanticValueRef {
    /// Original statement version.
    pub statement_version: SemanticValueHandle,
    /// Qualified world.
    pub world: SemanticValueHandle,
    /// Authored context/restriction.
    pub context: SemanticValueHandle,
    /// Positive support event.
    pub pro_event: SemanticValueHandle,
    /// Negative support event.
    pub contra_event: SemanticValueHandle,
    /// Shared conditioned measure.
    pub measure: SemanticValueHandle,
    /// Derivation dependency root.
    pub derivation: SemanticValueHandle,
}

/// Evaluation completion; none of these states is a Belnap truth value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum SemanticValueCompletion {
    /// Work remains.
    Incomplete = 0,
    /// Original proof/evaluation obligation completed.
    Complete = 1,
    /// Explicit resource or fuel exhaustion.
    Exhausted = 2,
    /// A referenced generation no longer belongs to the acquired version.
    Stale = 3,
    /// The common conditioning event has zero normalizer.
    ZeroNormalizer = 4,
    /// Arithmetic failed its declared numerical law.
    NumericFailure = 5,
    /// A requested numerical certificate could not be established.
    CertificateUnavailable = 6,
}

/// The sole numerical/projection law admitted by this native interface.
pub const SEMANTIC_VALUE_NUMERIC_LAW: [&str; 5] = [
    "log_wmc_f64",
    "nearest_even",
    "joint_00_10_01_11",
    "f32_nearest_even_outward_bounds",
    "explicit_certificates",
];

/// A contiguous learned feature range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticValueFeatureField {
    /// Behavioral field name.
    pub name: &'static str,
    /// First F32 column.
    pub offset: usize,
    /// Number of F32 columns.
    pub length: usize,
    /// Exposed scalar datatype.
    pub dtype: &'static str,
    /// Meaning and units of the values.
    pub units: &'static str,
    /// Native availability condition.
    pub defined_when: &'static str,
}

/// Availability properties of an actual native estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticValueEstimator {
    /// Stable estimator name.
    pub name: &'static str,
    /// Whether an exact sample count is meaningful.
    pub sample_count_defined: bool,
    /// Whether the confidence scalar is meaningful.
    pub confidence_defined: bool,
    /// Numerical error-certificate contract.
    pub error_bound_kind: &'static str,
}

/// Estimators currently implemented by the semantic-value producer.
pub const SEMANTIC_VALUE_ESTIMATORS: [SemanticValueEstimator; 1] = [SemanticValueEstimator {
    name: "exact_structure_f64",
    sample_count_defined: false,
    confidence_defined: false,
    error_bound_kind: "explicit_available_or_undefined",
}];

/// Scalar U64 fields in the unlearned query receipt. F64 payloads retain their
/// exact IEEE-754 bits. These references do not enter the learned byte lanes.
pub const SEMANTIC_VALUE_RECEIPT_FIELDS: &[&str] = &[
    "status",
    "original_statement",
    "statement_slot",
    "statement_generation",
    "value_slot",
    "value_generation",
    "world_ordinal",
    "context_ordinal",
    "condition_slot",
    "condition_generation",
    "graph_generation",
    "image_generation",
    "coefficient_generation",
    "measure_generation",
    "representation_generation",
    "pro_event_slot",
    "pro_event_generation",
    "contra_event_slot",
    "contra_event_generation",
    "derivation_slot",
    "derivation_generation",
    "query_kind",
    "proof_kind",
    "completion",
    "estimator_kind",
    "assignment_defined",
    "mass_defined",
    "error_bound_defined",
    "sample_count",
    "confidence_f64_bits",
    "log_normalizer_f64_bits",
    "joint_mass_00_f64_bits",
    "joint_mass_10_f64_bits",
    "joint_mass_01_f64_bits",
    "joint_mass_11_f64_bits",
    "error_bound_00_f64_bits",
    "error_bound_10_f64_bits",
    "error_bound_01_f64_bits",
    "error_bound_11_f64_bits",
    "parent_instance_0",
    "parent_instance_1",
    "parent_instance_2",
    "parent_instance_3",
    "publication_word",
    "parent_logical_digest_0",
    "parent_logical_digest_1",
    "parent_logical_digest_2",
    "parent_logical_digest_3",
    "world_identity_0",
    "world_identity_1",
    "world_identity_2",
    "world_identity_3",
    "context_identity_0",
    "context_identity_1",
    "context_identity_2",
    "context_identity_3",
    "measure_identity_0",
    "measure_identity_1",
    "measure_identity_2",
    "measure_identity_3",
    "numeric_law_identity_0",
    "numeric_law_identity_1",
    "numeric_law_identity_2",
    "numeric_law_identity_3",
    "semantic_rng_namespace_0",
    "semantic_rng_namespace_1",
    "semantic_rng_namespace_2",
    "semantic_rng_namespace_3",
    "semantic_rng_counter_0",
    "semantic_rng_counter_1",
    "semantic_rng_counter_2",
    "semantic_rng_counter_3",
    "semantic_rng_entry_slot",
    "semantic_rng_entry_generation",
    "native_query_identity_0",
    "native_query_identity_1",
    "native_query_identity_2",
    "native_query_identity_3",
];

/// Scalar U64 fields in every unlearned support/dependency row.
pub const SEMANTIC_VALUE_SUPPORT_FIELDS: &[&str] = &[
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
    "dependency_kind",
    "dependency_slot",
    "dependency_generation",
    "parent_dependency_slot",
    "parent_dependency_generation",
    "world_ordinal",
    "context_ordinal",
    "event_slot",
    "event_generation",
    "rule_slot",
    "rule_generation",
    "coefficient_generation",
    "measure_generation",
    "representation_generation",
    "semantic_rng_entry_slot",
    "semantic_rng_entry_generation",
];

/// Full structural schema shared by every scientific arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticValueFeedbackSchema {
    /// Public original-statement byte width.
    pub statement_bytes: usize,
    /// Contiguous ranges for the F32 learned projection.
    pub fields: Vec<SemanticValueFeatureField>,
    /// Full F32 row width.
    pub feature_width: usize,
}

impl SemanticValueFeedbackSchema {
    /// Declares actual feature coordinates without binding a task answer,
    /// intervention, mutable semantic value, or full-world content hash.
    pub fn new(statement_bytes: usize) -> Result<Self> {
        let statement_width = statement_bytes
            .checked_mul(9)
            .filter(|_| statement_bytes != 0)
            .ok_or_else(|| {
                XlogError::Compilation("semantic statement feature width overflows".into())
            })?;
        let layout = [
            ("statement_bit_presence", statement_width, "bits", "valid"),
            (
                "assignment_truth",
                4,
                "one_hot_true_false_both_neither",
                "assignment_defined",
            ),
            ("joint_masses", 4, "probability_00_10_01_11", "mass_defined"),
            (
                "absolute_error_bounds",
                4,
                "absolute_probability_error_00_10_01_11",
                "error_bound_defined",
            ),
            ("query_kind", 2, "one_hot_assignment_measure", "valid"),
            (
                "estimator_kind",
                SEMANTIC_VALUE_ESTIMATORS.len(),
                "one_hot_estimator",
                "valid",
            ),
            (
                "value_availability",
                3,
                "assignment_mass_error_defined",
                "valid",
            ),
            (
                "sample_count",
                1,
                "samples",
                "estimator.sample_count_defined",
            ),
            (
                "confidence",
                1,
                "probability",
                "estimator.confidence_defined",
            ),
        ];
        let mut offset = 0usize;
        let mut fields = Vec::with_capacity(layout.len());
        for (name, length, units, defined_when) in layout {
            fields.push(SemanticValueFeatureField {
                name,
                offset,
                length,
                dtype: "float32",
                units,
                defined_when,
            });
            offset = offset
                .checked_add(length)
                .filter(|&width| i64::try_from(width).is_ok())
                .ok_or_else(|| {
                    XlogError::Compilation("semantic value feature width overflows".into())
                })?;
        }
        Ok(Self {
            statement_bytes,
            fields,
            feature_width: offset,
        })
    }
}
