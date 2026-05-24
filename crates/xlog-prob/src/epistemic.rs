//! Bounded epistemic/probabilistic integration helpers for v0.9 fixtures.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{symbol, Result, ScalarType, XlogError};
use xlog_cuda::{CompareOp, CudaBuffer, CudaKernelProvider};
use xlog_ir::{EirEpistemicMode, EirEpistemicOp, EirTerm, EpistemicTupleMembershipBinding};
use xlog_logic::{
    ast::{Atom, EpistemicLiteral, EpistemicOp, Term},
    epistemic::{EpistemicWorldView, TruthValue},
};
use xlog_runtime::{
    EpistemicGpuExecutionResult, EpistemicGpuKernelTimingTrace, EpistemicGpuProviderIdentity,
};

/// Default tolerance for deterministic probability fixtures.
pub const EPISTEMIC_PROBABILITY_TOLERANCE: f64 = 1.0e-12;

/// Role epistemic choices play in probabilistic compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicProbabilisticRole {
    /// Epistemic choices are compiled as evidence conditions over the probabilistic query.
    EvidenceConditioning,
}

/// Semantic contract between epistemic and probabilistic layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicProbabilisticContract {
    /// How epistemic choices affect probabilistic inference.
    pub epistemic_role: EpistemicProbabilisticRole,
}

impl Default for EpistemicProbabilisticContract {
    fn default() -> Self {
        Self {
            epistemic_role: EpistemicProbabilisticRole::EvidenceConditioning,
        }
    }
}

/// Epistemic assumption operator used as probabilistic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicAssumptionKind {
    /// `know atom` assumption.
    Know,
    /// `possible atom` assumption.
    Possible,
}

impl EpistemicAssumptionKind {
    fn evidence_prefix(self) -> &'static str {
        match self {
            Self::Know => "know",
            Self::Possible => "possible",
        }
    }
}

/// Concrete tuple key term for nonzero-arity epistemic evidence conditioning.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicEvidenceTerm {
    /// Integer tuple key term.
    Integer(i64),
    /// String tuple key term.
    String(String),
    /// Interned symbol tuple key term.
    Symbol(u32),
}

impl EpistemicEvidenceTerm {
    /// Construct an integer tuple key term.
    pub fn integer(value: i64) -> Self {
        Self::Integer(value)
    }

    /// Construct a string tuple key term.
    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Construct an interned symbol tuple key term.
    pub fn symbol(value: u32) -> Self {
        Self::Symbol(value)
    }

    fn evidence_literal(&self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::String(value) => format!("{value:?}"),
            Self::Symbol(value) => format!("#{value}"),
        }
    }
}

/// One bounded epistemic assumption compiled as evidence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EpistemicAssumption {
    /// Assumption kind.
    pub kind: EpistemicAssumptionKind,
    /// Predicate name.
    pub predicate: String,
    /// Predicate arity.
    pub arity: usize,
    /// Concrete tuple terms for nonzero-arity evidence conditioning.
    pub terms: Vec<EpistemicEvidenceTerm>,
    /// Assumed evidence truth value.
    pub value: bool,
}

impl EpistemicAssumption {
    /// Construct a `know predicate/arity = value` assumption.
    pub fn known(predicate: impl Into<String>, arity: usize, value: bool) -> Self {
        Self {
            kind: EpistemicAssumptionKind::Know,
            predicate: predicate.into(),
            arity,
            terms: Vec::new(),
            value,
        }
    }

    /// Construct a `know predicate(terms...) = value` assumption.
    pub fn known_tuple(
        predicate: impl Into<String>,
        terms: Vec<EpistemicEvidenceTerm>,
        value: bool,
    ) -> Self {
        Self {
            kind: EpistemicAssumptionKind::Know,
            predicate: predicate.into(),
            arity: terms.len(),
            terms,
            value,
        }
    }

    /// Construct a `possible predicate/arity = value` assumption.
    pub fn possible(predicate: impl Into<String>, arity: usize, value: bool) -> Self {
        Self {
            kind: EpistemicAssumptionKind::Possible,
            predicate: predicate.into(),
            arity,
            terms: Vec::new(),
            value,
        }
    }

    /// Construct a `possible predicate(terms...) = value` assumption.
    pub fn possible_tuple(
        predicate: impl Into<String>,
        terms: Vec<EpistemicEvidenceTerm>,
        value: bool,
    ) -> Self {
        Self {
            kind: EpistemicAssumptionKind::Possible,
            predicate: predicate.into(),
            arity: terms.len(),
            terms,
            value,
        }
    }

    /// Return the compiler-facing evidence literal for this assumption.
    pub fn evidence_literal(&self) -> String {
        if self.terms.is_empty() {
            format!(
                "{}:{}/{}={}",
                self.kind.evidence_prefix(),
                self.predicate,
                self.arity,
                self.value
            )
        } else {
            let terms = self
                .terms
                .iter()
                .map(EpistemicEvidenceTerm::evidence_literal)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{}:{}/{}({})={}",
                self.kind.evidence_prefix(),
                self.predicate,
                self.arity,
                terms,
                self.value
            )
        }
    }

    fn same_evidence_key(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.predicate == other.predicate
            && self.arity == other.arity
            && self.terms == other.terms
    }
}

/// Knowledge compiler adapter kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerAdapterKind {
    /// Existing GPU D4/XGCF compiler.
    GpuD4,
    /// Alternative external Decision-DNNF text adapter.
    ExternalDdnnfText,
    /// Alternative external c2d Decision-DNNF compiler adapter.
    ExternalC2d,
    /// Alternative external miniC2D Decision-DNNF compiler adapter.
    ExternalMiniC2d,
}

/// Implementation status for an adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerAdapterSupport {
    /// Adapter is implemented in this crate.
    Implemented,
    /// Adapter is recorded as a design contract for a future implementation.
    DesignOnly,
}

/// Compiler input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerInputFormat {
    /// Device-resident GPU CNF.
    GpuCnf,
    /// DIMACS CNF text.
    DimacsCnf,
}

/// Compiler output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerOutputFormat {
    /// Device-resident XGCF circuit.
    Xgcf,
    /// Decision-DNNF text.
    DecisionDnnfText,
}

/// Knowledge compiler adapter metadata used by bounded fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeCompilerAdapter {
    /// Human-readable adapter name.
    pub name: String,
    /// Adapter kind.
    pub kind: CompilerAdapterKind,
    /// Implementation support status.
    pub support: CompilerAdapterSupport,
    /// Input format consumed by the adapter.
    pub input_format: CompilerInputFormat,
    /// Output format emitted by the adapter.
    pub output_format: CompilerOutputFormat,
    incremental_evidence: bool,
}

impl KnowledgeCompilerAdapter {
    /// Return the existing GPU-D4 adapter.
    pub fn gpu_d4() -> Self {
        Self {
            name: "gpu-d4".to_string(),
            kind: CompilerAdapterKind::GpuD4,
            support: CompilerAdapterSupport::Implemented,
            input_format: CompilerInputFormat::GpuCnf,
            output_format: CompilerOutputFormat::Xgcf,
            incremental_evidence: true,
        }
    }

    /// Return an alternative external Decision-DNNF text adapter design.
    pub fn external_ddnnf_text(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: CompilerAdapterKind::ExternalDdnnfText,
            support: CompilerAdapterSupport::DesignOnly,
            input_format: CompilerInputFormat::DimacsCnf,
            output_format: CompilerOutputFormat::DecisionDnnfText,
            incremental_evidence: false,
        }
    }

    /// Return the explicit c2d Decision-DNNF text adapter design.
    pub fn external_c2d() -> Self {
        Self {
            name: "c2d".to_string(),
            kind: CompilerAdapterKind::ExternalC2d,
            support: CompilerAdapterSupport::DesignOnly,
            input_format: CompilerInputFormat::DimacsCnf,
            output_format: CompilerOutputFormat::DecisionDnnfText,
            incremental_evidence: false,
        }
    }

    /// Return the explicit miniC2D Decision-DNNF text adapter design.
    pub fn external_mini_c2d() -> Self {
        Self {
            name: "miniC2D".to_string(),
            kind: CompilerAdapterKind::ExternalMiniC2d,
            support: CompilerAdapterSupport::DesignOnly,
            input_format: CompilerInputFormat::DimacsCnf,
            output_format: CompilerOutputFormat::DecisionDnnfText,
            incremental_evidence: false,
        }
    }

    /// Whether the adapter can update evidence without rebuilding the circuit.
    pub fn supports_incremental_evidence(&self) -> bool {
        self.incremental_evidence
    }
}

/// Circuit update mode for assumption changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitUpdateMode {
    /// Assumption was already active, so no circuit state changed.
    Unchanged,
    /// Evidence was updated without rebuilding the compiled circuit.
    IncrementalEvidence,
    /// The adapter does not support incremental evidence and rebuilt the circuit.
    FullRebuild,
}

/// Result of applying an epistemic assumption to a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitUpdate {
    /// Update mode used by the adapter.
    pub mode: CircuitUpdateMode,
    /// Number of compile operations performed by this circuit state.
    pub compile_count: usize,
    /// Stable circuit fingerprint after the update.
    pub circuit_fingerprint: u64,
}

/// Evidence derived from an accepted epistemic world view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedWorldViewEvidence {
    assumptions: Vec<EpistemicAssumption>,
    world_count: usize,
    gpu_epistemic_mode: Option<EirEpistemicMode>,
    gpu_tuple_key_column_reads: usize,
    gpu_final_tuple_row_filters: usize,
    gpu_final_tuple_negated_row_filters: usize,
    gpu_row_specific_membership_row_capacity: usize,
    gpu_row_filter_fallback_row_capacity: usize,
    gpu_checked_constraint_relations: usize,
    gpu_constraint_row_count_device_reads: usize,
}

impl AcceptedWorldViewEvidence {
    /// Construct evidence from a non-empty accepted world view.
    pub fn new(
        world_view: &EpistemicWorldView,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<Self> {
        if assumptions.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted world-view evidence".to_string(),
                context:
                    "probabilistic evidence requires at least one accepted epistemic assumption"
                        .to_string(),
            });
        }
        validate_world_view_assumptions(world_view, &assumptions)?;
        Ok(Self {
            assumptions,
            world_count: world_view.world_count(),
            gpu_epistemic_mode: None,
            gpu_tuple_key_column_reads: 0,
            gpu_final_tuple_row_filters: 0,
            gpu_final_tuple_negated_row_filters: 0,
            gpu_row_specific_membership_row_capacity: 0,
            gpu_row_filter_fallback_row_capacity: 0,
            gpu_checked_constraint_relations: 0,
            gpu_constraint_row_count_device_reads: 0,
        })
    }

    /// Construct evidence from an accepted GPU epistemic execution result.
    ///
    /// This is the production boundary used by probabilistic adapters: it
    /// accepts only results that used timed GPU candidate-generation,
    /// propagation, validation, stable-model tuple membership, world-view,
    /// accepted-candidate, final-result, and final-tuple kernels, zero
    /// hot-path host transfers, and a non-empty device final output.
    pub fn from_gpu_execution_result(
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<Self> {
        let provider_identity = EpistemicGpuProviderIdentity::from_provider(provider);
        if result.provider_identity != provider_identity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence provider mismatch: result device={} provider device={} \
                     result_device_ptr={} provider_device_ptr={} result_memory_ptr={} \
                     provider_memory_ptr={}",
                    result.provider_identity.device_ordinal,
                    provider_identity.device_ordinal,
                    result.provider_identity.device_ptr,
                    provider_identity.device_ptr,
                    result.provider_identity.memory_ptr,
                    provider_identity.memory_ptr
                ),
            });
        }
        if !result.prepared.preflight.cpu_fallbacks.is_zero() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: "probabilistic evidence requires zero epistemic CPU fallback counters"
                    .to_string(),
            });
        }
        result.require_runtime_dispatch_certification()?;
        result
            .model_membership
            .require_stable_model_tuple_source()?;
        if result.constraint_validation.violated_constraint_relations != 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence requires zero reduced constraint violations, got {} \
                     across {} checked constraint relations",
                    result.constraint_validation.violated_constraint_relations,
                    result.constraint_validation.checked_constraint_relations
                ),
            });
        }
        if result.constraint_validation.row_count_device_reads as usize
            > result.constraint_validation.checked_constraint_relations
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence constraint metadata reads cannot exceed checked \
                     reduced constraint relations, got reads={} checked={}",
                    result.constraint_validation.row_count_device_reads,
                    result.constraint_validation.checked_constraint_relations
                ),
            });
        }
        require_gpu_kernel_trace(
            "candidate generation",
            result.candidate_generation.kernel_launches,
            result.candidate_generation.host_write_ops,
            result.candidate_generation.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "candidate propagation",
            result.propagation.kernel_launches,
            result.propagation.host_write_ops,
            result.propagation.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "candidate validation",
            result.candidate_validation.kernel_launches,
            result.candidate_validation.host_write_ops,
            result.candidate_validation.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "model membership",
            result.model_membership.kernel_launches,
            result.model_membership.host_write_ops,
            result.model_membership.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "world-view validation",
            result.world_view_validation.kernel_launches,
            result.world_view_validation.host_write_ops,
            result.world_view_validation.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "accepted-candidate materialization",
            result.materialization.kernel_launches,
            result.materialization.host_write_ops,
            result.materialization.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "final-result materialization",
            result.final_result_materialization.kernel_launches,
            result.final_result_materialization.host_write_ops,
            result.final_result_materialization.kernel_timing,
        )?;
        require_gpu_kernel_trace(
            "final tuple materialization",
            result.final_tuple_materialization.kernel_launches,
            result.final_tuple_materialization.host_write_ops,
            result.final_tuple_materialization.kernel_timing,
        )?;
        // The runtime has already captured this via read_device_row_count during
        // the bounded final-result transfer; do not re-read it in the prob gate.
        let accepted_rows = result.final_result_transfer.final_output_rows;
        result
            .final_tuple_materialization
            .require_row_filter_materialization_evidence(
                "accepted GPU world-view evidence",
                accepted_rows,
            )?;
        if result.transfer_budget.tracked_dtoh_calls != 0
            || result.transfer_budget.tracked_htod_calls != 0
            || result.transfer_budget.tracked_data_plane_htod_calls != 0
            || result.transfer_budget.per_candidate_host_round_trips != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence requires zero hot-path transfers outside bounded \
                     launch metadata, got dtoh_calls={}, htod_calls={}, \
                     data_plane_htod_calls={}, launch_metadata_htod_calls={}, \
                     per_candidate_round_trips={}",
                    result.transfer_budget.tracked_dtoh_calls,
                    result.transfer_budget.tracked_htod_calls,
                    result.transfer_budget.tracked_data_plane_htod_calls,
                    result.transfer_budget.tracked_launch_metadata_htod_calls,
                    result.transfer_budget.per_candidate_host_round_trips
                ),
            });
        }
        require_accepted_gpu_semantic_trace(result)?;

        if accepted_rows == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: "probabilistic evidence requires non-empty accepted GPU final output"
                    .to_string(),
            });
        }
        if result.semantic_trace.accepted_candidates == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: "probabilistic evidence requires at least one GPU-accepted candidate"
                    .to_string(),
            });
        }
        let accepted_assumptions =
            accepted_gpu_evidence_assumptions(provider, result, &assumptions)?;

        Ok(Self {
            assumptions: accepted_assumptions,
            world_count: result.semantic_trace.accepted_world_views,
            gpu_epistemic_mode: Some(result.prepared.preflight.epistemic_mode),
            gpu_tuple_key_column_reads: result.model_membership.tuple_source_key_column_device_reads
                as usize,
            gpu_final_tuple_row_filters: result.final_tuple_materialization.row_filter_count,
            gpu_final_tuple_negated_row_filters: result
                .final_tuple_materialization
                .negated_row_filter_count,
            gpu_row_specific_membership_row_capacity: result
                .final_tuple_materialization
                .row_specific_membership_row_capacity,
            gpu_row_filter_fallback_row_capacity: result
                .final_tuple_materialization
                .row_filter_row_capacity_outside_model_slot_window,
            gpu_checked_constraint_relations: result
                .constraint_validation
                .checked_constraint_relations,
            gpu_constraint_row_count_device_reads: result
                .constraint_validation
                .row_count_device_reads as usize,
        })
    }

    /// Number of worlds used to validate this evidence.
    pub fn world_count(&self) -> usize {
        self.world_count
    }

    /// Accepted epistemic assumptions represented by this evidence.
    pub fn assumptions(&self) -> &[EpistemicAssumption] {
        &self.assumptions
    }

    pub(crate) fn with_assumptions(&self, assumptions: Vec<EpistemicAssumption>) -> Self {
        let mut evidence = self.clone();
        evidence.assumptions = assumptions;
        evidence
    }

    /// Epistemic mode reported by the accepted GPU runtime evidence, when present.
    pub fn gpu_epistemic_mode(&self) -> Option<EirEpistemicMode> {
        self.gpu_epistemic_mode
    }

    /// Number of accepted epistemic assumptions represented by this evidence.
    pub fn assumption_count(&self) -> usize {
        self.assumptions.len()
    }

    /// Accepted nonzero-arity epistemic assumptions represented by this evidence.
    pub fn nonzero_arity_assumption_count(&self) -> usize {
        self.assumptions
            .iter()
            .filter(|assumption| assumption.arity > 0)
            .count()
    }

    /// Maximum accepted epistemic assumption arity represented by this evidence.
    pub fn max_assumption_arity(&self) -> usize {
        self.assumptions
            .iter()
            .map(|assumption| assumption.arity)
            .max()
            .unwrap_or(0)
    }

    /// Tuple-key device column reads used while staging accepted GPU tuple evidence.
    pub fn gpu_tuple_key_column_reads(&self) -> usize {
        self.gpu_tuple_key_column_reads
    }

    /// GPU final-tuple row filters used to materialize variable-bound evidence.
    pub fn gpu_final_tuple_row_filters(&self) -> usize {
        self.gpu_final_tuple_row_filters
    }

    /// Negated GPU final-tuple row filters used to materialize variable-bound evidence.
    pub fn gpu_final_tuple_negated_row_filters(&self) -> usize {
        self.gpu_final_tuple_negated_row_filters
    }

    /// Final-output row capacity checked against row-specific GPU model slots.
    pub fn gpu_row_specific_membership_row_capacity(&self) -> usize {
        self.gpu_row_specific_membership_row_capacity
    }

    /// Final-output row capacity checked by fallback GPU row filters outside model slots.
    pub fn gpu_row_filter_fallback_row_capacity(&self) -> usize {
        self.gpu_row_filter_fallback_row_capacity
    }

    /// Reduced integrity-constraint relations checked by accepted GPU execution.
    pub fn gpu_checked_constraint_relations(&self) -> usize {
        self.gpu_checked_constraint_relations
    }

    /// Constraint row-count metadata reads used by accepted GPU execution.
    pub fn gpu_constraint_row_count_device_reads(&self) -> usize {
        self.gpu_constraint_row_count_device_reads
    }
}

fn validate_world_view_assumptions(
    world_view: &EpistemicWorldView,
    assumptions: &[EpistemicAssumption],
) -> Result<()> {
    for assumption in assumptions {
        let actual_value =
            world_view.evaluate(&epistemic_literal_for_assumption(assumption)?) == TruthValue::True;
        if actual_value != assumption.value {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence assumption {} was not accepted by world view",
                    assumption.evidence_literal()
                ),
            });
        }
    }
    Ok(())
}

fn epistemic_literal_for_assumption(assumption: &EpistemicAssumption) -> Result<EpistemicLiteral> {
    if assumption.arity > 0 && assumption.terms.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted world-view evidence".to_string(),
            context: format!(
                "nonzero probabilistic evidence assumption {} requires concrete tuple terms",
                assumption.evidence_literal()
            ),
        });
    }
    if !assumption.terms.is_empty() && assumption.terms.len() != assumption.arity {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumption {} has arity {}, but {} concrete terms",
                assumption.evidence_literal(),
                assumption.arity,
                assumption.terms.len()
            ),
        });
    }

    let terms = assumption
        .terms
        .iter()
        .map(epistemic_evidence_term_to_logic_term)
        .collect();
    let op = match assumption.kind {
        EpistemicAssumptionKind::Know => EpistemicOp::Know,
        EpistemicAssumptionKind::Possible => EpistemicOp::Possible,
    };

    Ok(EpistemicLiteral {
        op,
        negated: false,
        atom: Atom {
            predicate: assumption.predicate.clone(),
            terms,
        },
    })
}

fn epistemic_evidence_term_to_logic_term(term: &EpistemicEvidenceTerm) -> Term {
    match term {
        EpistemicEvidenceTerm::Integer(value) => Term::Integer(*value),
        EpistemicEvidenceTerm::String(value) => Term::String(value.clone()),
        EpistemicEvidenceTerm::Symbol(value) => Term::Symbol(*value),
    }
}

fn require_accepted_gpu_semantic_trace(result: &EpistemicGpuExecutionResult) -> Result<()> {
    let trace = &result.semantic_trace;
    let accounted_candidates = trace
        .accepted_candidates
        .checked_add(trace.rejected_candidates)
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence semantic trace candidate accounting overflowed: \
                 accepted={} rejected={}",
                trace.accepted_candidates, trace.rejected_candidates
            ),
        })?;
    if trace.generated_candidates != result.candidate_generation.generated_candidates
        || trace.tested_candidates != result.world_view_validation.candidates_checked
        || trace.accepted_candidates != trace.accepted_candidate_indices.len()
        || trace.rejected_candidates != trace.rejected_candidate_indices.len()
        || trace.accepted_world_views != trace.accepted_candidates
        || accounted_candidates != trace.generated_candidates
        || trace.cpu_candidate_enumerations != 0
        || trace.cpu_world_view_validations != 0
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence requires a consistent GPU semantic trace with zero CPU \
                 fallbacks, got generated={}, tested={}, expected_generated={}, \
                 expected_tested={}, accepted={} accepted_indices={}, accepted_world_views={}, \
                 rejected={} rejected_indices={}, cpu_candidates={}, cpu_world_views={}",
                trace.generated_candidates,
                trace.tested_candidates,
                result.candidate_generation.generated_candidates,
                result.world_view_validation.candidates_checked,
                trace.accepted_candidates,
                trace.accepted_candidate_indices.len(),
                trace.accepted_world_views,
                trace.rejected_candidates,
                trace.rejected_candidate_indices.len(),
                trace.cpu_candidate_enumerations,
                trace.cpu_world_view_validations
            ),
        });
    }
    Ok(())
}

fn accepted_gpu_evidence_assumptions(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
    assumptions: &[EpistemicAssumption],
) -> Result<Vec<EpistemicAssumption>> {
    let preflight = &result.prepared.preflight;
    if result.tuple_membership_bindings.len() != preflight.tuple_membership_binding_count {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence requires executed tuple-membership bindings, got {} \
                 bindings for preflight count {}",
                result.tuple_membership_bindings.len(),
                preflight.tuple_membership_binding_count
            ),
        });
    }
    if !assumptions.is_empty() && assumptions.len() != preflight.tuple_membership_binding_count {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence must cover every GPU-validated tuple-membership binding, \
                 or supply no assumption facts for gate-only production reuse; got {} assumptions \
                 for {} bindings",
                assumptions.len(),
                preflight.tuple_membership_binding_count
            ),
        });
    }
    let assumptions =
        resolve_gpu_evidence_assumptions(provider, result, assumptions, preflight.epistemic_mode)?;
    let mut know_bindings = BTreeSet::new();
    let mut possible_bindings = BTreeSet::new();
    let mut not_know_bindings = BTreeSet::new();
    let mut not_possible_bindings = BTreeSet::new();
    let mut bound_tuple_bindings = BTreeSet::new();
    let mut negated_bound_tuple_bindings = BTreeSet::new();
    let mut accepted_assumptions = Vec::with_capacity(assumptions.len());
    for assumption in &assumptions {
        if assumption.arity > 0 && assumption.terms.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "nonzero probabilistic evidence assumption {} requires concrete tuple terms",
                    assumption.evidence_literal()
                ),
            });
        }
        if !assumption.terms.is_empty() && assumption.terms.len() != assumption.arity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence assumption {} has arity {}, but {} concrete terms",
                    assumption.evidence_literal(),
                    assumption.arity,
                    assumption.terms.len()
                ),
            });
        }
        let Some(binding_match) =
            find_gpu_evidence_binding(provider, result, assumption, preflight.epistemic_mode)?
        else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence assumption {} was not validated by the accepted GPU \
                     tuple-membership bindings",
                    assumption.evidence_literal()
                ),
            });
        };
        if accepted_assumptions
            .iter()
            .any(|previous: &EpistemicAssumption| {
                binding_match
                    .accepted_assumption
                    .same_evidence_key(previous)
            })
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence duplicates epistemic assumption key {}",
                    binding_match.accepted_assumption.evidence_literal()
                ),
            });
        }
        let binding = binding_match.binding;
        let binding_key = (binding.literal_index, binding.reduction_index);
        match (binding.op, binding.negated) {
            (EirEpistemicOp::Know, false) => {
                know_bindings.insert(binding_key);
            }
            (EirEpistemicOp::Possible, false) => {
                possible_bindings.insert(binding_key);
            }
            (EirEpistemicOp::Know, true) => {
                not_know_bindings.insert(binding_key);
            }
            (EirEpistemicOp::Possible, true) => {
                not_possible_bindings.insert(binding_key);
            }
        }
        if binding_match.matched_concrete_tuple_key
            && binding.bound_output_columns.iter().any(Option::is_some)
        {
            bound_tuple_bindings.insert(binding_key);
            if binding.negated {
                negated_bound_tuple_bindings.insert(binding_key);
            }
        }
        accepted_assumptions.push(binding_match.accepted_assumption);
    }
    if know_bindings.len() > preflight.know_operator_count
        || possible_bindings.len() > preflight.possible_operator_count
        || not_know_bindings.len() > preflight.not_know_operator_count
        || not_possible_bindings.len() > preflight.not_possible_operator_count
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumptions exceed GPU-validated operator counts: \
                 know={}/{} possible={}/{} not_know={}/{} not_possible={}/{}",
                know_bindings.len(),
                preflight.know_operator_count,
                possible_bindings.len(),
                preflight.possible_operator_count,
                not_know_bindings.len(),
                preflight.not_know_operator_count,
                not_possible_bindings.len(),
                preflight.not_possible_operator_count
            ),
        });
    }
    let final_tuple_trace = result.final_tuple_materialization;
    if bound_tuple_bindings.len() > final_tuple_trace.row_filter_count
        || negated_bound_tuple_bindings.len() > final_tuple_trace.negated_row_filter_count
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence supplied variable-bound tuple assumptions without \
                 matching GPU final-tuple row-filter materialization: bound={}/{} \
                 negated_bound={}/{}",
                bound_tuple_bindings.len(),
                final_tuple_trace.row_filter_count,
                negated_bound_tuple_bindings.len(),
                final_tuple_trace.negated_row_filter_count
            ),
        });
    }
    Ok(accepted_assumptions)
}

fn resolve_gpu_evidence_assumptions(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
    assumptions: &[EpistemicAssumption],
    mode: EirEpistemicMode,
) -> Result<Vec<EpistemicAssumption>> {
    if assumptions.is_empty() {
        return concrete_gpu_evidence_assumptions_for_all_bindings(provider, result);
    }

    let mut resolved = BTreeSet::new();
    for assumption in assumptions {
        if assumption.arity > 0 && assumption.terms.is_empty() {
            for concrete in concrete_gpu_evidence_assumptions(provider, result, assumption, mode)? {
                resolved.insert(concrete);
            }
        } else {
            resolved.insert(assumption.clone());
        }
    }
    Ok(resolved.into_iter().collect())
}

fn concrete_gpu_evidence_assumptions_for_all_bindings(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
) -> Result<Vec<EpistemicAssumption>> {
    if result.tuple_membership_bindings.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: "probabilistic evidence requires at least one accepted GPU tuple-membership \
                      binding"
                .to_string(),
        });
    }

    let mut resolved = BTreeSet::new();
    for binding in &result.tuple_membership_bindings {
        let assumption = EpistemicAssumption {
            kind: match binding.op {
                EirEpistemicOp::Know => EpistemicAssumptionKind::Know,
                EirEpistemicOp::Possible => EpistemicAssumptionKind::Possible,
            },
            predicate: binding.predicate.clone(),
            arity: binding.arity,
            terms: Vec::new(),
            value: !binding.negated,
        };
        if binding.arity == 0 {
            resolved.insert(assumption);
        } else {
            for concrete in concrete_gpu_evidence_assumptions_for_binding(
                provider,
                result.tuple_evidence_output(),
                &assumption,
                binding,
            )? {
                resolved.insert(concrete);
            }
        }
    }

    if resolved.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: "accepted GPU tuple-membership bindings did not materialize any \
                      probabilistic evidence assumptions"
                .to_string(),
        });
    }
    Ok(resolved.into_iter().collect())
}

fn concrete_gpu_evidence_assumptions(
    provider: &CudaKernelProvider,
    result: &EpistemicGpuExecutionResult,
    assumption: &EpistemicAssumption,
    mode: EirEpistemicMode,
) -> Result<Vec<EpistemicAssumption>> {
    let candidate_bindings = result
        .tuple_membership_bindings
        .iter()
        .filter(|binding| {
            assumption_kind_matches_binding(assumption, binding, mode)
                && assumption.predicate == binding.predicate
                && assumption.value != binding.negated
        })
        .collect::<Vec<_>>();

    if candidate_bindings.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "nonzero probabilistic evidence assumption {} was not validated by any accepted \
                 GPU tuple-membership binding",
                assumption.evidence_literal()
            ),
        });
    }

    let arity_matched = candidate_bindings
        .iter()
        .copied()
        .filter(|binding| binding.arity == assumption.arity)
        .collect::<Vec<_>>();
    if arity_matched.is_empty() {
        let available_arities = candidate_bindings
            .iter()
            .map(|binding| binding.arity)
            .collect::<BTreeSet<_>>();
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "nonzero probabilistic evidence assumption {} requires arity {}, but accepted \
                 GPU tuple-membership bindings for the predicate have arities {:?}",
                assumption.evidence_literal(),
                assumption.arity,
                available_arities
            ),
        });
    }

    let mut concrete = BTreeSet::new();
    for binding in arity_matched {
        for assumption in concrete_gpu_evidence_assumptions_for_binding(
            provider,
            result.tuple_evidence_output(),
            assumption,
            binding,
        )? {
            concrete.insert(assumption);
        }
    }

    if concrete.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "nonzero probabilistic evidence assumption {} did not materialize any concrete \
                 GPU tuple evidence",
                assumption.evidence_literal()
            ),
        });
    }
    Ok(concrete.into_iter().collect())
}

fn concrete_gpu_evidence_assumptions_for_binding(
    provider: &CudaKernelProvider,
    final_output: &CudaBuffer,
    assumption: &EpistemicAssumption,
    binding: &EpistemicTupleMembershipBinding,
) -> Result<Vec<EpistemicAssumption>> {
    if binding.arity == 0 {
        return Ok(vec![assumption.clone()]);
    }
    if binding.key_terms.len() != binding.arity
        || binding.bound_output_columns.len() != binding.key_terms.len()
    {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "GPU tuple-membership binding for {}/{} has inconsistent key metadata: \
                 key_terms={} bound_output_columns={}",
                binding.predicate,
                binding.arity,
                binding.key_terms.len(),
                binding.bound_output_columns.len()
            ),
        });
    }

    let output_rows = provider.device_row_count(final_output)?;
    if output_rows == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "nonzero probabilistic evidence assumption {} requires non-empty GPU final output",
                assumption.evidence_literal()
            ),
        });
    }

    let mut output_columns = BTreeMap::new();
    for output_col in binding.bound_output_columns.iter().flatten() {
        if !output_columns.contains_key(output_col) {
            let terms =
                download_gpu_final_output_evidence_terms(provider, final_output, *output_col)?;
            if terms.len() != output_rows {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "accepted GPU world-view evidence".to_string(),
                    context: format!(
                        "GPU final-output column {} produced {} evidence terms for {} rows",
                        output_col,
                        terms.len(),
                        output_rows
                    ),
                });
            }
            output_columns.insert(*output_col, terms);
        }
    }

    let row_count = if output_columns.is_empty() {
        1
    } else {
        output_rows
    };
    let mut concrete = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let mut terms = Vec::with_capacity(binding.key_terms.len());
        for (key_term, output_col) in binding
            .key_terms
            .iter()
            .zip(binding.bound_output_columns.iter())
        {
            match key_term {
                EirTerm::Variable(_) => {
                    let Some(output_col) = *output_col else {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "accepted GPU world-view evidence".to_string(),
                            context: format!(
                                "probabilistic evidence assumption {} has an unbound variable \
                                 tuple key",
                                assumption.evidence_literal()
                            ),
                        });
                    };
                    let column_terms = output_columns.get(&output_col).ok_or_else(|| {
                        XlogError::UnsupportedEpistemicConstruct {
                            construct: "accepted GPU world-view evidence".to_string(),
                            context: format!(
                                "GPU final-output column {} was not staged for {}",
                                output_col,
                                assumption.evidence_literal()
                            ),
                        }
                    })?;
                    terms.push(column_terms[row].clone());
                }
                _ => terms.push(evidence_term_from_ground_eir_term(key_term, assumption)?),
            }
        }
        concrete.push(match assumption.kind {
            EpistemicAssumptionKind::Know => {
                EpistemicAssumption::known_tuple(&assumption.predicate, terms, assumption.value)
            }
            EpistemicAssumptionKind::Possible => {
                EpistemicAssumption::possible_tuple(&assumption.predicate, terms, assumption.value)
            }
        });
    }
    Ok(concrete)
}

fn download_gpu_final_output_evidence_terms(
    provider: &CudaKernelProvider,
    final_output: &CudaBuffer,
    output_col: usize,
) -> Result<Vec<EpistemicEvidenceTerm>> {
    let col_type = final_output
        .schema()
        .column_type(output_col)
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence references missing GPU final-output column {}",
                output_col
            ),
        })?;
    match col_type {
        ScalarType::U32 => Ok(provider
            .download_column::<u32>(final_output, output_col)?
            .into_iter()
            .map(|value| EpistemicEvidenceTerm::Integer(i64::from(value)))
            .collect()),
        ScalarType::U64 => provider
            .download_column::<u64>(final_output, output_col)?
            .into_iter()
            .map(|value| {
                i64::try_from(value)
                    .map(EpistemicEvidenceTerm::Integer)
                    .map_err(|_| XlogError::UnsupportedEpistemicConstruct {
                        construct: "accepted GPU world-view evidence".to_string(),
                        context: format!(
                            "GPU final-output column {} value {} exceeds exact evidence i64 \
                                 range",
                            output_col, value
                        ),
                    })
            })
            .collect(),
        ScalarType::I32 => Ok(provider
            .download_column::<i32>(final_output, output_col)?
            .into_iter()
            .map(|value| EpistemicEvidenceTerm::Integer(i64::from(value)))
            .collect()),
        ScalarType::I64 => Ok(provider
            .download_column::<i64>(final_output, output_col)?
            .into_iter()
            .map(EpistemicEvidenceTerm::Integer)
            .collect()),
        ScalarType::Symbol => Ok(provider
            .download_column::<u32>(final_output, output_col)?
            .into_iter()
            .map(EpistemicEvidenceTerm::Symbol)
            .collect()),
        ScalarType::Bool | ScalarType::F32 | ScalarType::F64 => {
            Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "GPU final-output column {} type {:?} cannot be used as exact epistemic \
                     evidence",
                    output_col, col_type
                ),
            })
        }
    }
}

fn evidence_term_from_ground_eir_term(
    term: &EirTerm,
    assumption: &EpistemicAssumption,
) -> Result<EpistemicEvidenceTerm> {
    match term {
        EirTerm::Integer(value) => Ok(EpistemicEvidenceTerm::Integer(*value)),
        EirTerm::String(value) => Ok(EpistemicEvidenceTerm::String(value.clone())),
        EirTerm::Symbol(value) => Ok(EpistemicEvidenceTerm::Symbol(*value)),
        EirTerm::Variable(_)
        | EirTerm::Anonymous
        | EirTerm::FloatBits(_)
        | EirTerm::List(_)
        | EirTerm::Cons { .. }
        | EirTerm::Compound { .. }
        | EirTerm::PredRef(_)
        | EirTerm::Aggregate { .. } => Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumption {} uses unsupported tuple key term {:?}",
                assumption.evidence_literal(),
                term
            ),
        }),
    }
}

struct GpuEvidenceBindingMatch<'a> {
    binding: &'a EpistemicTupleMembershipBinding,
    accepted_assumption: EpistemicAssumption,
    matched_concrete_tuple_key: bool,
}

fn find_gpu_evidence_binding<'a>(
    provider: &CudaKernelProvider,
    result: &'a EpistemicGpuExecutionResult,
    assumption: &EpistemicAssumption,
    mode: EirEpistemicMode,
) -> Result<Option<GpuEvidenceBindingMatch<'a>>> {
    let mut saw_final_tuple_miss = false;
    for binding in result
        .tuple_membership_bindings
        .iter()
        .filter(|binding| assumption_matches_gpu_binding(assumption, binding, mode))
    {
        if assumption.arity > 0
            && !assumption.terms.is_empty()
            && binding.bound_output_columns.iter().any(Option::is_some)
        {
            let matched_rows = gpu_final_output_rows_matching_assumption(
                provider,
                result.tuple_evidence_output(),
                assumption,
                binding,
            )?;
            if matched_rows == 0 {
                saw_final_tuple_miss = true;
                continue;
            }
        }
        return Ok(Some(GpuEvidenceBindingMatch {
            binding,
            accepted_assumption: assumption.clone(),
            matched_concrete_tuple_key: !assumption.terms.is_empty(),
        }));
    }
    if saw_final_tuple_miss {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumption {} did not match any GPU-materialized final \
                 tuple row",
                assumption.evidence_literal()
            ),
        });
    }
    Ok(None)
}

fn gpu_final_output_rows_matching_assumption(
    provider: &CudaKernelProvider,
    final_output: &CudaBuffer,
    assumption: &EpistemicAssumption,
    binding: &EpistemicTupleMembershipBinding,
) -> Result<usize> {
    let mut filtered: Option<CudaBuffer> = None;
    let mut checked_variable_terms = 0usize;

    for ((assumption_term, binding_term), bound_output_column) in assumption
        .terms
        .iter()
        .zip(binding.key_terms.iter())
        .zip(binding.bound_output_columns.iter())
    {
        if !matches!(binding_term, EirTerm::Variable(_)) {
            continue;
        }
        let Some(output_col) = *bound_output_column else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence assumption {} has an unbound variable tuple key",
                    assumption.evidence_literal()
                ),
            });
        };
        let input = filtered.as_ref().unwrap_or(final_output);
        filtered = Some(filter_gpu_final_output_by_evidence_term(
            provider,
            input,
            output_col,
            assumption_term,
            assumption,
        )?);
        checked_variable_terms += 1;
    }

    if checked_variable_terms == 0 {
        return provider.device_row_count(final_output);
    }
    let filtered = filtered
        .as_ref()
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumption {} did not produce a GPU final-output filter",
                assumption.evidence_literal()
            ),
        })?;
    provider.device_row_count(filtered)
}

fn filter_gpu_final_output_by_evidence_term(
    provider: &CudaKernelProvider,
    input: &CudaBuffer,
    output_col: usize,
    term: &EpistemicEvidenceTerm,
    assumption: &EpistemicAssumption,
) -> Result<CudaBuffer> {
    let col_type = input.schema().column_type(output_col).ok_or_else(|| {
        XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence assumption {} references missing final-output column {}",
                assumption.evidence_literal(),
                output_col
            ),
        }
    })?;

    match (col_type, term) {
        (ScalarType::U32, EpistemicEvidenceTerm::Integer(value)) => {
            let value = u32::try_from(*value)
                .map_err(|_| evidence_term_type_error(assumption, output_col, col_type, term))?;
            provider.filter::<u32>(input, output_col, value, CompareOp::Eq)
        }
        (ScalarType::U64, EpistemicEvidenceTerm::Integer(value)) => {
            let value = u64::try_from(*value)
                .map_err(|_| evidence_term_type_error(assumption, output_col, col_type, term))?;
            provider.filter::<u64>(input, output_col, value, CompareOp::Eq)
        }
        (ScalarType::I32, EpistemicEvidenceTerm::Integer(value)) => {
            let value = i32::try_from(*value)
                .map_err(|_| evidence_term_type_error(assumption, output_col, col_type, term))?;
            provider.filter::<i32>(input, output_col, value, CompareOp::Eq)
        }
        (ScalarType::I64, EpistemicEvidenceTerm::Integer(value)) => {
            provider.filter::<i64>(input, output_col, *value, CompareOp::Eq)
        }
        (ScalarType::Symbol, EpistemicEvidenceTerm::Symbol(value)) => {
            provider.filter::<u32>(input, output_col, *value, CompareOp::Eq)
        }
        (ScalarType::Symbol, EpistemicEvidenceTerm::String(value)) => {
            let value = symbol::intern(value);
            provider.filter::<u32>(input, output_col, value, CompareOp::Eq)
        }
        _ => Err(evidence_term_type_error(
            assumption, output_col, col_type, term,
        )),
    }
}

fn evidence_term_type_error(
    assumption: &EpistemicAssumption,
    output_col: usize,
    col_type: ScalarType,
    term: &EpistemicEvidenceTerm,
) -> XlogError {
    XlogError::UnsupportedEpistemicConstruct {
        construct: "accepted GPU world-view evidence".to_string(),
        context: format!(
            "probabilistic evidence assumption {} term {:?} is incompatible with \
             GPU final-output column {} type {:?}",
            assumption.evidence_literal(),
            term,
            output_col,
            col_type
        ),
    }
}

fn assumption_matches_gpu_binding(
    assumption: &EpistemicAssumption,
    binding: &EpistemicTupleMembershipBinding,
    mode: EirEpistemicMode,
) -> bool {
    if assumption_kind_matches_binding(assumption, binding, mode)
        && assumption.predicate == binding.predicate
        && assumption.arity == binding.arity
        && assumption.value != binding.negated
    {
        assumption_terms_match_binding(assumption, binding)
    } else {
        false
    }
}

fn assumption_kind_matches_op(kind: EpistemicAssumptionKind, op: EirEpistemicOp) -> bool {
    matches!(
        (kind, op),
        (EpistemicAssumptionKind::Know, EirEpistemicOp::Know)
            | (EpistemicAssumptionKind::Possible, EirEpistemicOp::Possible)
    )
}

fn assumption_kind_matches_binding(
    assumption: &EpistemicAssumption,
    binding: &EpistemicTupleMembershipBinding,
    mode: EirEpistemicMode,
) -> bool {
    assumption_kind_matches_op(assumption.kind, binding.op)
        || (matches!(mode, EirEpistemicMode::Faeel)
            && assumption.kind == EpistemicAssumptionKind::Know
            && assumption.value
            && !binding.negated
            && matches!(binding.op, EirEpistemicOp::Possible))
}

fn assumption_terms_match_binding(
    assumption: &EpistemicAssumption,
    binding: &EpistemicTupleMembershipBinding,
) -> bool {
    if assumption.arity == 0 {
        return assumption.terms.is_empty() && binding.key_terms.is_empty();
    }
    if assumption.terms.is_empty() {
        return false;
    }
    if assumption.terms.len() != binding.key_terms.len()
        || binding.bound_output_columns.len() != binding.key_terms.len()
    {
        return false;
    }

    assumption
        .terms
        .iter()
        .zip(binding.key_terms.iter())
        .zip(binding.bound_output_columns.iter())
        .all(
            |((assumption_term, binding_term), bound_output_column)| match binding_term {
                EirTerm::Variable(_) => bound_output_column.is_some(),
                EirTerm::Integer(value) => {
                    matches!(assumption_term, EpistemicEvidenceTerm::Integer(v) if v == value)
                }
                EirTerm::String(value) => {
                    matches!(assumption_term, EpistemicEvidenceTerm::String(v) if v == value)
                }
                EirTerm::Symbol(value) => {
                    matches!(assumption_term, EpistemicEvidenceTerm::Symbol(v) if v == value)
                }
                EirTerm::Anonymous
                | EirTerm::FloatBits(_)
                | EirTerm::List(_)
                | EirTerm::Cons { .. }
                | EirTerm::Compound { .. }
                | EirTerm::PredRef(_)
                | EirTerm::Aggregate { .. } => false,
            },
        )
}

fn require_gpu_kernel_trace(
    phase: &'static str,
    kernel_launches: u32,
    host_write_ops: u32,
    kernel_timing: EpistemicGpuKernelTimingTrace,
) -> Result<()> {
    if kernel_launches == 0 || host_write_ops != 0 || !kernel_timing.is_recorded() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence requires GPU {phase} trace with nonzero launches and \
                 zero host writes plus CUDA-event timing, got launches={kernel_launches}, \
                 host_writes={host_write_ops}, timing_recorded={}",
                kernel_timing.is_recorded()
            ),
        });
    }
    Ok(())
}

/// Deterministic probability value with a comparison tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityValue {
    /// Probability value after normalization.
    pub probability: f64,
    /// Absolute tolerance for comparisons.
    pub tolerance: f64,
}

impl ProbabilityValue {
    /// Return true when this probability is within tolerance of `expected`.
    pub fn within_tolerance(&self, expected: f64) -> bool {
        (self.probability - expected).abs() <= self.tolerance
    }
}

/// Bounded circuit state for epistemic/probabilistic fixtures.
#[derive(Debug, Clone)]
pub struct EpistemicCircuit {
    adapter: KnowledgeCompilerAdapter,
    base_probability: f64,
    conditioned_probabilities: BTreeMap<EpistemicAssumption, f64>,
    active_assumptions: BTreeSet<EpistemicAssumption>,
    compile_count: usize,
    incremental_update_count: usize,
    circuit_fingerprint: u64,
    tolerance: f64,
}

impl EpistemicCircuit {
    /// Compile a bounded circuit fixture with optional assumption-conditioned probabilities.
    pub fn compile(
        base_probability: f64,
        conditioned_probabilities: Vec<(EpistemicAssumption, f64)>,
        adapter: KnowledgeCompilerAdapter,
    ) -> Result<Self> {
        let base_probability = normalize_probability(
            base_probability,
            EPISTEMIC_PROBABILITY_TOLERANCE,
            "epistemic base probability",
        )?;
        let mut conditioned = BTreeMap::new();
        for (assumption, probability) in conditioned_probabilities {
            let probability = normalize_probability(
                probability,
                EPISTEMIC_PROBABILITY_TOLERANCE,
                "epistemic conditioned probability",
            )?;
            conditioned.insert(assumption, probability);
        }

        let active_assumptions = BTreeSet::new();
        let circuit_fingerprint = circuit_fingerprint(
            &adapter,
            base_probability,
            &conditioned,
            &active_assumptions,
        );

        Ok(Self {
            adapter,
            base_probability,
            conditioned_probabilities: conditioned,
            active_assumptions,
            compile_count: 1,
            incremental_update_count: 0,
            circuit_fingerprint,
            tolerance: EPISTEMIC_PROBABILITY_TOLERANCE,
        })
    }

    /// Return the semantic contract for this circuit.
    pub fn semantic_contract(&self) -> EpistemicProbabilisticContract {
        EpistemicProbabilisticContract::default()
    }

    /// Return active compiler evidence literals in deterministic order.
    pub fn compiler_evidence_literals(&self) -> Vec<String> {
        self.active_assumptions
            .iter()
            .map(EpistemicAssumption::evidence_literal)
            .collect()
    }

    /// Return the current query probability.
    pub fn query_probability(&self) -> ProbabilityValue {
        let probability = self
            .active_assumptions
            .iter()
            .find_map(|assumption| self.conditioned_probabilities.get(assumption))
            .copied()
            .unwrap_or(self.base_probability);

        ProbabilityValue {
            probability,
            tolerance: self.tolerance,
        }
    }

    /// Apply an epistemic assumption as probabilistic evidence.
    pub fn apply_assumption(&mut self, assumption: EpistemicAssumption) -> Result<CircuitUpdate> {
        if self.active_assumptions.contains(&assumption) {
            return Ok(self.update_result(CircuitUpdateMode::Unchanged));
        }

        let stale_assumptions = self
            .active_assumptions
            .iter()
            .filter(|active| active.same_evidence_key(&assumption))
            .cloned()
            .collect::<Vec<_>>();
        for stale in stale_assumptions {
            self.active_assumptions.remove(&stale);
        }
        self.active_assumptions.insert(assumption);

        if self.adapter.supports_incremental_evidence() {
            self.incremental_update_count += 1;
            return Ok(self.update_result(CircuitUpdateMode::IncrementalEvidence));
        }

        self.compile_count += 1;
        self.circuit_fingerprint = circuit_fingerprint(
            &self.adapter,
            self.base_probability,
            &self.conditioned_probabilities,
            &self.active_assumptions,
        );
        Ok(self.update_result(CircuitUpdateMode::FullRebuild))
    }

    /// Apply epistemic evidence that has already passed world-view validation.
    pub fn apply_accepted_world_view(
        &mut self,
        evidence: AcceptedWorldViewEvidence,
    ) -> Result<CircuitUpdate> {
        let mut mode = CircuitUpdateMode::Unchanged;
        for assumption in evidence.assumptions {
            let update = self.apply_assumption(assumption)?;
            mode = combine_update_modes(mode, update.mode);
        }
        Ok(CircuitUpdate {
            mode,
            compile_count: self.compile_count,
            circuit_fingerprint: self.circuit_fingerprint,
        })
    }

    /// Return the stable circuit fingerprint.
    pub fn circuit_fingerprint(&self) -> u64 {
        self.circuit_fingerprint
    }

    /// Return the number of incremental evidence updates applied.
    pub fn incremental_update_count(&self) -> usize {
        self.incremental_update_count
    }

    fn update_result(&self, mode: CircuitUpdateMode) -> CircuitUpdate {
        CircuitUpdate {
            mode,
            compile_count: self.compile_count,
            circuit_fingerprint: self.circuit_fingerprint,
        }
    }
}

/// Convert log-space `P(query and evidence)` and `P(evidence)` into `P(query | evidence)`.
pub fn conditional_probability_from_logs(
    log_joint: f64,
    log_evidence: f64,
    tolerance: f64,
) -> Result<ProbabilityValue> {
    validate_tolerance(tolerance)?;
    if !log_joint.is_finite() || !log_evidence.is_finite() {
        return Err(XlogError::Compilation(
            "epistemic probability logs must be finite".to_string(),
        ));
    }

    let raw = (log_joint - log_evidence).exp();
    Ok(ProbabilityValue {
        probability: normalize_probability(raw, tolerance, "epistemic conditional probability")?,
        tolerance,
    })
}

fn normalize_probability(value: f64, tolerance: f64, context: &str) -> Result<f64> {
    validate_tolerance(tolerance)?;
    if !value.is_finite() {
        return Err(XlogError::Compilation(format!(
            "{context} must be finite, got {value}"
        )));
    }
    if value < 0.0 {
        if value >= -tolerance {
            return Ok(0.0);
        }
        return Err(XlogError::Compilation(format!(
            "{context} below 0 by more than tolerance: {value}"
        )));
    }
    if value > 1.0 {
        if value <= 1.0 + tolerance {
            return Ok(1.0);
        }
        return Err(XlogError::Compilation(format!(
            "{context} above 1 by more than tolerance: {value}"
        )));
    }
    Ok(value)
}

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance >= 0.0 {
        Ok(())
    } else {
        Err(XlogError::Compilation(format!(
            "epistemic probability tolerance must be finite and non-negative, got {tolerance}"
        )))
    }
}

fn combine_update_modes(left: CircuitUpdateMode, right: CircuitUpdateMode) -> CircuitUpdateMode {
    match (left, right) {
        (CircuitUpdateMode::FullRebuild, _) | (_, CircuitUpdateMode::FullRebuild) => {
            CircuitUpdateMode::FullRebuild
        }
        (CircuitUpdateMode::IncrementalEvidence, _)
        | (_, CircuitUpdateMode::IncrementalEvidence) => CircuitUpdateMode::IncrementalEvidence,
        (CircuitUpdateMode::Unchanged, CircuitUpdateMode::Unchanged) => {
            CircuitUpdateMode::Unchanged
        }
    }
}

fn circuit_fingerprint(
    adapter: &KnowledgeCompilerAdapter,
    base_probability: f64,
    conditioned_probabilities: &BTreeMap<EpistemicAssumption, f64>,
    active_assumptions: &BTreeSet<EpistemicAssumption>,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    mix_u64(&mut hash, adapter.kind as u64);
    mix_u64(&mut hash, adapter.support as u64);
    mix_str(&mut hash, &adapter.name);
    mix_u64(&mut hash, base_probability.to_bits());
    for (assumption, probability) in conditioned_probabilities {
        mix_assumption(&mut hash, assumption);
        mix_u64(&mut hash, probability.to_bits());
    }
    for assumption in active_assumptions {
        mix_assumption(&mut hash, assumption);
    }
    hash
}

fn mix_assumption(hash: &mut u64, assumption: &EpistemicAssumption) {
    mix_u64(hash, assumption.kind as u64);
    mix_str(hash, &assumption.predicate);
    mix_u64(hash, assumption.arity as u64);
    for term in &assumption.terms {
        mix_evidence_term(hash, term);
    }
    mix_u64(hash, u64::from(assumption.value));
}

fn mix_evidence_term(hash: &mut u64, term: &EpistemicEvidenceTerm) {
    match term {
        EpistemicEvidenceTerm::Integer(value) => {
            mix_u64(hash, 0);
            mix_u64(hash, *value as u64);
        }
        EpistemicEvidenceTerm::String(value) => {
            mix_u64(hash, 1);
            mix_str(hash, value);
        }
        EpistemicEvidenceTerm::Symbol(value) => {
            mix_u64(hash, 2);
            mix_u64(hash, u64::from(*value));
        }
    }
}

fn mix_str(hash: &mut u64, value: &str) {
    for byte in value.as_bytes() {
        mix_u64(hash, u64::from(*byte));
    }
}

fn mix_u64(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
}
