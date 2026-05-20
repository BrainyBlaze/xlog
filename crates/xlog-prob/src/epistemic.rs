//! Bounded epistemic/probabilistic integration helpers for v0.9 fixtures.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_ir::EirEpistemicMode;
use xlog_logic::epistemic::EpistemicWorldView;
use xlog_runtime::{read_device_row_count, EpistemicGpuExecutionResult};

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
}

impl AcceptedWorldViewEvidence {
    /// Construct evidence from a non-empty accepted world view.
    pub fn new(
        world_view: &EpistemicWorldView,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<Self> {
        Ok(Self {
            assumptions,
            world_count: world_view.world_count(),
            gpu_epistemic_mode: None,
        })
    }

    /// Construct evidence from an accepted GPU epistemic execution result.
    ///
    /// This is the production boundary used by probabilistic adapters: it
    /// accepts only results that used stable-model tuple membership, GPU
    /// world-view/final-result/final-tuple kernels, zero hot-path host
    /// transfers, and a non-empty device final output.
    pub fn from_gpu_execution_result(
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<Self> {
        if !result.prepared.preflight.cpu_fallbacks.is_zero() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: "probabilistic evidence requires zero epistemic CPU fallback counters"
                    .to_string(),
            });
        }
        result
            .model_membership
            .require_stable_model_tuple_source()?;
        require_gpu_kernel_trace(
            "model membership",
            result.model_membership.kernel_launches,
            result.model_membership.host_write_ops,
        )?;
        require_gpu_kernel_trace(
            "world-view validation",
            result.world_view_validation.kernel_launches,
            result.world_view_validation.host_write_ops,
        )?;
        require_gpu_kernel_trace(
            "final-result materialization",
            result.final_result_materialization.kernel_launches,
            result.final_result_materialization.host_write_ops,
        )?;
        require_gpu_kernel_trace(
            "final tuple materialization",
            result.final_tuple_materialization.kernel_launches,
            result.final_tuple_materialization.host_write_ops,
        )?;
        if result.transfer_budget.tracked_dtoh_calls != 0
            || result.transfer_budget.tracked_htod_calls != 0
            || result.transfer_budget.per_candidate_host_round_trips != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: format!(
                    "probabilistic evidence requires zero hot-path transfers, got dtoh_calls={}, \
                     htod_calls={}, per_candidate_round_trips={}",
                    result.transfer_budget.tracked_dtoh_calls,
                    result.transfer_budget.tracked_htod_calls,
                    result.transfer_budget.per_candidate_host_round_trips
                ),
            });
        }

        let accepted_rows = read_device_row_count(provider, &result.final_output)?;
        if accepted_rows == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted GPU world-view evidence".to_string(),
                context: "probabilistic evidence requires non-empty accepted GPU final output"
                    .to_string(),
            });
        }

        Ok(Self {
            assumptions,
            world_count: accepted_rows,
            gpu_epistemic_mode: Some(result.prepared.preflight.epistemic_mode),
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

    /// Epistemic mode reported by the accepted GPU runtime evidence, when present.
    pub fn gpu_epistemic_mode(&self) -> Option<EirEpistemicMode> {
        self.gpu_epistemic_mode
    }

    /// Number of accepted epistemic assumptions represented by this evidence.
    pub fn assumption_count(&self) -> usize {
        self.assumptions.len()
    }
}

fn require_gpu_kernel_trace(
    phase: &'static str,
    kernel_launches: u32,
    host_write_ops: u32,
) -> Result<()> {
    if kernel_launches == 0 || host_write_ops != 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted GPU world-view evidence".to_string(),
            context: format!(
                "probabilistic evidence requires GPU {phase} trace with nonzero launches and \
                 zero host writes, got launches={kernel_launches}, host_writes={host_write_ops}"
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
    mix_u64(hash, u64::from(assumption.value));
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
