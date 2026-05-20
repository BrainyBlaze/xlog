//! Production GPU exact-path adapter for accepted epistemic evidence.
//!
//! This module is intentionally thin. It gates probabilistic execution on
//! accepted world-view evidence, then routes into the existing GPU-native exact
//! provenance path instead of using the bounded epistemic fixture circuit.

use std::collections::BTreeSet;
use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_ir::EirEpistemicMode;
use xlog_logic::ast::Program;
#[cfg(feature = "host-io")]
use xlog_logic::ast::{Atom, Evidence, Term};
#[cfg(feature = "host-io")]
use xlog_logic::parse_program;
use xlog_runtime::{EpistemicGpuBatchExecutionResult, EpistemicGpuExecutionResult};

use crate::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
#[cfg(feature = "host-io")]
use crate::epistemic::EpistemicAssumptionKind;
#[cfg(feature = "host-io")]
use crate::epistemic::EpistemicEvidenceTerm;
use crate::epistemic::{
    AcceptedWorldViewEvidence, CircuitUpdate, CircuitUpdateMode, EpistemicAssumption,
    EpistemicCircuit,
};
use crate::exact::{ExactDdnnfProgram, GpuConfig};
#[cfg(feature = "host-io")]
use crate::exact::{ExactResult, ExactResultWithGrads};
use crate::pir::{PirNode, PirNodeId};
use crate::provenance::{extract_from_program, extract_from_source, Provenance};

/// Production capability status for probabilistic paths required by v0.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicProbProductionCapabilityStatus {
    /// Existing GPU-native production path is available.
    Available,
    /// Required GPU-native production path is not implemented.
    Blocked,
}

/// Capability report for the probabilistic production adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicProbProductionCapabilities {
    /// Exact/provenance compilation through `ExactDdnnfProgram`.
    pub gpu_exact_provenance: EpistemicProbProductionCapabilityStatus,
    /// GPU PIR upload and CNF encoding path.
    pub gpu_pir_cnf: EpistemicProbProductionCapabilityStatus,
    /// Bounded compile-plus-evaluate knowledge-compilation path.
    pub gpu_knowledge_compilation: EpistemicProbProductionCapabilityStatus,
    /// GPU query and gradient evaluation path.
    pub gpu_exact_query_and_gradient: EpistemicProbProductionCapabilityStatus,
    /// Whether the bounded fixture circuit may satisfy production metrics.
    pub fixture_circuit_allowed: bool,
    /// Blocker reason for knowledge-compilation production coverage, or empty when available.
    pub gpu_knowledge_compilation_blocker: &'static str,
}

/// Return the current probabilistic production capability report.
pub fn production_capabilities() -> EpistemicProbProductionCapabilities {
    EpistemicProbProductionCapabilities {
        gpu_exact_provenance: EpistemicProbProductionCapabilityStatus::Available,
        gpu_pir_cnf: EpistemicProbProductionCapabilityStatus::Available,
        gpu_knowledge_compilation: EpistemicProbProductionCapabilityStatus::Available,
        gpu_exact_query_and_gradient: EpistemicProbProductionCapabilityStatus::Available,
        fixture_circuit_allowed: false,
        gpu_knowledge_compilation_blocker: "",
    }
}

/// Trace counters proving the production adapter stayed on the GPU exact path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicProbProductionTrace {
    /// Number of source compiles routed through `ExactDdnnfProgram`.
    pub gpu_exact_source_compiles: u64,
    /// Number of parsed-program compiles routed through `ExactDdnnfProgram`.
    pub gpu_exact_program_compiles: u64,
    /// Number of accepted world-view evidence objects consumed as a gate.
    pub accepted_world_view_evidence_consumed: u64,
    /// Number of accepted G91 GPU world-view evidence objects consumed as a gate.
    pub accepted_g91_world_view_evidence_consumed: u64,
    /// Number of accepted FAEEL GPU world-view evidence objects consumed as a gate.
    pub accepted_faeel_world_view_evidence_consumed: u64,
    /// Number of accepted epistemic assumptions consumed from world-view evidence.
    pub accepted_evidence_assumptions_consumed: u64,
    /// Number of accepted GPU batch evidence records consumed as a gate.
    pub accepted_gpu_batch_evidence_consumed: u64,
    /// Number of accepted GPU batch components consumed as individual evidence records.
    pub accepted_gpu_batch_component_evidence_consumed: u64,
    /// Number of accepted evidence applications that updated caller-owned incremental circuits.
    ///
    /// This is fixture coverage only and is intentionally excluded from production path events.
    pub accepted_incremental_circuit_updates: u64,
    /// Number of GPU exact query evaluations routed through `ExactDdnnfProgram`.
    pub gpu_exact_query_evaluations: u64,
    /// Number of source GPU exact query evaluations routed through `ExactDdnnfProgram`.
    pub gpu_source_exact_query_evaluations: u64,
    /// Number of parsed-program GPU exact query evaluations routed through `ExactDdnnfProgram`.
    pub gpu_program_exact_query_evaluations: u64,
    /// Number of GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_exact_gradient_evaluations: u64,
    /// Number of source-conditioned GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_source_conditioned_gradient_evaluations: u64,
    /// Number of parsed-program-conditioned GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_program_conditioned_gradient_evaluations: u64,
    /// Number of accepted PIR graphs uploaded through the existing GPU PIR layout.
    pub gpu_pir_graph_uploads: u64,
    /// Number of source accepted PIR graphs uploaded through the existing GPU PIR layout.
    pub gpu_source_pir_graph_uploads: u64,
    /// Number of parsed-program accepted PIR graphs uploaded through the existing GPU PIR layout.
    pub gpu_program_pir_graph_uploads: u64,
    /// Number of accepted PIR root sets encoded through the existing GPU CNF encoder.
    pub gpu_cnf_encodes: u64,
    /// Number of source accepted PIR root sets encoded through the existing GPU CNF encoder.
    pub gpu_source_cnf_encodes: u64,
    /// Number of parsed-program accepted PIR root sets encoded through the existing GPU CNF encoder.
    pub gpu_program_cnf_encodes: u64,
    /// Number of accepted compile-and-evaluate runs through the GPU exact path.
    pub gpu_knowledge_compilation_end_to_end_runs: u64,
    /// Number of accepted source compile-and-evaluate runs through the GPU exact path.
    pub gpu_source_knowledge_compilation_end_to_end_runs: u64,
    /// Number of accepted parsed-program compile-and-evaluate runs through the GPU exact path.
    pub gpu_program_knowledge_compilation_end_to_end_runs: u64,
    /// Number of accepted assumptions compiled as exact evidence facts.
    pub gpu_conditioned_evidence_facts: u64,
    /// Number of accepted nonzero-arity assumptions compiled as exact evidence facts.
    pub gpu_conditioned_nonzero_arity_evidence_facts: u64,
    /// Maximum accepted exact evidence tuple arity observed across conditioned paths.
    pub gpu_conditioned_max_evidence_arity: u32,
    /// Number of false accepted assumptions compiled as exact evidence facts.
    pub gpu_conditioned_negative_evidence_facts: u64,
    /// Number of source-conditioned accepted assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_evidence_facts: u64,
    /// Number of source-conditioned nonzero-arity assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_nonzero_arity_evidence_facts: u64,
    /// Maximum source-conditioned accepted exact evidence tuple arity observed.
    pub gpu_source_conditioned_max_evidence_arity: u32,
    /// Number of parsed-program-conditioned accepted assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_evidence_facts: u64,
    /// Number of parsed-program-conditioned nonzero-arity assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_nonzero_arity_evidence_facts: u64,
    /// Maximum parsed-program-conditioned accepted exact evidence tuple arity observed.
    pub gpu_program_conditioned_max_evidence_arity: u32,
    /// Number of false source-conditioned assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_negative_evidence_facts: u64,
    /// Number of false parsed-program-conditioned assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_negative_evidence_facts: u64,
    /// Number of true `know` assumptions compiled as exact evidence facts.
    pub gpu_conditioned_know_evidence_facts: u64,
    /// Number of true `possible` assumptions compiled as exact evidence facts.
    pub gpu_conditioned_possible_evidence_facts: u64,
    /// Number of false `know` assumptions compiled as exact evidence facts.
    pub gpu_conditioned_not_known_evidence_facts: u64,
    /// Number of false `possible` assumptions compiled as exact evidence facts.
    pub gpu_conditioned_not_possible_evidence_facts: u64,
    /// Number of source-conditioned true `know` assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_know_evidence_facts: u64,
    /// Number of source-conditioned true `possible` assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_possible_evidence_facts: u64,
    /// Number of source-conditioned false `know` assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_not_known_evidence_facts: u64,
    /// Number of source-conditioned false `possible` assumptions compiled as exact evidence facts.
    pub gpu_source_conditioned_not_possible_evidence_facts: u64,
    /// Number of parsed-program-conditioned true `know` assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_know_evidence_facts: u64,
    /// Number of parsed-program-conditioned true `possible` assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_possible_evidence_facts: u64,
    /// Number of parsed-program-conditioned false `know` assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_not_known_evidence_facts: u64,
    /// Number of parsed-program-conditioned false `possible` assumptions compiled as exact evidence facts.
    pub gpu_program_conditioned_not_possible_evidence_facts: u64,
    /// CPU-only probability recomputations performed by this adapter.
    pub cpu_only_probability_recomputations: u64,
    /// Fixture `EpistemicCircuit` evaluations performed by this adapter.
    pub fixture_circuit_evaluations: u64,
}

impl EpistemicProbProductionTrace {
    fn gpu_production_path_events(&self) -> u64 {
        self.gpu_exact_source_compiles
            .saturating_add(self.gpu_exact_program_compiles)
            .saturating_add(self.gpu_exact_query_evaluations)
            .saturating_add(self.gpu_source_exact_query_evaluations)
            .saturating_add(self.gpu_program_exact_query_evaluations)
            .saturating_add(self.gpu_exact_gradient_evaluations)
            .saturating_add(self.gpu_source_conditioned_gradient_evaluations)
            .saturating_add(self.gpu_program_conditioned_gradient_evaluations)
            .saturating_add(self.gpu_pir_graph_uploads)
            .saturating_add(self.gpu_source_pir_graph_uploads)
            .saturating_add(self.gpu_program_pir_graph_uploads)
            .saturating_add(self.gpu_cnf_encodes)
            .saturating_add(self.gpu_source_cnf_encodes)
            .saturating_add(self.gpu_program_cnf_encodes)
            .saturating_add(self.gpu_knowledge_compilation_end_to_end_runs)
            .saturating_add(self.gpu_source_knowledge_compilation_end_to_end_runs)
            .saturating_add(self.gpu_program_knowledge_compilation_end_to_end_runs)
    }

    /// Require that no CPU-only probability recomputation counters were used.
    pub fn require_zero_cpu_recompute(&self) -> Result<()> {
        if self.cpu_only_probability_recomputations != 0 || self.fixture_circuit_evaluations != 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production adapter".to_string(),
                context: format!(
                    "CPU probabilistic fallback counters must be zero, got recompute={} fixture={}",
                    self.cpu_only_probability_recomputations, self.fixture_circuit_evaluations
                ),
            });
        }
        Ok(())
    }

    /// Require that this trace is eligible for v0.9 production probability metrics.
    ///
    /// This gate only proves fixture containment for an accepted probabilistic
    /// path. It does not claim the broader G090 probabilistic goal is complete.
    pub fn require_production_metric_eligibility(&self) -> Result<()> {
        let capabilities = production_capabilities();
        if capabilities.fixture_circuit_allowed {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "bounded EpistemicCircuit fixtures are not allowed for production metrics"
                    .to_string(),
            });
        }
        if capabilities.gpu_exact_provenance != EpistemicProbProductionCapabilityStatus::Available {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "GPU exact/provenance production capability is not available".to_string(),
            });
        }
        if capabilities.gpu_pir_cnf != EpistemicProbProductionCapabilityStatus::Available {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "GPU PIR/CNF production capability is not available".to_string(),
            });
        }
        if capabilities.gpu_knowledge_compilation
            != EpistemicProbProductionCapabilityStatus::Available
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: capabilities.gpu_knowledge_compilation_blocker.to_string(),
            });
        }
        if capabilities.gpu_exact_query_and_gradient
            != EpistemicProbProductionCapabilityStatus::Available
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "GPU exact query/gradient production capability is not available"
                    .to_string(),
            });
        }
        if self.accepted_world_view_evidence_consumed == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "production probability metrics require accepted world-view evidence"
                    .to_string(),
            });
        }
        if self.gpu_production_path_events() == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "production probability metrics require an existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"
                    .to_string(),
            });
        }
        self.require_zero_cpu_recompute()
    }
}

/// Device-side PIR/CNF evidence produced after accepted epistemic gating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicProbPirCnfEvidence {
    /// Number of host provenance PIR nodes uploaded to the GPU PIR layout.
    pub pir_nodes: usize,
    /// Number of roots supplied to GPU CNF encoding.
    pub root_count: usize,
    /// GPU CNF variable capacity emitted by `encode_cnf_gpu`.
    pub cnf_var_cap: u32,
    /// GPU CNF clause capacity emitted by `encode_cnf_gpu`.
    pub cnf_clause_cap: u32,
    /// GPU CNF literal capacity emitted by `encode_cnf_gpu`.
    pub cnf_lit_cap: u32,
}

/// One accepted GPU epistemic execution record used for probabilistic production gating.
#[derive(Clone, Copy)]
pub struct EpistemicProbGpuExecutionEvidence<'a> {
    /// Accepted GPU execution result whose world-view boundary must be validated.
    pub result: &'a EpistemicGpuExecutionResult,
    /// Epistemic assumptions represented by the accepted world view.
    pub assumptions: &'a [EpistemicAssumption],
}

/// Accepted GPU batch execution evidence used for probabilistic production gating.
pub struct EpistemicProbGpuBatchExecutionEvidence<'a> {
    /// Accepted GPU batch execution result whose aggregate trace and timing must be validated.
    pub batch: &'a EpistemicGpuBatchExecutionResult,
    /// Epistemic assumptions represented by each accepted component world view.
    pub assumptions_by_component: &'a [&'a [EpistemicAssumption]],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpistemicProbPirCnfPath {
    Source,
    Program,
}

/// Thin adapter from accepted epistemic evidence to the existing GPU exact path.
pub struct EpistemicProbProductionAdapter {
    config: GpuConfig,
    trace: EpistemicProbProductionTrace,
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpistemicProbConditionedEvidencePath {
    Source,
    Program,
}

impl EpistemicProbProductionAdapter {
    /// Create a production adapter with a GPU exact inference configuration.
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            trace: EpistemicProbProductionTrace {
                cpu_only_probability_recomputations: 0,
                fixture_circuit_evaluations: 0,
                ..EpistemicProbProductionTrace::default()
            },
        }
    }

    /// Return current production-path trace counters.
    pub fn trace(&self) -> EpistemicProbProductionTrace {
        self.trace
    }

    /// Apply accepted world-view evidence to a caller-owned incremental circuit fixture.
    ///
    /// This records the accepted evidence boundary and zero-CPU guard, but it is not a
    /// production metric event. Production metric eligibility still requires an
    /// existing GPU exact/provenance/PIR/CNF/knowledge-compilation path counter.
    pub fn apply_accepted_world_view_to_circuit(
        &mut self,
        circuit: &mut EpistemicCircuit,
        evidence: AcceptedWorldViewEvidence,
    ) -> Result<CircuitUpdate> {
        self.consume_accepted_evidence(&evidence)?;
        let update = circuit.apply_accepted_world_view(evidence)?;
        if update.mode == CircuitUpdateMode::IncrementalEvidence {
            self.trace.accepted_incremental_circuit_updates = self
                .trace
                .accepted_incremental_circuit_updates
                .saturating_add(1);
        }
        self.trace.require_zero_cpu_recompute()?;
        Ok(update)
    }

    /// Apply accepted GPU epistemic execution evidence to a caller-owned incremental circuit.
    pub fn apply_accepted_world_view_to_circuit_with_gpu_execution_result(
        &mut self,
        circuit: &mut EpistemicCircuit,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<CircuitUpdate> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.apply_accepted_world_view_to_circuit(circuit, evidence)
    }

    /// Apply accepted split/batch GPU epistemic execution evidence to an incremental circuit.
    pub fn apply_accepted_world_views_to_circuit_for_gpu_batch_execution_result(
        &mut self,
        circuit: &mut EpistemicCircuit,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<CircuitUpdate>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic incremental circuit batch production",
        )?;

        let mut updates = Vec::with_capacity(accepted.len());
        for evidence in accepted {
            updates.push(self.apply_accepted_world_view_to_circuit(circuit, evidence)?);
        }
        Ok(updates)
    }

    fn accepted_world_views_from_gpu_batch_execution_evidence(
        &mut self,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
        construct: &str,
    ) -> Result<Vec<AcceptedWorldViewEvidence>> {
        if evidence.batch.results.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: "probabilistic batch gating requires at least one accepted GPU component"
                    .to_string(),
            });
        }
        if evidence.assumptions_by_component.len() != evidence.batch.results.len() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "assumption group count {} does not match GPU batch component count {}",
                    evidence.assumptions_by_component.len(),
                    evidence.batch.results.len()
                ),
            });
        }

        let batch_trace = evidence.batch.trace;
        if batch_trace.component_count != evidence.batch.results.len()
            || batch_trace.gpu_runtime_component_executions != evidence.batch.results.len()
            || batch_trace.cpu_recomposition_steps != 0
            || batch_trace.cpu_candidate_enumerations != 0
            || batch_trace.cpu_world_view_validations != 0
            || batch_trace.tracked_dtoh_calls != 0
            || batch_trace.tracked_htod_calls != 0
            || batch_trace.per_candidate_host_round_trips != 0
            || !batch_trace.aggregate_kernel_timing.is_recorded()
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "accepted GPU batch evidence requires complete GPU component execution and \
                     zero CPU/host fallback counters plus aggregate CUDA-event timing, got \
                     components={}/{}, recomposition={}, cpu_candidates={}, cpu_world_views={}, \
                     dtoh_calls={}, htod_calls={}, round_trips={}, aggregate_timing_recorded={}",
                    batch_trace.gpu_runtime_component_executions,
                    batch_trace.component_count,
                    batch_trace.cpu_recomposition_steps,
                    batch_trace.cpu_candidate_enumerations,
                    batch_trace.cpu_world_view_validations,
                    batch_trace.tracked_dtoh_calls,
                    batch_trace.tracked_htod_calls,
                    batch_trace.per_candidate_host_round_trips,
                    batch_trace.aggregate_kernel_timing.is_recorded()
                ),
            });
        }

        let mut accepted = Vec::with_capacity(evidence.batch.results.len());
        for (result, assumptions) in evidence
            .batch
            .results
            .iter()
            .zip(evidence.assumptions_by_component.iter())
        {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                (*assumptions).to_vec(),
            )?);
        }

        self.trace.accepted_gpu_batch_evidence_consumed = self
            .trace
            .accepted_gpu_batch_evidence_consumed
            .saturating_add(1);
        self.trace.accepted_gpu_batch_component_evidence_consumed = self
            .trace
            .accepted_gpu_batch_component_evidence_consumed
            .saturating_add(accepted.len() as u64);

        Ok(accepted)
    }

    #[cfg(feature = "host-io")]
    fn record_conditioned_evidence_counts(
        &mut self,
        counts: EpistemicProbConditionedEvidenceCounts,
        path: EpistemicProbConditionedEvidencePath,
    ) {
        self.trace.gpu_conditioned_evidence_facts = self
            .trace
            .gpu_conditioned_evidence_facts
            .saturating_add(counts.total as u64);
        self.trace.gpu_conditioned_nonzero_arity_evidence_facts = self
            .trace
            .gpu_conditioned_nonzero_arity_evidence_facts
            .saturating_add(counts.nonzero_arity as u64);
        self.trace.gpu_conditioned_max_evidence_arity = self
            .trace
            .gpu_conditioned_max_evidence_arity
            .max(counts.max_arity);
        self.trace.gpu_conditioned_negative_evidence_facts = self
            .trace
            .gpu_conditioned_negative_evidence_facts
            .saturating_add(counts.negative as u64);
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                self.trace.gpu_source_conditioned_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_evidence_facts
                    .saturating_add(counts.total as u64);
                self.trace
                    .gpu_source_conditioned_nonzero_arity_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_nonzero_arity_evidence_facts
                    .saturating_add(counts.nonzero_arity as u64);
                self.trace.gpu_source_conditioned_max_evidence_arity = self
                    .trace
                    .gpu_source_conditioned_max_evidence_arity
                    .max(counts.max_arity);
                self.trace.gpu_source_conditioned_negative_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_negative_evidence_facts
                    .saturating_add(counts.negative as u64);
                self.trace.gpu_source_conditioned_know_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_know_evidence_facts
                    .saturating_add(counts.know as u64);
                self.trace.gpu_source_conditioned_possible_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_possible_evidence_facts
                    .saturating_add(counts.possible as u64);
                self.trace.gpu_source_conditioned_not_known_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_not_known_evidence_facts
                    .saturating_add(counts.not_known as u64);
                self.trace
                    .gpu_source_conditioned_not_possible_evidence_facts = self
                    .trace
                    .gpu_source_conditioned_not_possible_evidence_facts
                    .saturating_add(counts.not_possible as u64);
            }
            EpistemicProbConditionedEvidencePath::Program => {
                self.trace.gpu_program_conditioned_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_evidence_facts
                    .saturating_add(counts.total as u64);
                self.trace
                    .gpu_program_conditioned_nonzero_arity_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_nonzero_arity_evidence_facts
                    .saturating_add(counts.nonzero_arity as u64);
                self.trace.gpu_program_conditioned_max_evidence_arity = self
                    .trace
                    .gpu_program_conditioned_max_evidence_arity
                    .max(counts.max_arity);
                self.trace.gpu_program_conditioned_negative_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_negative_evidence_facts
                    .saturating_add(counts.negative as u64);
                self.trace.gpu_program_conditioned_know_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_know_evidence_facts
                    .saturating_add(counts.know as u64);
                self.trace.gpu_program_conditioned_possible_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_possible_evidence_facts
                    .saturating_add(counts.possible as u64);
                self.trace.gpu_program_conditioned_not_known_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_not_known_evidence_facts
                    .saturating_add(counts.not_known as u64);
                self.trace
                    .gpu_program_conditioned_not_possible_evidence_facts = self
                    .trace
                    .gpu_program_conditioned_not_possible_evidence_facts
                    .saturating_add(counts.not_possible as u64);
            }
        }
        self.trace.gpu_conditioned_know_evidence_facts = self
            .trace
            .gpu_conditioned_know_evidence_facts
            .saturating_add(counts.know as u64);
        self.trace.gpu_conditioned_possible_evidence_facts = self
            .trace
            .gpu_conditioned_possible_evidence_facts
            .saturating_add(counts.possible as u64);
        self.trace.gpu_conditioned_not_known_evidence_facts = self
            .trace
            .gpu_conditioned_not_known_evidence_facts
            .saturating_add(counts.not_known as u64);
        self.trace.gpu_conditioned_not_possible_evidence_facts = self
            .trace
            .gpu_conditioned_not_possible_evidence_facts
            .saturating_add(counts.not_possible as u64);
    }

    /// Compile source through the existing GPU-native exact/provenance path.
    pub fn compile_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactDdnnfProgram> {
        self.consume_accepted_evidence(evidence)?;
        let program = ExactDdnnfProgram::compile_source_with_gpu(source, self.config)?;
        self.trace.gpu_exact_source_compiles =
            self.trace.gpu_exact_source_compiles.saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(program)
    }

    /// Compile source through the GPU exact path after accepted GPU epistemic execution.
    pub fn compile_source_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactDdnnfProgram> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_source_with_accepted_world_view(source, &evidence)
    }

    /// Compile source once per accepted GPU epistemic execution result.
    pub fn compile_source_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactDdnnfProgram>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic source exact compile batch".to_string(),
                context: "batched source exact compile requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut programs = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            programs.push(self.compile_source_with_accepted_world_view(source, evidence)?);
        }
        Ok(programs)
    }

    /// Compile source once per accepted split/batch GPU epistemic component.
    pub fn compile_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactDdnnfProgram>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic source exact compile batch production",
        )?;

        let mut programs = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            programs.push(self.compile_source_with_accepted_world_view(source, evidence)?);
        }
        Ok(programs)
    }

    /// Compile a parsed program through the existing GPU-native exact/provenance path.
    pub fn compile_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactDdnnfProgram> {
        self.consume_accepted_evidence(evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(program, self.config)?;
        self.trace.gpu_exact_program_compiles =
            self.trace.gpu_exact_program_compiles.saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(exact)
    }

    /// Compile a parsed program through the GPU exact path after accepted GPU epistemic execution.
    pub fn compile_program_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactDdnnfProgram> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_program_with_accepted_world_view(program, &evidence)
    }

    /// Compile a parsed program once per accepted GPU epistemic execution result.
    pub fn compile_program_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactDdnnfProgram>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic parsed-program exact compile batch".to_string(),
                context:
                    "batched parsed-program exact compile requires at least one accepted GPU result"
                        .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut programs = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            programs.push(self.compile_program_with_accepted_world_view(program, evidence)?);
        }
        Ok(programs)
    }

    /// Compile a parsed program once per accepted split/batch GPU epistemic component.
    pub fn compile_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactDdnnfProgram>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic parsed-program exact compile batch production",
        )?;

        let mut programs = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            programs.push(self.compile_program_with_accepted_world_view(program, evidence)?);
        }
        Ok(programs)
    }

    /// Compile source and evaluate queries through the existing GPU exact path after one accepted gate.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        self.consume_accepted_evidence(evidence)?;
        let program = ExactDdnnfProgram::compile_source_with_gpu(source, self.config)?;
        self.trace.gpu_exact_source_compiles =
            self.trace.gpu_exact_source_compiles.saturating_add(1);
        let result = program.evaluate()?;
        self.trace.gpu_exact_query_evaluations =
            self.trace.gpu_exact_query_evaluations.saturating_add(1);
        self.trace.gpu_source_exact_query_evaluations = self
            .trace
            .gpu_source_exact_query_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_source_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_source_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile source and evaluate queries after accepted GPU epistemic execution.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResult> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_source_with_accepted_world_view(source, &evidence)
    }

    /// Compile and evaluate source once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production batch".to_string(),
                context: "batched knowledge compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results
                .push(self.compile_and_evaluate_source_with_accepted_world_view(source, evidence)?);
        }
        Ok(results)
    }

    /// Compile and evaluate source once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results
                .push(self.compile_and_evaluate_source_with_accepted_world_view(source, evidence)?);
        }
        Ok(results)
    }

    /// Compile source with accepted zero-arity epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        self.consume_accepted_evidence(evidence)?;
        let (program, evidence_counts) = condition_source_with_accepted_evidence(source, evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        self.trace.gpu_exact_source_compiles =
            self.trace.gpu_exact_source_compiles.saturating_add(1);
        self.record_conditioned_evidence_counts(
            evidence_counts,
            EpistemicProbConditionedEvidencePath::Source,
        );
        let result = exact.evaluate()?;
        self.trace.gpu_exact_query_evaluations =
            self.trace.gpu_exact_query_evaluations.saturating_add(1);
        self.trace.gpu_source_exact_query_evaluations = self
            .trace
            .gpu_source_exact_query_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_source_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_source_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile source with accepted GPU epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResult> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_conditioned_source_with_accepted_world_view(source, &evidence)
    }

    /// Compile conditioned source once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned production batch".to_string(),
                context: "batched conditioned knowledge compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_source_with_accepted_world_view(
                    source, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile conditioned source once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_source_with_accepted_world_view(
                    source, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile source with accepted epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        self.consume_accepted_evidence(evidence)?;
        let (program, evidence_counts) = condition_source_with_accepted_evidence(source, evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        self.trace.gpu_exact_source_compiles =
            self.trace.gpu_exact_source_compiles.saturating_add(1);
        self.record_conditioned_evidence_counts(
            evidence_counts,
            EpistemicProbConditionedEvidencePath::Source,
        );
        let result = exact.evaluate_gpu_with_grads()?;
        self.trace.gpu_exact_gradient_evaluations =
            self.trace.gpu_exact_gradient_evaluations.saturating_add(1);
        self.trace.gpu_source_conditioned_gradient_evaluations = self
            .trace
            .gpu_source_conditioned_gradient_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_source_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_source_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile source with accepted GPU epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResultWithGrads> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_conditioned_source_with_grads_with_accepted_world_view(
            source, &evidence,
        )
    }

    /// Compile conditioned source gradients once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResultWithGrads>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned gradient production batch"
                    .to_string(),
                context: "batched conditioned gradient compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_source_with_grads_with_accepted_world_view(
                    source, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile conditioned source gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_source_with_grads_with_accepted_world_view(
                    source, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile a parsed program with accepted epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        self.consume_accepted_evidence(evidence)?;
        let (program, evidence_counts) =
            condition_program_with_accepted_evidence(program, evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        self.trace.gpu_exact_program_compiles =
            self.trace.gpu_exact_program_compiles.saturating_add(1);
        self.record_conditioned_evidence_counts(
            evidence_counts,
            EpistemicProbConditionedEvidencePath::Program,
        );
        let result = exact.evaluate()?;
        self.trace.gpu_exact_query_evaluations =
            self.trace.gpu_exact_query_evaluations.saturating_add(1);
        self.trace.gpu_program_exact_query_evaluations = self
            .trace
            .gpu_program_exact_query_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_program_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_program_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile a parsed program with accepted GPU epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResult> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_conditioned_program_with_accepted_world_view(program, &evidence)
    }

    /// Compile conditioned parsed program once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned production batch".to_string(),
                context: "batched conditioned program compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_program_with_accepted_world_view(
                    program, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile conditioned parsed program once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_program_with_accepted_world_view(
                    program, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile a parsed program with accepted epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        self.consume_accepted_evidence(evidence)?;
        let (program, evidence_counts) =
            condition_program_with_accepted_evidence(program, evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        self.trace.gpu_exact_program_compiles =
            self.trace.gpu_exact_program_compiles.saturating_add(1);
        self.record_conditioned_evidence_counts(
            evidence_counts,
            EpistemicProbConditionedEvidencePath::Program,
        );
        let result = exact.evaluate_gpu_with_grads()?;
        self.trace.gpu_exact_gradient_evaluations =
            self.trace.gpu_exact_gradient_evaluations.saturating_add(1);
        self.trace.gpu_program_conditioned_gradient_evaluations = self
            .trace
            .gpu_program_conditioned_gradient_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_program_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_program_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile a parsed program with accepted GPU epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResultWithGrads> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_conditioned_program_with_grads_with_accepted_world_view(
            program, &evidence,
        )
    }

    /// Compile conditioned parsed-program gradients once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResultWithGrads>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned gradient production batch"
                    .to_string(),
                context: "batched conditioned program gradient compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_program_with_grads_with_accepted_world_view(
                    program, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile conditioned parsed-program gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_conditioned_program_with_grads_with_accepted_world_view(
                    program, evidence,
                )?,
            );
        }
        Ok(results)
    }

    /// Compile a parsed program and evaluate queries through the existing GPU exact path.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        self.consume_accepted_evidence(evidence)?;
        let exact = ExactDdnnfProgram::compile_from_program(program, self.config)?;
        self.trace.gpu_exact_program_compiles =
            self.trace.gpu_exact_program_compiles.saturating_add(1);
        let result = exact.evaluate()?;
        self.trace.gpu_exact_query_evaluations =
            self.trace.gpu_exact_query_evaluations.saturating_add(1);
        self.trace.gpu_program_exact_query_evaluations = self
            .trace
            .gpu_program_exact_query_evaluations
            .saturating_add(1);
        self.trace.gpu_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.gpu_program_knowledge_compilation_end_to_end_runs = self
            .trace
            .gpu_program_knowledge_compilation_end_to_end_runs
            .saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile a parsed program and evaluate queries after accepted GPU epistemic execution.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResult> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.compile_and_evaluate_program_with_accepted_world_view(program, &evidence)
    }

    /// Compile and evaluate a parsed program once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic parsed-program production batch".to_string(),
                context: "batched parsed-program knowledge compilation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_program_with_accepted_world_view(program, evidence)?,
            );
        }
        Ok(results)
    }

    /// Compile and evaluate a parsed program once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.compile_and_evaluate_program_with_accepted_world_view(program, evidence)?,
            );
        }
        Ok(results)
    }

    /// Encode source through the existing GPU PIR and CNF production path.
    pub fn encode_source_pir_cnf_with_accepted_world_view(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        let provenance = extract_from_source(source)?;
        self.encode_provenance_pir_cnf_with_accepted_world_view(
            provenance,
            provider,
            evidence,
            EpistemicProbPirCnfPath::Source,
        )
    }

    /// Encode source PIR/CNF after accepted GPU epistemic execution.
    pub fn encode_source_pir_cnf_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.encode_source_pir_cnf_with_accepted_world_view(source, provider, &evidence)
    }

    /// Encode source PIR/CNF once per accepted GPU epistemic execution result.
    pub fn encode_source_pir_cnf_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic source PIR/CNF production batch".to_string(),
                context:
                    "batched source PIR/CNF encoding requires at least one accepted GPU result"
                        .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.encode_source_pir_cnf_with_accepted_world_view(source, provider, evidence)?,
            );
        }
        Ok(results)
    }

    /// Encode source PIR/CNF once per accepted split/batch GPU epistemic component.
    pub fn encode_source_pir_cnf_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider.as_ref(),
            evidence,
            "epistemic probabilistic source PIR/CNF batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.encode_source_pir_cnf_with_accepted_world_view(source, provider, evidence)?,
            );
        }
        Ok(results)
    }

    /// Encode a parsed program through the existing GPU PIR and CNF production path.
    pub fn encode_program_pir_cnf_with_accepted_world_view(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        let provenance = extract_from_program(program)?;
        self.encode_provenance_pir_cnf_with_accepted_world_view(
            provenance,
            provider,
            evidence,
            EpistemicProbPirCnfPath::Program,
        )
    }

    /// Encode parsed-program PIR/CNF after accepted GPU epistemic execution.
    pub fn encode_program_pir_cnf_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.encode_program_pir_cnf_with_accepted_world_view(program, provider, &evidence)
    }

    /// Encode parsed-program PIR/CNF once per accepted GPU epistemic execution result.
    pub fn encode_program_pir_cnf_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic parsed-program PIR/CNF production batch"
                    .to_string(),
                context:
                    "batched parsed-program PIR/CNF encoding requires at least one accepted GPU result"
                        .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.encode_program_pir_cnf_with_accepted_world_view(program, provider, evidence)?,
            );
        }
        Ok(results)
    }

    /// Encode parsed-program PIR/CNF once per accepted split/batch GPU epistemic component.
    pub fn encode_program_pir_cnf_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider.as_ref(),
            evidence,
            "epistemic probabilistic parsed-program PIR/CNF batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(
                self.encode_program_pir_cnf_with_accepted_world_view(program, provider, evidence)?,
            );
        }
        Ok(results)
    }

    /// Evaluate GPU exact query probabilities after accepted world-view evidence was consumed.
    #[cfg(feature = "host-io")]
    pub fn evaluate(
        &mut self,
        program: &ExactDdnnfProgram,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        self.consume_accepted_evidence(evidence)?;
        let result = program.evaluate()?;
        self.trace.gpu_exact_query_evaluations =
            self.trace.gpu_exact_query_evaluations.saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Evaluate GPU exact query probabilities after accepted GPU epistemic execution.
    #[cfg(feature = "host-io")]
    pub fn evaluate_with_gpu_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResult> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.evaluate(program, &evidence)
    }

    /// Evaluate GPU exact query probabilities once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn evaluate_for_gpu_execution_results(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic query evaluation production batch".to_string(),
                context: "batched query evaluation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(self.evaluate(program, evidence)?);
        }
        Ok(results)
    }

    /// Evaluate GPU exact query probabilities once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn evaluate_for_gpu_batch_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic query evaluation batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(self.evaluate(program, evidence)?);
        }
        Ok(results)
    }

    /// Evaluate GPU exact gradients after accepted world-view evidence was consumed.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads(
        &mut self,
        program: &ExactDdnnfProgram,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        self.consume_accepted_evidence(evidence)?;
        let result = program.evaluate_gpu_with_grads()?;
        self.trace.gpu_exact_gradient_evaluations =
            self.trace.gpu_exact_gradient_evaluations.saturating_add(1);
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Evaluate GPU exact gradients after accepted GPU epistemic execution.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads_with_gpu_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<ExactResultWithGrads> {
        let evidence =
            AcceptedWorldViewEvidence::from_gpu_execution_result(provider, result, assumptions)?;
        self.evaluate_gpu_with_grads(program, &evidence)
    }

    /// Evaluate GPU exact gradients once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads_for_gpu_execution_results(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResultWithGrads>> {
        if evidence_records.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic gradient evaluation production batch"
                    .to_string(),
                context: "batched gradient evaluation requires at least one accepted GPU result"
                    .to_string(),
            });
        }

        let mut accepted = Vec::with_capacity(evidence_records.len());
        for record in evidence_records {
            accepted.push(AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                record.result,
                record.assumptions.to_vec(),
            )?);
        }

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(self.evaluate_gpu_with_grads(program, evidence)?);
        }
        Ok(results)
    }

    /// Evaluate GPU exact gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads_for_gpu_batch_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
            provider,
            evidence,
            "epistemic probabilistic gradient evaluation batch production",
        )?;

        let mut results = Vec::with_capacity(accepted.len());
        for evidence in &accepted {
            results.push(self.evaluate_gpu_with_grads(program, evidence)?);
        }
        Ok(results)
    }

    fn consume_accepted_evidence(&mut self, evidence: &AcceptedWorldViewEvidence) -> Result<()> {
        if evidence.world_count() == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted world-view evidence".to_string(),
                context: "probabilistic production path requires a non-empty accepted world view"
                    .to_string(),
            });
        }
        self.trace.accepted_world_view_evidence_consumed = self
            .trace
            .accepted_world_view_evidence_consumed
            .saturating_add(1);
        match evidence.gpu_epistemic_mode() {
            Some(EirEpistemicMode::G91) => {
                self.trace.accepted_g91_world_view_evidence_consumed = self
                    .trace
                    .accepted_g91_world_view_evidence_consumed
                    .saturating_add(1);
            }
            Some(EirEpistemicMode::Faeel) => {
                self.trace.accepted_faeel_world_view_evidence_consumed = self
                    .trace
                    .accepted_faeel_world_view_evidence_consumed
                    .saturating_add(1);
            }
            None => {}
        }
        self.trace.accepted_evidence_assumptions_consumed = self
            .trace
            .accepted_evidence_assumptions_consumed
            .saturating_add(evidence.assumption_count() as u64);
        self.trace.require_zero_cpu_recompute()
    }

    fn encode_provenance_pir_cnf_with_accepted_world_view(
        &mut self,
        provenance: Provenance,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
        path: EpistemicProbPirCnfPath,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        self.consume_accepted_evidence(evidence)?;
        let roots = production_pir_roots(&provenance)?;
        if roots.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted probabilistic PIR/CNF production path".to_string(),
                context: "GPU PIR/CNF evidence requires at least one query, evidence, or probabilistic variable root".to_string(),
            });
        }
        let gpu_pir = GpuPirGraph::from_host(&provenance.pir, provider)?;
        self.trace.gpu_pir_graph_uploads = self.trace.gpu_pir_graph_uploads.saturating_add(1);
        match path {
            EpistemicProbPirCnfPath::Source => {
                self.trace.gpu_source_pir_graph_uploads =
                    self.trace.gpu_source_pir_graph_uploads.saturating_add(1);
            }
            EpistemicProbPirCnfPath::Program => {
                self.trace.gpu_program_pir_graph_uploads =
                    self.trace.gpu_program_pir_graph_uploads.saturating_add(1);
            }
        }
        let gpu_roots = GpuPirRoots::from_host(&roots, provider)?;
        let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, provider)?;
        self.trace.gpu_cnf_encodes = self.trace.gpu_cnf_encodes.saturating_add(1);
        match path {
            EpistemicProbPirCnfPath::Source => {
                self.trace.gpu_source_cnf_encodes =
                    self.trace.gpu_source_cnf_encodes.saturating_add(1);
            }
            EpistemicProbPirCnfPath::Program => {
                self.trace.gpu_program_cnf_encodes =
                    self.trace.gpu_program_cnf_encodes.saturating_add(1);
            }
        }
        self.trace.require_zero_cpu_recompute()?;
        Ok(EpistemicProbPirCnfEvidence {
            pir_nodes: provenance.pir.len(),
            root_count: roots.len(),
            cnf_var_cap: encoding.cnf.var_cap,
            cnf_clause_cap: encoding.cnf.clause_cap,
            cnf_lit_cap: encoding.cnf.lit_cap,
        })
    }
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EpistemicProbConditionedEvidenceCounts {
    total: usize,
    nonzero_arity: usize,
    max_arity: u32,
    negative: usize,
    know: usize,
    possible: usize,
    not_known: usize,
    not_possible: usize,
}

#[cfg(feature = "host-io")]
fn condition_source_with_accepted_evidence(
    source: &str,
    evidence: &AcceptedWorldViewEvidence,
) -> Result<(Program, EpistemicProbConditionedEvidenceCounts)> {
    let program = parse_program(source)?;
    condition_program_with_accepted_evidence(&program, evidence)
}

#[cfg(feature = "host-io")]
fn condition_program_with_accepted_evidence(
    program: &Program,
    evidence: &AcceptedWorldViewEvidence,
) -> Result<(Program, EpistemicProbConditionedEvidenceCounts)> {
    if evidence.assumptions().is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted probabilistic evidence conditioning".to_string(),
            context: "conditioned exact path requires at least one accepted epistemic assumption"
                .to_string(),
        });
    }

    let mut program = program.clone();
    let mut counts = EpistemicProbConditionedEvidenceCounts::default();
    for assumption in evidence.assumptions() {
        if assumption.arity == 0 {
            if !assumption.terms.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "accepted probabilistic evidence conditioning".to_string(),
                    context: format!(
                        "zero-arity exact evidence must not carry tuple terms, got {}/{}",
                        assumption.predicate, assumption.arity
                    ),
                });
            }
        } else if assumption.terms.len() != assumption.arity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted probabilistic evidence conditioning".to_string(),
                context: format!(
                    "nonzero exact evidence conditioning requires {} concrete tuple terms, got {} for {}/{}",
                    assumption.arity,
                    assumption.terms.len(),
                    assumption.predicate,
                    assumption.arity
                ),
            });
        }
        counts.total += 1;
        if assumption.arity > 0 {
            counts.nonzero_arity += 1;
            counts.max_arity = counts.max_arity.max(assumption.arity as u32);
        }
        if !assumption.value {
            counts.negative += 1;
        }
        match (assumption.kind, assumption.value) {
            (EpistemicAssumptionKind::Know, true) => counts.know += 1,
            (EpistemicAssumptionKind::Possible, true) => counts.possible += 1,
            (EpistemicAssumptionKind::Know, false) => counts.not_known += 1,
            (EpistemicAssumptionKind::Possible, false) => counts.not_possible += 1,
        }
        program.evidence.push(Evidence {
            atom: Atom {
                predicate: assumption.predicate.clone(),
                terms: assumption
                    .terms
                    .iter()
                    .map(evidence_term_to_ast_term)
                    .collect(),
            },
            value: assumption.value,
        });
    }

    Ok((program, counts))
}

#[cfg(feature = "host-io")]
fn evidence_term_to_ast_term(term: &EpistemicEvidenceTerm) -> Term {
    match term {
        EpistemicEvidenceTerm::Integer(value) => Term::Integer(*value),
        EpistemicEvidenceTerm::String(value) => Term::String(value.clone()),
        EpistemicEvidenceTerm::Symbol(value) => Term::Symbol(*value),
    }
}

fn production_pir_roots(provenance: &Provenance) -> Result<Vec<PirNodeId>> {
    let mut roots = BTreeSet::new();

    for (atom, value) in &provenance.evidence {
        if let Some(id) = provenance.query_formula(&atom.predicate, &atom.args) {
            roots.insert(id);
        } else if *value {
            return Err(XlogError::Execution(format!(
                "Exact inference error: evidence atom is never derivable: {}",
                atom.predicate
            )));
        }
    }

    for atom in &provenance.queries {
        if let Some(id) = provenance.query_formula(&atom.predicate, &atom.args) {
            roots.insert(id);
        }
    }

    for (idx, node) in provenance.pir.nodes().iter().enumerate() {
        if matches!(
            node,
            PirNode::Decision { .. } | PirNode::Lit { .. } | PirNode::NegLit { .. }
        ) {
            roots.insert(PirNodeId::from_u32(idx as u32));
        }
    }

    Ok(roots.into_iter().collect())
}
