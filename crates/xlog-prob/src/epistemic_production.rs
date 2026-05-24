//! Production GPU exact-path adapter for accepted epistemic evidence.
//!
//! This module is intentionally thin. It gates probabilistic execution on
//! accepted world-view evidence, then routes into the existing GPU-native exact
//! provenance path instead of using the bounded epistemic fixture circuit.

use std::collections::BTreeSet;
use std::sync::Arc;

use xlog_core::{symbol, Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_ir::EirEpistemicMode;
use xlog_logic::ast::{Atom, Evidence, Program, Term};
use xlog_logic::parse_program;
use xlog_runtime::{EpistemicGpuBatchExecutionResult, EpistemicGpuExecutionResult};

use crate::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
#[cfg(feature = "host-io")]
use crate::epistemic::EpistemicAssumptionKind;
use crate::epistemic::EpistemicEvidenceTerm;
use crate::epistemic::{
    AcceptedWorldViewEvidence, CircuitUpdate, CircuitUpdateMode, EpistemicAssumption,
    EpistemicCircuit,
};
#[cfg(feature = "host-io")]
use crate::exact::ExactProgramOrigin;
use crate::exact::{ExactDdnnfProgram, GpuConfig};
#[cfg(feature = "host-io")]
use crate::exact::{ExactResult, ExactResultWithGrads};
use crate::pir::{PirNode, PirNodeId};
use crate::provenance::Value;
use crate::provenance::{extract_from_program, Provenance};

macro_rules! epistemic_prob_trace_transaction {
    ($adapter:ident, $body:block) => {{
        let trace_before = $adapter.trace;
        let result: Result<_> = (|| $body)();
        match result {
            Ok(value) => Ok(value),
            Err(err) => {
                $adapter.trace = trace_before;
                Err(err)
            }
        }
    }};
}

macro_rules! checked_prob_trace_counter_inc {
    ($adapter:ident, $field:ident) => {{
        $adapter.trace.$field = EpistemicProbProductionAdapter::checked_trace_counter_add(
            $adapter.trace.$field,
            1,
            stringify!($field),
        )?;
    }};
}

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
    /// Number of accepted nonzero-arity epistemic assumptions consumed from GPU evidence.
    pub accepted_gpu_nonzero_arity_evidence_assumptions_consumed: u64,
    /// Maximum accepted GPU evidence tuple arity consumed by this adapter.
    pub accepted_gpu_max_evidence_arity_consumed: u32,
    /// GPU tuple-key column reads consumed from accepted world-view evidence.
    pub accepted_gpu_tuple_key_column_reads_consumed: u64,
    /// GPU final-tuple row filters consumed from accepted world-view evidence.
    pub accepted_gpu_final_tuple_row_filters_consumed: u64,
    /// Negated GPU final-tuple row filters consumed from accepted world-view evidence.
    pub accepted_gpu_final_tuple_negated_row_filters_consumed: u64,
    /// Row-specific GPU model-slot capacity consumed from accepted world-view evidence.
    pub accepted_gpu_row_specific_membership_row_capacity_consumed: u64,
    /// Fallback GPU row-filter capacity consumed outside bounded model-slot windows.
    pub accepted_gpu_row_filter_fallback_row_capacity_consumed: u64,
    /// Reduced integrity-constraint relations checked by accepted GPU evidence.
    pub accepted_gpu_constraint_relations_checked_consumed: u64,
    /// Constraint row-count metadata reads consumed from accepted GPU evidence.
    pub accepted_gpu_constraint_row_count_device_reads_consumed: u64,
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
    /// Number of source GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_source_exact_gradient_evaluations: u64,
    /// Number of parsed-program GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_program_exact_gradient_evaluations: u64,
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
    /// GPU exact/provenance/PIR/CNF/knowledge-compilation events that occurred inside accepted evidence gates.
    pub accepted_gpu_production_path_events: u64,
    /// Number of accepted source compile-and-evaluate runs through the GPU exact path.
    pub gpu_source_knowledge_compilation_end_to_end_runs: u64,
    /// Number of accepted parsed-program compile-and-evaluate runs through the GPU exact path.
    pub gpu_program_knowledge_compilation_end_to_end_runs: u64,
    /// Number of accepted assumptions compiled as exact evidence facts.
    pub gpu_conditioned_evidence_facts: u64,
    /// Number of accepted world-view evidence objects compiled into conditioned exact evidence.
    pub accepted_conditioned_world_view_evidence_consumed: u64,
    /// Number of source-conditioned accepted world-view evidence objects compiled as exact evidence.
    pub accepted_source_conditioned_world_view_evidence_consumed: u64,
    /// Number of parsed-program-conditioned accepted world-view evidence objects compiled as exact evidence.
    pub accepted_program_conditioned_world_view_evidence_consumed: u64,
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
    fn checked_gpu_production_path_events(&self) -> Result<u64> {
        Self::checked_production_event_sum(
            "gpu_production_path_events",
            &[
                self.gpu_exact_source_compiles,
                self.gpu_exact_program_compiles,
                self.gpu_exact_query_evaluations,
                self.gpu_source_exact_query_evaluations,
                self.gpu_program_exact_query_evaluations,
                self.gpu_exact_gradient_evaluations,
                self.gpu_source_exact_gradient_evaluations,
                self.gpu_program_exact_gradient_evaluations,
                self.gpu_source_conditioned_gradient_evaluations,
                self.gpu_program_conditioned_gradient_evaluations,
                self.gpu_pir_graph_uploads,
                self.gpu_source_pir_graph_uploads,
                self.gpu_program_pir_graph_uploads,
                self.gpu_cnf_encodes,
                self.gpu_source_cnf_encodes,
                self.gpu_program_cnf_encodes,
                self.gpu_knowledge_compilation_end_to_end_runs,
                self.gpu_source_knowledge_compilation_end_to_end_runs,
                self.gpu_program_knowledge_compilation_end_to_end_runs,
            ],
        )
    }

    fn checked_production_event_sum(counter: &str, values: &[u64]) -> Result<u64> {
        values.iter().try_fold(0u64, |acc, value| {
            acc.checked_add(*value)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic production trace accounting".to_string(),
                    context: format!(
                        "GPU probability production counter {counter} overflowed while adding \
                         {value} to {acc}"
                    ),
                })
        })
    }

    fn require_pir_cnf_accounting_pair(
        construct: &'static str,
        pir_graph_uploads: u64,
        cnf_encodes: u64,
        path: &'static str,
    ) -> Result<()> {
        if pir_graph_uploads != cnf_encodes {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "PIR/CNF production accounting must match for {path} path, got \
                     pir_graph_uploads={} cnf_encodes={}",
                    pir_graph_uploads, cnf_encodes
                ),
            });
        }
        Ok(())
    }

    fn require_pir_cnf_accounting(&self) -> Result<()> {
        let construct = "epistemic probabilistic production metric gate";
        Self::require_pir_cnf_accounting_pair(
            construct,
            self.gpu_pir_graph_uploads,
            self.gpu_cnf_encodes,
            "aggregate",
        )?;
        Self::require_pir_cnf_accounting_pair(
            construct,
            self.gpu_source_pir_graph_uploads,
            self.gpu_source_cnf_encodes,
            "source",
        )?;
        Self::require_pir_cnf_accounting_pair(
            construct,
            self.gpu_program_pir_graph_uploads,
            self.gpu_program_cnf_encodes,
            "program",
        )
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

    /// Require internally consistent GPU tuple-membership evidence counters.
    pub fn require_accepted_gpu_tuple_evidence_trace(&self) -> Result<()> {
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
            > self.accepted_evidence_assumptions_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity evidence assumptions cannot exceed accepted \
                     evidence assumptions: nonzero={} total={}",
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
                    self.accepted_evidence_assumptions_consumed
                ),
            });
        }
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed == 0
            && self.accepted_gpu_max_evidence_arity_consumed > 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted max evidence arity {} requires at least one accepted \
                     nonzero-arity GPU evidence assumption",
                    self.accepted_gpu_max_evidence_arity_consumed
                ),
            });
        }
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed > 0
            && self.accepted_gpu_max_evidence_arity_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity GPU evidence requires accepted max evidence arity, \
                     got nonzero_assumptions={} max_arity=0",
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
                ),
            });
        }
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed == 0
            && self.accepted_gpu_tuple_key_column_reads_consumed != 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted tuple-key reads require accepted nonzero-arity GPU evidence, got \
                     nonzero_assumptions=0 tuple_key_reads={}",
                    self.accepted_gpu_tuple_key_column_reads_consumed
                ),
            });
        }
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed > 0
            && self.accepted_gpu_tuple_key_column_reads_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity GPU evidence requires tuple-key device column reads, \
                     got nonzero_assumptions={} tuple_key_reads=0",
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_negated_row_filters_consumed
            > self.accepted_gpu_final_tuple_row_filters_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted negated final-tuple row filters cannot exceed total row filters: \
                     negated={} total={}",
                    self.accepted_gpu_final_tuple_negated_row_filters_consumed,
                    self.accepted_gpu_final_tuple_row_filters_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_row_filters_consumed == 0
            && (self.accepted_gpu_row_specific_membership_row_capacity_consumed != 0
                || self.accepted_gpu_row_filter_fallback_row_capacity_consumed != 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted row-specific/fallback tuple capacity requires accepted GPU row \
                     filters, got row_filters=0 row_specific_capacity={} fallback_capacity={}",
                    self.accepted_gpu_row_specific_membership_row_capacity_consumed,
                    self.accepted_gpu_row_filter_fallback_row_capacity_consumed
                ),
            });
        }
        if self.accepted_gpu_final_tuple_row_filters_consumed > 0
            && self.accepted_gpu_row_specific_membership_row_capacity_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU final-tuple row filters require row-specific model-slot \
                     capacity, got row_filters={} row_specific_capacity=0",
                    self.accepted_gpu_final_tuple_row_filters_consumed
                ),
            });
        }
        if self.accepted_gpu_constraint_row_count_device_reads_consumed
            > self.accepted_gpu_constraint_relations_checked_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted constraint row-count device reads cannot exceed checked reduced \
                     constraint relations, got reads={} checked={}",
                    self.accepted_gpu_constraint_row_count_device_reads_consumed,
                    self.accepted_gpu_constraint_relations_checked_consumed
                ),
            });
        }
        Ok(())
    }

    /// Require internally consistent accepted GPU world-view evidence counters.
    pub fn require_accepted_gpu_world_view_evidence_trace(&self) -> Result<()> {
        let mode_count = self
            .accepted_g91_world_view_evidence_consumed
            .checked_add(self.accepted_faeel_world_view_evidence_consumed)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "accepted GPU world-view mode counters overflowed".to_string(),
            })?;
        if self.accepted_world_view_evidence_consumed != 0
            && mode_count != self.accepted_world_view_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU world-view evidence must be classified by epistemic mode, got \
                     evidence={} g91={} faeel={}",
                    self.accepted_world_view_evidence_consumed,
                    self.accepted_g91_world_view_evidence_consumed,
                    self.accepted_faeel_world_view_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
            > self.accepted_evidence_assumptions_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted nonzero-arity GPU evidence assumptions cannot exceed accepted \
                     assumptions, got nonzero={} assumptions={}",
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
                    self.accepted_evidence_assumptions_consumed
                ),
            });
        }
        if self.accepted_world_view_evidence_consumed != 0
            && self.accepted_evidence_assumptions_consumed
                < self.accepted_world_view_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU world-view evidence requires at least one accepted epistemic \
                     assumption per evidence record, got evidence={} assumptions={}",
                    self.accepted_world_view_evidence_consumed,
                    self.accepted_evidence_assumptions_consumed
                ),
            });
        }
        if self.accepted_gpu_batch_component_evidence_consumed
            < self.accepted_gpu_batch_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU batch component evidence must cover accepted batch evidence, \
                     got batches={} components={}",
                    self.accepted_gpu_batch_evidence_consumed,
                    self.accepted_gpu_batch_component_evidence_consumed
                ),
            });
        }
        if self.accepted_gpu_batch_component_evidence_consumed
            > self.accepted_world_view_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU batch component evidence cannot exceed accepted world-view \
                     evidence, got components={} evidence={}",
                    self.accepted_gpu_batch_component_evidence_consumed,
                    self.accepted_world_view_evidence_consumed
                ),
            });
        }
        if self.accepted_conditioned_world_view_evidence_consumed
            > self.accepted_world_view_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted conditioned world-view evidence cannot exceed accepted world-view \
                     evidence, got conditioned={} evidence={}",
                    self.accepted_conditioned_world_view_evidence_consumed,
                    self.accepted_world_view_evidence_consumed
                ),
            });
        }
        Ok(())
    }

    fn require_conditioned_counter_sum(
        counter: &'static str,
        aggregate: u64,
        source: u64,
        program: u64,
    ) -> Result<()> {
        let expected = source.checked_add(program).ok_or_else(|| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned evidence counter {counter} overflowed while adding source={} \
                     program={}",
                    source, program
                ),
            }
        })?;
        if aggregate != expected {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned evidence counter {counter} must equal source+program, got \
                     aggregate={} source={} program={}",
                    aggregate, source, program
                ),
            });
        }
        Ok(())
    }

    fn require_gpu_path_counter_sum(
        counter: &'static str,
        aggregate: u64,
        source: u64,
        program: u64,
    ) -> Result<()> {
        let expected = source.checked_add(program).ok_or_else(|| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "GPU production path counter {counter} overflowed while adding source={} \
                     program={}",
                    source, program
                ),
            }
        })?;
        if aggregate != expected {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "GPU production path accounting must match source+program for {counter}, \
                     got aggregate={} source={} program={}",
                    aggregate, source, program
                ),
            });
        }
        Ok(())
    }

    fn require_gpu_path_accounting(&self) -> Result<()> {
        Self::require_gpu_path_counter_sum(
            "exact_query_evaluations",
            self.gpu_exact_query_evaluations,
            self.gpu_source_exact_query_evaluations,
            self.gpu_program_exact_query_evaluations,
        )?;
        Self::require_gpu_path_counter_sum(
            "exact_gradient_evaluations",
            self.gpu_exact_gradient_evaluations,
            self.gpu_source_exact_gradient_evaluations,
            self.gpu_program_exact_gradient_evaluations,
        )?;
        Self::require_gpu_path_counter_sum(
            "pir_graph_uploads",
            self.gpu_pir_graph_uploads,
            self.gpu_source_pir_graph_uploads,
            self.gpu_program_pir_graph_uploads,
        )?;
        Self::require_gpu_path_counter_sum(
            "cnf_encodes",
            self.gpu_cnf_encodes,
            self.gpu_source_cnf_encodes,
            self.gpu_program_cnf_encodes,
        )?;
        Self::require_gpu_path_counter_sum(
            "knowledge_compilation_end_to_end_runs",
            self.gpu_knowledge_compilation_end_to_end_runs,
            self.gpu_source_knowledge_compilation_end_to_end_runs,
            self.gpu_program_knowledge_compilation_end_to_end_runs,
        )
    }

    /// Require internally consistent conditioned exact-evidence counters.
    pub fn require_conditioned_evidence_trace(&self) -> Result<()> {
        Self::require_conditioned_counter_sum(
            "accepted_world_view_evidence",
            self.accepted_conditioned_world_view_evidence_consumed,
            self.accepted_source_conditioned_world_view_evidence_consumed,
            self.accepted_program_conditioned_world_view_evidence_consumed,
        )?;
        Self::require_conditioned_counter_sum(
            "evidence_facts",
            self.gpu_conditioned_evidence_facts,
            self.gpu_source_conditioned_evidence_facts,
            self.gpu_program_conditioned_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "nonzero_arity_evidence_facts",
            self.gpu_conditioned_nonzero_arity_evidence_facts,
            self.gpu_source_conditioned_nonzero_arity_evidence_facts,
            self.gpu_program_conditioned_nonzero_arity_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "negative_evidence_facts",
            self.gpu_conditioned_negative_evidence_facts,
            self.gpu_source_conditioned_negative_evidence_facts,
            self.gpu_program_conditioned_negative_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "know_evidence_facts",
            self.gpu_conditioned_know_evidence_facts,
            self.gpu_source_conditioned_know_evidence_facts,
            self.gpu_program_conditioned_know_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "possible_evidence_facts",
            self.gpu_conditioned_possible_evidence_facts,
            self.gpu_source_conditioned_possible_evidence_facts,
            self.gpu_program_conditioned_possible_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "not_known_evidence_facts",
            self.gpu_conditioned_not_known_evidence_facts,
            self.gpu_source_conditioned_not_known_evidence_facts,
            self.gpu_program_conditioned_not_known_evidence_facts,
        )?;
        Self::require_conditioned_counter_sum(
            "not_possible_evidence_facts",
            self.gpu_conditioned_not_possible_evidence_facts,
            self.gpu_source_conditioned_not_possible_evidence_facts,
            self.gpu_program_conditioned_not_possible_evidence_facts,
        )?;

        if (self.gpu_conditioned_evidence_facts != 0
            && self.accepted_conditioned_world_view_evidence_consumed == 0)
            || (self.gpu_source_conditioned_evidence_facts != 0
                && self.accepted_source_conditioned_world_view_evidence_consumed == 0)
            || (self.gpu_program_conditioned_evidence_facts != 0
                && self.accepted_program_conditioned_world_view_evidence_consumed == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned exact evidence facts require accepted conditioned world-view \
                     evidence, got facts={} evidence={} source_facts={} source_evidence={} \
                     program_facts={} program_evidence={}",
                    self.gpu_conditioned_evidence_facts,
                    self.accepted_conditioned_world_view_evidence_consumed,
                    self.gpu_source_conditioned_evidence_facts,
                    self.accepted_source_conditioned_world_view_evidence_consumed,
                    self.gpu_program_conditioned_evidence_facts,
                    self.accepted_program_conditioned_world_view_evidence_consumed
                ),
            });
        }

        if self.gpu_conditioned_evidence_facts
            < self.accepted_conditioned_world_view_evidence_consumed
            || self.gpu_source_conditioned_evidence_facts
                < self.accepted_source_conditioned_world_view_evidence_consumed
            || self.gpu_program_conditioned_evidence_facts
                < self.accepted_program_conditioned_world_view_evidence_consumed
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned exact evidence facts must cover each accepted conditioned \
                     world-view evidence record, got facts={} evidence={} source_facts={} \
                     source_evidence={} program_facts={} program_evidence={}",
                    self.gpu_conditioned_evidence_facts,
                    self.accepted_conditioned_world_view_evidence_consumed,
                    self.gpu_source_conditioned_evidence_facts,
                    self.accepted_source_conditioned_world_view_evidence_consumed,
                    self.gpu_program_conditioned_evidence_facts,
                    self.accepted_program_conditioned_world_view_evidence_consumed
                ),
            });
        }

        if self.gpu_conditioned_nonzero_arity_evidence_facts > self.gpu_conditioned_evidence_facts
            || self.gpu_source_conditioned_nonzero_arity_evidence_facts
                > self.gpu_source_conditioned_evidence_facts
            || self.gpu_program_conditioned_nonzero_arity_evidence_facts
                > self.gpu_program_conditioned_evidence_facts
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned nonzero-arity facts cannot exceed conditioned evidence facts: \
                     nonzero={} total={} source_nonzero={} source_total={} program_nonzero={} \
                     program_total={}",
                    self.gpu_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_conditioned_evidence_facts,
                    self.gpu_source_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_source_conditioned_evidence_facts,
                    self.gpu_program_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_program_conditioned_evidence_facts
                ),
            });
        }
        if self.gpu_conditioned_negative_evidence_facts > self.gpu_conditioned_evidence_facts
            || self.gpu_source_conditioned_negative_evidence_facts
                > self.gpu_source_conditioned_evidence_facts
            || self.gpu_program_conditioned_negative_evidence_facts
                > self.gpu_program_conditioned_evidence_facts
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned negative facts cannot exceed conditioned evidence facts: \
                     negative={} total={} source_negative={} source_total={} program_negative={} \
                     program_total={}",
                    self.gpu_conditioned_negative_evidence_facts,
                    self.gpu_conditioned_evidence_facts,
                    self.gpu_source_conditioned_negative_evidence_facts,
                    self.gpu_source_conditioned_evidence_facts,
                    self.gpu_program_conditioned_negative_evidence_facts,
                    self.gpu_program_conditioned_evidence_facts
                ),
            });
        }

        let operator_fact_count = self
            .gpu_conditioned_know_evidence_facts
            .checked_add(self.gpu_conditioned_possible_evidence_facts)
            .and_then(|sum| sum.checked_add(self.gpu_conditioned_not_known_evidence_facts))
            .and_then(|sum| sum.checked_add(self.gpu_conditioned_not_possible_evidence_facts))
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: "conditioned operator evidence fact counters overflowed".to_string(),
            })?;
        if operator_fact_count != self.gpu_conditioned_evidence_facts {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned operator evidence facts must equal total evidence facts, got \
                     operators={} total={}",
                    operator_fact_count, self.gpu_conditioned_evidence_facts
                ),
            });
        }

        let source_operator_fact_count = self
            .gpu_source_conditioned_know_evidence_facts
            .checked_add(self.gpu_source_conditioned_possible_evidence_facts)
            .and_then(|sum| sum.checked_add(self.gpu_source_conditioned_not_known_evidence_facts))
            .and_then(|sum| {
                sum.checked_add(self.gpu_source_conditioned_not_possible_evidence_facts)
            })
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: "source conditioned operator evidence fact counters overflowed"
                    .to_string(),
            })?;
        if source_operator_fact_count != self.gpu_source_conditioned_evidence_facts {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "source conditioned operator evidence facts must equal source evidence \
                     facts, got operators={} total={}",
                    source_operator_fact_count, self.gpu_source_conditioned_evidence_facts
                ),
            });
        }

        let program_operator_fact_count = self
            .gpu_program_conditioned_know_evidence_facts
            .checked_add(self.gpu_program_conditioned_possible_evidence_facts)
            .and_then(|sum| sum.checked_add(self.gpu_program_conditioned_not_known_evidence_facts))
            .and_then(|sum| {
                sum.checked_add(self.gpu_program_conditioned_not_possible_evidence_facts)
            })
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: "program conditioned operator evidence fact counters overflowed"
                    .to_string(),
            })?;
        if program_operator_fact_count != self.gpu_program_conditioned_evidence_facts {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "program conditioned operator evidence facts must equal program evidence \
                     facts, got operators={} total={}",
                    program_operator_fact_count, self.gpu_program_conditioned_evidence_facts
                ),
            });
        }

        let expected_max_arity = self
            .gpu_source_conditioned_max_evidence_arity
            .max(self.gpu_program_conditioned_max_evidence_arity);
        if self.gpu_conditioned_max_evidence_arity != expected_max_arity {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned max evidence arity must equal max(source, program), got \
                     aggregate={} source={} program={}",
                    self.gpu_conditioned_max_evidence_arity,
                    self.gpu_source_conditioned_max_evidence_arity,
                    self.gpu_program_conditioned_max_evidence_arity
                ),
            });
        }
        if (self.gpu_conditioned_nonzero_arity_evidence_facts == 0)
            != (self.gpu_conditioned_max_evidence_arity == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned max evidence arity must be nonzero exactly when nonzero-arity \
                     facts are present, got nonzero={} max_arity={}",
                    self.gpu_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_conditioned_max_evidence_arity
                ),
            });
        }
        if (self.gpu_source_conditioned_nonzero_arity_evidence_facts == 0)
            != (self.gpu_source_conditioned_max_evidence_arity == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "source conditioned max evidence arity must be nonzero exactly when \
                     source nonzero-arity facts are present, got nonzero={} max_arity={}",
                    self.gpu_source_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_source_conditioned_max_evidence_arity
                ),
            });
        }
        if (self.gpu_program_conditioned_nonzero_arity_evidence_facts == 0)
            != (self.gpu_program_conditioned_max_evidence_arity == 0)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "program conditioned max evidence arity must be nonzero exactly when \
                     program nonzero-arity facts are present, got nonzero={} max_arity={}",
                    self.gpu_program_conditioned_nonzero_arity_evidence_facts,
                    self.gpu_program_conditioned_max_evidence_arity
                ),
            });
        }
        if (self.accepted_evidence_assumptions_consumed != 0
            || self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed != 0
            || self.accepted_gpu_max_evidence_arity_consumed != 0)
            && (self.gpu_conditioned_evidence_facts > self.accepted_evidence_assumptions_consumed
                || self.gpu_conditioned_nonzero_arity_evidence_facts
                    > self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
                || self.gpu_conditioned_max_evidence_arity
                    > self.accepted_gpu_max_evidence_arity_consumed)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned evidence facts must be bounded by accepted GPU evidence, got \
                     facts={}/{} nonzero={}/{} max_arity={}/{}",
                    self.gpu_conditioned_evidence_facts,
                    self.accepted_evidence_assumptions_consumed,
                    self.gpu_conditioned_nonzero_arity_evidence_facts,
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
                    self.gpu_conditioned_max_evidence_arity,
                    self.accepted_gpu_max_evidence_arity_consumed
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
        let gpu_production_path_events = self.checked_gpu_production_path_events()?;
        if gpu_production_path_events == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "production probability metrics require an existing GPU exact/provenance/PIR/CNF/knowledge-compilation counter"
                    .to_string(),
            });
        }
        if self.accepted_gpu_production_path_events == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: "production probability metrics require GPU exact/provenance/PIR/CNF/knowledge-compilation work inside an accepted world-view evidence gate"
                    .to_string(),
            });
        }
        if self.accepted_gpu_production_path_events > gpu_production_path_events {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU probability production events cannot exceed total GPU production events: accepted={} total={}",
                    self.accepted_gpu_production_path_events, gpu_production_path_events
                ),
            });
        }
        if self.accepted_gpu_production_path_events < self.accepted_world_view_evidence_consumed {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production metric gate".to_string(),
                context: format!(
                    "accepted GPU probability production events must cover each accepted \
                     world-view evidence record, got accepted_events={} evidence={}",
                    self.accepted_gpu_production_path_events,
                    self.accepted_world_view_evidence_consumed
                ),
            });
        }
        self.require_accepted_gpu_world_view_evidence_trace()?;
        self.require_accepted_gpu_tuple_evidence_trace()?;
        self.require_conditioned_evidence_trace()?;
        self.require_pir_cnf_accounting()?;
        self.require_gpu_path_accounting()?;
        self.require_zero_cpu_recompute()
    }

    fn require_conditioned_evidence_metric_witness(&self) -> Result<()> {
        if self.accepted_conditioned_world_view_evidence_consumed == 0
            || self.gpu_conditioned_evidence_facts == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context:
                    "production probability metrics require accepted world-view evidence compiled as exact evidence facts"
                        .to_string(),
            });
        }
        Ok(())
    }

    /// Require the stricter metric subset for accepted world-view evidence conditioning.
    ///
    /// General production eligibility proves fixture containment plus GPU exact/PIR/CNF/
    /// knowledge-compilation reuse. This gate additionally proves at least one accepted
    /// world view was compiled into exact evidence facts rather than only used as a
    /// production-path admission gate.
    pub fn require_conditioned_evidence_metric_eligibility(&self) -> Result<()> {
        self.require_production_metric_eligibility()?;
        self.require_conditioned_evidence_metric_witness()?;
        if self.gpu_conditioned_nonzero_arity_evidence_facts > 0
            && self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed == 0
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic conditioned evidence metric gate".to_string(),
                context: format!(
                    "conditioned nonzero-arity evidence facts require accepted GPU nonzero-arity \
                     assumptions, got conditioned_nonzero={} accepted_nonzero={}",
                    self.gpu_conditioned_nonzero_arity_evidence_facts,
                    self.accepted_gpu_nonzero_arity_evidence_assumptions_consumed
                ),
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
    fn checked_trace_counter_add(current: u64, delta: u64, counter: &str) -> Result<u64> {
        current
            .checked_add(delta)
            .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production trace accounting".to_string(),
                context: format!(
                    "accepted GPU probability trace counter {counter} overflowed while adding \
                    {delta} to {current}"
                ),
            })
    }

    #[cfg(feature = "host-io")]
    fn record_gpu_exact_query_evaluation(&mut self, program: &ExactDdnnfProgram) -> Result<()> {
        checked_prob_trace_counter_inc!(self, gpu_exact_query_evaluations);
        match program.origin() {
            ExactProgramOrigin::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_exact_query_evaluations);
            }
            ExactProgramOrigin::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_exact_query_evaluations);
            }
        }
        Ok(())
    }

    #[cfg(feature = "host-io")]
    fn record_gpu_exact_gradient_evaluation(&mut self, program: &ExactDdnnfProgram) -> Result<()> {
        self.record_gpu_exact_gradient_evaluation_for_origin(program.origin())
    }

    #[cfg(feature = "host-io")]
    fn record_gpu_exact_gradient_evaluation_for_origin(
        &mut self,
        origin: ExactProgramOrigin,
    ) -> Result<()> {
        checked_prob_trace_counter_inc!(self, gpu_exact_gradient_evaluations);
        match origin {
            ExactProgramOrigin::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_exact_gradient_evaluations);
            }
            ExactProgramOrigin::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_exact_gradient_evaluations);
            }
        }
        Ok(())
    }

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
        epistemic_prob_trace_transaction!(self, {
            self.consume_accepted_evidence(&evidence)?;
            let update = circuit.apply_accepted_world_view(evidence)?;
            if update.mode == CircuitUpdateMode::IncrementalEvidence {
                self.trace.accepted_incremental_circuit_updates = Self::checked_trace_counter_add(
                    self.trace.accepted_incremental_circuit_updates,
                    1,
                    "accepted_incremental_circuit_updates",
                )?;
            }
            self.trace.require_zero_cpu_recompute()?;
            Ok(update)
        })
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
        epistemic_prob_trace_transaction!(self, {
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
        })
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
            || batch_trace.cpu_solver_search_fallbacks != 0
            || batch_trace.cpu_probability_recomputations != 0
            || batch_trace.tracked_dtoh_calls != 0
            || batch_trace.tracked_htod_calls != 0
            || batch_trace.tracked_data_plane_htod_calls != 0
            || batch_trace.per_candidate_host_round_trips != 0
            || batch_trace.violated_constraint_relations != 0
            || !batch_trace.aggregate_kernel_timing.is_recorded()
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "accepted GPU batch evidence requires complete GPU component execution and \
                     zero CPU/host fallback counters outside bounded launch metadata plus \
                     aggregate CUDA-event timing, got \
                     components={}/{}, recomposition={}, cpu_candidates={}, cpu_world_views={}, \
                     cpu_solver_search={}, cpu_probability_recompute={}, dtoh_calls={}, \
                     htod_calls={}, data_plane_htod_calls={}, launch_metadata_htod_calls={}, \
                     round_trips={}, constraint_violations={}, aggregate_timing_recorded={}",
                    batch_trace.gpu_runtime_component_executions,
                    batch_trace.component_count,
                    batch_trace.cpu_recomposition_steps,
                    batch_trace.cpu_candidate_enumerations,
                    batch_trace.cpu_world_view_validations,
                    batch_trace.cpu_solver_search_fallbacks,
                    batch_trace.cpu_probability_recomputations,
                    batch_trace.tracked_dtoh_calls,
                    batch_trace.tracked_htod_calls,
                    batch_trace.tracked_data_plane_htod_calls,
                    batch_trace.tracked_launch_metadata_htod_calls,
                    batch_trace.per_candidate_host_round_trips,
                    batch_trace.violated_constraint_relations,
                    batch_trace.aggregate_kernel_timing.is_recorded()
                ),
            });
        }
        evidence.batch.require_trace_matches_components(construct)?;

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

        self.trace.accepted_gpu_batch_evidence_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_batch_evidence_consumed,
            1,
            "accepted_gpu_batch_evidence_consumed",
        )?;
        self.trace.accepted_gpu_batch_component_evidence_consumed =
            Self::checked_trace_counter_add(
                self.trace.accepted_gpu_batch_component_evidence_consumed,
                accepted.len() as u64,
                "accepted_gpu_batch_component_evidence_consumed",
            )?;

        Ok(accepted)
    }

    #[cfg(feature = "host-io")]
    fn record_conditioned_evidence_counts(
        &mut self,
        counts: EpistemicProbConditionedEvidenceCounts,
        path: EpistemicProbConditionedEvidencePath,
    ) -> Result<()> {
        macro_rules! add_counter {
            ($field:ident, $delta:expr) => {
                self.trace.$field =
                    Self::checked_trace_counter_add(self.trace.$field, $delta, stringify!($field))?;
            };
        }

        add_counter!(accepted_conditioned_world_view_evidence_consumed, 1);
        add_counter!(gpu_conditioned_evidence_facts, counts.total as u64);
        add_counter!(
            gpu_conditioned_nonzero_arity_evidence_facts,
            counts.nonzero_arity as u64
        );
        self.trace.gpu_conditioned_max_evidence_arity = self
            .trace
            .gpu_conditioned_max_evidence_arity
            .max(counts.max_arity);
        add_counter!(
            gpu_conditioned_negative_evidence_facts,
            counts.negative as u64
        );
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                add_counter!(accepted_source_conditioned_world_view_evidence_consumed, 1);
                add_counter!(gpu_source_conditioned_evidence_facts, counts.total as u64);
                add_counter!(
                    gpu_source_conditioned_nonzero_arity_evidence_facts,
                    counts.nonzero_arity as u64
                );
                self.trace.gpu_source_conditioned_max_evidence_arity = self
                    .trace
                    .gpu_source_conditioned_max_evidence_arity
                    .max(counts.max_arity);
                add_counter!(
                    gpu_source_conditioned_negative_evidence_facts,
                    counts.negative as u64
                );
                add_counter!(
                    gpu_source_conditioned_know_evidence_facts,
                    counts.know as u64
                );
                add_counter!(
                    gpu_source_conditioned_possible_evidence_facts,
                    counts.possible as u64
                );
                add_counter!(
                    gpu_source_conditioned_not_known_evidence_facts,
                    counts.not_known as u64
                );
                add_counter!(
                    gpu_source_conditioned_not_possible_evidence_facts,
                    counts.not_possible as u64
                );
            }
            EpistemicProbConditionedEvidencePath::Program => {
                add_counter!(accepted_program_conditioned_world_view_evidence_consumed, 1);
                add_counter!(gpu_program_conditioned_evidence_facts, counts.total as u64);
                add_counter!(
                    gpu_program_conditioned_nonzero_arity_evidence_facts,
                    counts.nonzero_arity as u64
                );
                self.trace.gpu_program_conditioned_max_evidence_arity = self
                    .trace
                    .gpu_program_conditioned_max_evidence_arity
                    .max(counts.max_arity);
                add_counter!(
                    gpu_program_conditioned_negative_evidence_facts,
                    counts.negative as u64
                );
                add_counter!(
                    gpu_program_conditioned_know_evidence_facts,
                    counts.know as u64
                );
                add_counter!(
                    gpu_program_conditioned_possible_evidence_facts,
                    counts.possible as u64
                );
                add_counter!(
                    gpu_program_conditioned_not_known_evidence_facts,
                    counts.not_known as u64
                );
                add_counter!(
                    gpu_program_conditioned_not_possible_evidence_facts,
                    counts.not_possible as u64
                );
            }
        }
        add_counter!(gpu_conditioned_know_evidence_facts, counts.know as u64);
        add_counter!(
            gpu_conditioned_possible_evidence_facts,
            counts.possible as u64
        );
        add_counter!(
            gpu_conditioned_not_known_evidence_facts,
            counts.not_known as u64
        );
        add_counter!(
            gpu_conditioned_not_possible_evidence_facts,
            counts.not_possible as u64
        );
        Ok(())
    }

    fn record_accepted_gpu_production_path_events_since(
        &mut self,
        events_before: u64,
    ) -> Result<()> {
        let events_after = self.trace.checked_gpu_production_path_events()?;
        let delta = events_after.checked_sub(events_before).ok_or_else(|| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic probabilistic production trace accounting".to_string(),
                context: format!(
                    "accepted GPU probability production events decreased from {events_before} to \
                     {events_after}"
                ),
            }
        })?;
        self.trace.accepted_gpu_production_path_events = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_production_path_events,
            delta,
            "accepted_gpu_production_path_events",
        )?;
        Ok(())
    }

    /// Compile source through the existing GPU-native exact/provenance path.
    pub fn compile_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactDdnnfProgram> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            let program = ExactDdnnfProgram::compile_source_with_gpu(source, self.config)?;
            require_gpu_exact_backend(&program, "epistemic probabilistic source exact compile")?;
            checked_prob_trace_counter_inc!(self, gpu_exact_source_compiles);
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(program)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic source exact compile batch".to_string(),
                    context:
                        "batched source exact compile requires at least one accepted GPU result"
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
        })
    }

    /// Compile source once per accepted split/batch GPU epistemic component.
    pub fn compile_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactDdnnfProgram>> {
        epistemic_prob_trace_transaction!(self, {
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
        })
    }

    /// Compile a parsed program through the existing GPU-native exact/provenance path.
    pub fn compile_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactDdnnfProgram> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            let exact = ExactDdnnfProgram::compile_from_program(program, self.config)?;
            require_gpu_exact_backend(
                &exact,
                "epistemic probabilistic parsed-program exact compile",
            )?;
            checked_prob_trace_counter_inc!(self, gpu_exact_program_compiles);
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(exact)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic parsed-program exact compile batch"
                        .to_string(),
                    context: "batched parsed-program exact compile requires at least one accepted GPU result"
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
        })
    }

    /// Compile a parsed program once per accepted split/batch GPU epistemic component.
    pub fn compile_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactDdnnfProgram>> {
        epistemic_prob_trace_transaction!(self, {
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
        })
    }

    /// Compile source and evaluate queries through the existing GPU exact path after one accepted gate.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            let program = ExactDdnnfProgram::compile_source_with_gpu(source, self.config)?;
            require_gpu_exact_backend(
                &program,
                "epistemic probabilistic source exact compile/evaluate",
            )?;
            checked_prob_trace_counter_inc!(self, gpu_exact_source_compiles);
            let result = program.evaluate()?;
            checked_prob_trace_counter_inc!(self, gpu_exact_query_evaluations);
            checked_prob_trace_counter_inc!(self, gpu_source_exact_query_evaluations);
            checked_prob_trace_counter_inc!(self, gpu_knowledge_compilation_end_to_end_runs);
            checked_prob_trace_counter_inc!(self, gpu_source_knowledge_compilation_end_to_end_runs);
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(result)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic production batch".to_string(),
                    context:
                        "batched knowledge compilation requires at least one accepted GPU result"
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
                    self.compile_and_evaluate_source_with_accepted_world_view(source, evidence)?,
                );
            }
            Ok(results)
        })
    }

    /// Compile and evaluate source once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider,
                evidence,
                "epistemic probabilistic batch production",
            )?;

            let mut results = Vec::with_capacity(accepted.len());
            for evidence in &accepted {
                results.push(
                    self.compile_and_evaluate_source_with_accepted_world_view(source, evidence)?,
                );
            }
            Ok(results)
        })
    }

    /// Compile source with accepted zero-arity epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        epistemic_prob_trace_transaction!(self, {
            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            self.compile_and_evaluate_conditioned_program_with_path(
                &program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Source,
                "epistemic probabilistic conditioned source exact compile/evaluate",
            )
        })
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
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.compile_and_evaluate_conditioned_program_with_path(
                &program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Source,
                "epistemic probabilistic conditioned source exact compile/evaluate",
            )
        })
    }

    /// Compile conditioned source once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
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

            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.compile_and_evaluate_conditioned_program_with_path(
                    &program,
                    &provenance,
                    evidence,
                    EpistemicProbConditionedEvidencePath::Source,
                    "epistemic probabilistic conditioned source exact compile/evaluate",
                )?);
            }
            Ok(results)
        })
    }

    /// Compile conditioned source once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider,
                evidence,
                "epistemic probabilistic batch production",
            )?;

            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.compile_and_evaluate_conditioned_program_with_path(
                    &program,
                    &provenance,
                    evidence,
                    EpistemicProbConditionedEvidencePath::Source,
                    "epistemic probabilistic conditioned source exact compile/evaluate",
                )?);
            }
            Ok(results)
        })
    }

    /// Compile source with accepted epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_with_accepted_world_view(
        &mut self,
        source: &str,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        epistemic_prob_trace_transaction!(self, {
            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                &program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Source,
                "epistemic probabilistic conditioned source exact gradient",
            )
        })
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
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                &program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Source,
                "epistemic probabilistic conditioned source exact gradient",
            )
        })
    }

    /// Compile conditioned source gradients once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResultWithGrads>> {
        epistemic_prob_trace_transaction!(self, {
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

            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(
                    self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                        &program,
                        &provenance,
                        evidence,
                        EpistemicProbConditionedEvidencePath::Source,
                        "epistemic probabilistic conditioned source exact gradient",
                    )?,
                );
            }
            Ok(results)
        })
    }

    /// Compile conditioned source gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider,
                evidence,
                "epistemic probabilistic batch production",
            )?;

            let program = parse_program(source)?;
            let provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(
                    self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                        &program,
                        &provenance,
                        evidence,
                        EpistemicProbConditionedEvidencePath::Source,
                        "epistemic probabilistic conditioned source exact gradient",
                    )?,
                );
            }
            Ok(results)
        })
    }

    #[cfg(feature = "host-io")]
    fn compile_and_evaluate_conditioned_program_with_path(
        &mut self,
        program: &Program,
        provenance: &Provenance,
        evidence: &AcceptedWorldViewEvidence,
        path: EpistemicProbConditionedEvidencePath,
        backend_context: &'static str,
    ) -> Result<ExactResult> {
        self.require_accepted_evidence(evidence)?;
        let production_events_before = self.trace.checked_gpu_production_path_events()?;
        let (program, evidence_counts) = condition_program_with_accepted_evidence_using_provenance(
            program, provenance, evidence,
        )?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        require_gpu_exact_backend(&exact, backend_context)?;
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_exact_source_compiles);
            }
            EpistemicProbConditionedEvidencePath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_exact_program_compiles);
            }
        }
        self.record_conditioned_evidence_counts(evidence_counts, path)?;
        let result = exact.evaluate()?;
        checked_prob_trace_counter_inc!(self, gpu_exact_query_evaluations);
        checked_prob_trace_counter_inc!(self, gpu_knowledge_compilation_end_to_end_runs);
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_exact_query_evaluations);
                checked_prob_trace_counter_inc!(
                    self,
                    gpu_source_knowledge_compilation_end_to_end_runs
                );
            }
            EpistemicProbConditionedEvidencePath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_exact_query_evaluations);
                checked_prob_trace_counter_inc!(
                    self,
                    gpu_program_knowledge_compilation_end_to_end_runs
                );
            }
        }
        self.record_accepted_gpu_production_path_events_since(production_events_before)?;
        self.record_accepted_evidence(evidence)?;
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    #[cfg(feature = "host-io")]
    fn compile_and_evaluate_conditioned_program_with_grads_with_path(
        &mut self,
        program: &Program,
        provenance: &Provenance,
        evidence: &AcceptedWorldViewEvidence,
        path: EpistemicProbConditionedEvidencePath,
        backend_context: &'static str,
    ) -> Result<ExactResultWithGrads> {
        self.require_accepted_evidence(evidence)?;
        let production_events_before = self.trace.checked_gpu_production_path_events()?;
        let (program, evidence_counts) = condition_program_with_accepted_evidence_using_provenance(
            program, provenance, evidence,
        )?;
        let exact = ExactDdnnfProgram::compile_from_program(&program, self.config)?;
        require_gpu_exact_backend(&exact, backend_context)?;
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_exact_source_compiles);
            }
            EpistemicProbConditionedEvidencePath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_exact_program_compiles);
            }
        }
        self.record_conditioned_evidence_counts(evidence_counts, path)?;
        let result = exact.evaluate_gpu_with_grads()?;
        let origin = match path {
            EpistemicProbConditionedEvidencePath::Source => ExactProgramOrigin::Source,
            EpistemicProbConditionedEvidencePath::Program => ExactProgramOrigin::Program,
        };
        self.record_gpu_exact_gradient_evaluation_for_origin(origin)?;
        checked_prob_trace_counter_inc!(self, gpu_knowledge_compilation_end_to_end_runs);
        match path {
            EpistemicProbConditionedEvidencePath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_conditioned_gradient_evaluations);
                checked_prob_trace_counter_inc!(
                    self,
                    gpu_source_knowledge_compilation_end_to_end_runs
                );
            }
            EpistemicProbConditionedEvidencePath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_conditioned_gradient_evaluations);
                checked_prob_trace_counter_inc!(
                    self,
                    gpu_program_knowledge_compilation_end_to_end_runs
                );
            }
        }
        self.record_accepted_gpu_production_path_events_since(production_events_before)?;
        self.record_accepted_evidence(evidence)?;
        self.trace.require_zero_cpu_recompute()?;
        Ok(result)
    }

    /// Compile a parsed program with accepted epistemic assumptions as exact evidence.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        epistemic_prob_trace_transaction!(self, {
            let provenance = extract_from_program(program)?;
            self.compile_and_evaluate_conditioned_program_with_path(
                program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Program,
                "epistemic probabilistic conditioned parsed-program exact compile/evaluate",
            )
        })
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
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let provenance = extract_from_program(program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.compile_and_evaluate_conditioned_program_with_path(
                program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Program,
                "epistemic probabilistic conditioned parsed-program exact compile/evaluate",
            )
        })
    }

    /// Compile conditioned parsed program once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
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

            let provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.compile_and_evaluate_conditioned_program_with_path(
                    program,
                    &provenance,
                    evidence,
                    EpistemicProbConditionedEvidencePath::Program,
                    "epistemic probabilistic conditioned parsed-program exact compile/evaluate",
                )?);
            }
            Ok(results)
        })
    }

    /// Compile conditioned parsed program once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider,
                evidence,
                "epistemic probabilistic batch production",
            )?;

            let provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.compile_and_evaluate_conditioned_program_with_path(
                    program,
                    &provenance,
                    evidence,
                    EpistemicProbConditionedEvidencePath::Program,
                    "epistemic probabilistic conditioned parsed-program exact compile/evaluate",
                )?);
            }
            Ok(results)
        })
    }

    /// Compile a parsed program with accepted epistemic assumptions as exact evidence and evaluate gradients.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        epistemic_prob_trace_transaction!(self, {
            let provenance = extract_from_program(program)?;
            self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Program,
                "epistemic probabilistic conditioned parsed-program exact gradient",
            )
        })
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
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let provenance = extract_from_program(program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                program,
                &provenance,
                evidence,
                EpistemicProbConditionedEvidencePath::Program,
                "epistemic probabilistic conditioned parsed-program exact gradient",
            )
        })
    }

    /// Compile conditioned parsed-program gradients once per accepted GPU epistemic execution result.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<ExactResultWithGrads>> {
        epistemic_prob_trace_transaction!(self, {
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

            let provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(
                    self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                        program,
                        &provenance,
                        evidence,
                        EpistemicProbConditionedEvidencePath::Program,
                        "epistemic probabilistic conditioned parsed-program exact gradient",
                    )?,
                );
            }
            Ok(results)
        })
    }

    /// Compile conditioned parsed-program gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider,
                evidence,
                "epistemic probabilistic batch production",
            )?;

            let provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(
                    self.compile_and_evaluate_conditioned_program_with_grads_with_path(
                        program,
                        &provenance,
                        evidence,
                        EpistemicProbConditionedEvidencePath::Program,
                        "epistemic probabilistic conditioned parsed-program exact gradient",
                    )?,
                );
            }
            Ok(results)
        })
    }

    /// Compile a parsed program and evaluate queries through the existing GPU exact path.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_with_accepted_world_view(
        &mut self,
        program: &Program,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            let exact = ExactDdnnfProgram::compile_from_program(program, self.config)?;
            require_gpu_exact_backend(
                &exact,
                "epistemic probabilistic parsed-program exact compile/evaluate",
            )?;
            checked_prob_trace_counter_inc!(self, gpu_exact_program_compiles);
            let result = exact.evaluate()?;
            checked_prob_trace_counter_inc!(self, gpu_exact_query_evaluations);
            checked_prob_trace_counter_inc!(self, gpu_program_exact_query_evaluations);
            checked_prob_trace_counter_inc!(self, gpu_knowledge_compilation_end_to_end_runs);
            checked_prob_trace_counter_inc!(
                self,
                gpu_program_knowledge_compilation_end_to_end_runs
            );
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(result)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic parsed-program production batch"
                        .to_string(),
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
        })
    }

    /// Compile and evaluate a parsed program once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn compile_and_evaluate_program_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
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
        })
    }

    /// Encode source through the existing GPU PIR and CNF production path.
    pub fn encode_source_pir_cnf_with_accepted_world_view(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let program = parse_program(source)?;
            let base_provenance = extract_from_program(&program)?;
            self.encode_program_pir_cnf_with_base_provenance(
                &program,
                &base_provenance,
                provider,
                evidence,
                EpistemicProbPirCnfPath::Source,
            )
        })
    }

    /// Encode source PIR/CNF after accepted GPU epistemic execution.
    pub fn encode_source_pir_cnf_with_gpu_execution_result(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let program = parse_program(source)?;
            let base_provenance = extract_from_program(&program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &base_provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.encode_program_pir_cnf_with_base_provenance(
                &program,
                &base_provenance,
                provider,
                evidence,
                EpistemicProbPirCnfPath::Source,
            )
        })
    }

    /// Encode source PIR/CNF once per accepted GPU epistemic execution result.
    pub fn encode_source_pir_cnf_for_gpu_execution_results(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic source PIR/CNF production batch"
                        .to_string(),
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

            let program = parse_program(source)?;
            let base_provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &base_provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.encode_program_pir_cnf_with_base_provenance(
                    &program,
                    &base_provenance,
                    provider,
                    evidence,
                    EpistemicProbPirCnfPath::Source,
                )?);
            }
            Ok(results)
        })
    }

    /// Encode source PIR/CNF once per accepted split/batch GPU epistemic component.
    pub fn encode_source_pir_cnf_for_gpu_batch_execution_result(
        &mut self,
        source: &str,
        provider: &Arc<CudaKernelProvider>,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider.as_ref(),
                evidence,
                "epistemic probabilistic source PIR/CNF batch production",
            )?;

            let program = parse_program(source)?;
            let base_provenance = extract_from_program(&program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &base_provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.encode_program_pir_cnf_with_base_provenance(
                    &program,
                    &base_provenance,
                    provider,
                    evidence,
                    EpistemicProbPirCnfPath::Source,
                )?);
            }
            Ok(results)
        })
    }

    /// Encode a parsed program through the existing GPU PIR and CNF production path.
    pub fn encode_program_pir_cnf_with_accepted_world_view(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let base_provenance = extract_from_program(program)?;
            self.encode_program_pir_cnf_with_base_provenance(
                program,
                &base_provenance,
                provider,
                evidence,
                EpistemicProbPirCnfPath::Program,
            )
        })
    }

    /// Encode parsed-program PIR/CNF after accepted GPU epistemic execution.
    pub fn encode_program_pir_cnf_with_gpu_execution_result(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        result: &EpistemicGpuExecutionResult,
        assumptions: Vec<EpistemicAssumption>,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived = assumptions.is_empty();
            let evidence = AcceptedWorldViewEvidence::from_gpu_execution_result(
                provider,
                result,
                assumptions,
            )?;
            let base_provenance = extract_from_program(program)?;
            let filtered_evidence;
            let evidence = if auto_derived {
                filtered_evidence =
                    evidence_with_provenance_backed_assumptions(&evidence, &base_provenance)?;
                &filtered_evidence
            } else {
                &evidence
            };
            self.encode_program_pir_cnf_with_base_provenance(
                program,
                &base_provenance,
                provider,
                evidence,
                EpistemicProbPirCnfPath::Program,
            )
        })
    }

    /// Encode parsed-program PIR/CNF once per accepted GPU epistemic execution result.
    pub fn encode_program_pir_cnf_for_gpu_execution_results(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence_records: &[EpistemicProbGpuExecutionEvidence<'_>],
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        epistemic_prob_trace_transaction!(self, {
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

            let base_provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (record, evidence) in evidence_records.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if record.assumptions.is_empty() {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &base_provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.encode_program_pir_cnf_with_base_provenance(
                    program,
                    &base_provenance,
                    provider,
                    evidence,
                    EpistemicProbPirCnfPath::Program,
                )?);
            }
            Ok(results)
        })
    }

    /// Encode parsed-program PIR/CNF once per accepted split/batch GPU epistemic component.
    pub fn encode_program_pir_cnf_for_gpu_batch_execution_result(
        &mut self,
        program: &Program,
        provider: &Arc<CudaKernelProvider>,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<EpistemicProbPirCnfEvidence>> {
        epistemic_prob_trace_transaction!(self, {
            let auto_derived_by_component = evidence
                .assumptions_by_component
                .iter()
                .map(|assumptions| assumptions.is_empty())
                .collect::<Vec<_>>();
            let accepted = self.accepted_world_views_from_gpu_batch_execution_evidence(
                provider.as_ref(),
                evidence,
                "epistemic probabilistic parsed-program PIR/CNF batch production",
            )?;

            let base_provenance = extract_from_program(program)?;
            let mut results = Vec::with_capacity(accepted.len());
            for (auto_derived, evidence) in auto_derived_by_component.iter().zip(accepted.iter()) {
                let filtered_evidence;
                let evidence = if *auto_derived {
                    filtered_evidence =
                        evidence_with_provenance_backed_assumptions(evidence, &base_provenance)?;
                    &filtered_evidence
                } else {
                    evidence
                };
                results.push(self.encode_program_pir_cnf_with_base_provenance(
                    program,
                    &base_provenance,
                    provider,
                    evidence,
                    EpistemicProbPirCnfPath::Program,
                )?);
            }
            Ok(results)
        })
    }

    /// Evaluate GPU exact query probabilities after accepted world-view evidence was consumed.
    #[cfg(feature = "host-io")]
    pub fn evaluate(
        &mut self,
        program: &ExactDdnnfProgram,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResult> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            require_gpu_exact_backend(program, "epistemic probabilistic exact query evaluation")?;
            let result = program.evaluate()?;
            self.record_gpu_exact_query_evaluation(program)?;
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(result)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic query evaluation production batch"
                        .to_string(),
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
        })
    }

    /// Evaluate GPU exact query probabilities once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn evaluate_for_gpu_batch_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResult>> {
        epistemic_prob_trace_transaction!(self, {
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
        })
    }

    /// Evaluate GPU exact gradients after accepted world-view evidence was consumed.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads(
        &mut self,
        program: &ExactDdnnfProgram,
        evidence: &AcceptedWorldViewEvidence,
    ) -> Result<ExactResultWithGrads> {
        epistemic_prob_trace_transaction!(self, {
            self.require_accepted_evidence(evidence)?;
            let production_events_before = self.trace.checked_gpu_production_path_events()?;
            require_gpu_exact_backend(
                program,
                "epistemic probabilistic exact gradient evaluation",
            )?;
            let result = program.evaluate_gpu_with_grads()?;
            self.record_gpu_exact_gradient_evaluation(program)?;
            self.record_accepted_gpu_production_path_events_since(production_events_before)?;
            self.record_accepted_evidence(evidence)?;
            self.trace.require_zero_cpu_recompute()?;
            Ok(result)
        })
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
        epistemic_prob_trace_transaction!(self, {
            if evidence_records.is_empty() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic probabilistic gradient evaluation production batch"
                        .to_string(),
                    context:
                        "batched gradient evaluation requires at least one accepted GPU result"
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
        })
    }

    /// Evaluate GPU exact gradients once per accepted split/batch GPU epistemic component.
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads_for_gpu_batch_execution_result(
        &mut self,
        program: &ExactDdnnfProgram,
        provider: &CudaKernelProvider,
        evidence: EpistemicProbGpuBatchExecutionEvidence<'_>,
    ) -> Result<Vec<ExactResultWithGrads>> {
        epistemic_prob_trace_transaction!(self, {
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
        })
    }

    fn consume_accepted_evidence(&mut self, evidence: &AcceptedWorldViewEvidence) -> Result<()> {
        self.require_accepted_evidence(evidence)?;
        self.record_accepted_evidence(evidence)?;
        self.trace.require_zero_cpu_recompute()
    }

    fn require_accepted_evidence(&self, evidence: &AcceptedWorldViewEvidence) -> Result<()> {
        if evidence.world_count() == 0 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted world-view evidence".to_string(),
                context: "probabilistic production path requires a non-empty accepted world view"
                    .to_string(),
            });
        }
        if evidence.gpu_epistemic_mode().is_none() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted world-view evidence".to_string(),
                context: "probabilistic production path requires GPU execution evidence; CPU \
                     world-view evidence is oracle-only"
                    .to_string(),
            });
        }
        self.trace.require_zero_cpu_recompute()
    }

    fn record_accepted_evidence(&mut self, evidence: &AcceptedWorldViewEvidence) -> Result<()> {
        self.trace.accepted_world_view_evidence_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_world_view_evidence_consumed,
            1,
            "accepted_world_view_evidence_consumed",
        )?;
        match evidence.gpu_epistemic_mode() {
            Some(EirEpistemicMode::G91) => {
                self.trace.accepted_g91_world_view_evidence_consumed =
                    Self::checked_trace_counter_add(
                        self.trace.accepted_g91_world_view_evidence_consumed,
                        1,
                        "accepted_g91_world_view_evidence_consumed",
                    )?;
            }
            Some(EirEpistemicMode::Faeel) => {
                self.trace.accepted_faeel_world_view_evidence_consumed =
                    Self::checked_trace_counter_add(
                        self.trace.accepted_faeel_world_view_evidence_consumed,
                        1,
                        "accepted_faeel_world_view_evidence_consumed",
                    )?;
            }
            None => {}
        }
        self.trace.accepted_evidence_assumptions_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_evidence_assumptions_consumed,
            evidence.assumption_count() as u64,
            "accepted_evidence_assumptions_consumed",
        )?;
        self.trace
            .accepted_gpu_nonzero_arity_evidence_assumptions_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_nonzero_arity_evidence_assumptions_consumed,
                evidence.nonzero_arity_assumption_count() as u64,
                "accepted_gpu_nonzero_arity_evidence_assumptions_consumed",
            )?;
        if evidence.max_assumption_arity() > u32::MAX as usize {
            return Err(XlogError::ResourceExhausted {
                context: "accepted GPU probability evidence max arity".to_string(),
                estimated_bytes: evidence.max_assumption_arity() as u64,
                budget_bytes: u32::MAX as u64,
            });
        }
        self.trace.accepted_gpu_max_evidence_arity_consumed = self
            .trace
            .accepted_gpu_max_evidence_arity_consumed
            .max(evidence.max_assumption_arity() as u32);
        self.trace.accepted_gpu_tuple_key_column_reads_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_tuple_key_column_reads_consumed,
            evidence.gpu_tuple_key_column_reads() as u64,
            "accepted_gpu_tuple_key_column_reads_consumed",
        )?;
        self.trace.accepted_gpu_final_tuple_row_filters_consumed = Self::checked_trace_counter_add(
            self.trace.accepted_gpu_final_tuple_row_filters_consumed,
            evidence.gpu_final_tuple_row_filters() as u64,
            "accepted_gpu_final_tuple_row_filters_consumed",
        )?;
        self.trace
            .accepted_gpu_final_tuple_negated_row_filters_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_final_tuple_negated_row_filters_consumed,
                evidence.gpu_final_tuple_negated_row_filters() as u64,
                "accepted_gpu_final_tuple_negated_row_filters_consumed",
            )?;
        self.trace
            .accepted_gpu_row_specific_membership_row_capacity_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_row_specific_membership_row_capacity_consumed,
                evidence.gpu_row_specific_membership_row_capacity() as u64,
                "accepted_gpu_row_specific_membership_row_capacity_consumed",
            )?;
        self.trace
            .accepted_gpu_row_filter_fallback_row_capacity_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_row_filter_fallback_row_capacity_consumed,
                evidence.gpu_row_filter_fallback_row_capacity() as u64,
                "accepted_gpu_row_filter_fallback_row_capacity_consumed",
            )?;
        self.trace
            .accepted_gpu_constraint_relations_checked_consumed = Self::checked_trace_counter_add(
            self.trace
                .accepted_gpu_constraint_relations_checked_consumed,
            evidence.gpu_checked_constraint_relations() as u64,
            "accepted_gpu_constraint_relations_checked_consumed",
        )?;
        self.trace
            .accepted_gpu_constraint_row_count_device_reads_consumed =
            Self::checked_trace_counter_add(
                self.trace
                    .accepted_gpu_constraint_row_count_device_reads_consumed,
                evidence.gpu_constraint_row_count_device_reads() as u64,
                "accepted_gpu_constraint_row_count_device_reads_consumed",
            )?;
        Ok(())
    }

    fn encode_program_pir_cnf_with_base_provenance(
        &mut self,
        program: &Program,
        base_provenance: &Provenance,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
        path: EpistemicProbPirCnfPath,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        self.require_accepted_evidence(evidence)?;
        let conditioned_provenance = if evidence.assumptions().is_empty() {
            None
        } else {
            let conditioned_program = condition_program_with_available_evidence_using_provenance(
                program,
                base_provenance,
                evidence,
            )?;
            Some(extract_from_program(&conditioned_program)?)
        };
        let provenance = conditioned_provenance.as_ref().unwrap_or(base_provenance);
        self.encode_provenance_pir_cnf_with_accepted_world_view(
            provenance, provider, evidence, path,
        )
    }

    fn encode_provenance_pir_cnf_with_accepted_world_view(
        &mut self,
        provenance: &Provenance,
        provider: &Arc<CudaKernelProvider>,
        evidence: &AcceptedWorldViewEvidence,
        path: EpistemicProbPirCnfPath,
    ) -> Result<EpistemicProbPirCnfEvidence> {
        self.require_accepted_evidence(evidence)?;
        let production_events_before = self.trace.checked_gpu_production_path_events()?;
        let roots = production_pir_roots(provenance)?;
        if roots.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted probabilistic PIR/CNF production path".to_string(),
                context: "GPU PIR/CNF evidence requires at least one query, evidence, or probabilistic variable root".to_string(),
            });
        }
        let gpu_pir = GpuPirGraph::from_host(&provenance.pir, provider)?;
        checked_prob_trace_counter_inc!(self, gpu_pir_graph_uploads);
        match path {
            EpistemicProbPirCnfPath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_pir_graph_uploads);
            }
            EpistemicProbPirCnfPath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_pir_graph_uploads);
            }
        }
        let gpu_roots = GpuPirRoots::from_host(&roots, provider)?;
        let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, provider)?;
        checked_prob_trace_counter_inc!(self, gpu_cnf_encodes);
        match path {
            EpistemicProbPirCnfPath::Source => {
                checked_prob_trace_counter_inc!(self, gpu_source_cnf_encodes);
            }
            EpistemicProbPirCnfPath::Program => {
                checked_prob_trace_counter_inc!(self, gpu_program_cnf_encodes);
            }
        }
        let pir_cnf_evidence = EpistemicProbPirCnfEvidence {
            pir_nodes: provenance.pir.len(),
            root_count: roots.len(),
            cnf_var_cap: encoding.cnf.var_cap,
            cnf_clause_cap: encoding.cnf.clause_cap,
            cnf_lit_cap: encoding.cnf.lit_cap,
        };
        self.record_accepted_gpu_production_path_events_since(production_events_before)?;
        self.record_accepted_evidence(evidence)?;
        self.trace.require_zero_cpu_recompute()?;
        Ok(pir_cnf_evidence)
    }
}

fn require_gpu_exact_backend(program: &ExactDdnnfProgram, construct: &'static str) -> Result<()> {
    if !program.uses_gpu_production_backend() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: construct.to_string(),
            context:
                "production probability metrics require a compiled GPU D4 exact/provenance/PIR/CNF backend; \
                      empty roots and count-lift-only evaluation cannot satisfy accepted epistemic probability reuse"
                    .to_string(),
        });
    }
    Ok(())
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

fn condition_program_with_available_evidence_using_provenance(
    program: &Program,
    base_provenance: &Provenance,
    evidence: &AcceptedWorldViewEvidence,
) -> Result<Program> {
    if evidence.assumptions().is_empty() {
        return Ok(program.clone());
    }

    let mut program = program.clone();
    let mut applied = 0usize;
    for assumption in evidence.assumptions() {
        validate_conditioned_assumption_shape(assumption)?;
        let Some(atom) = conditioned_evidence_atom_for_assumption(base_provenance, assumption)
        else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted probabilistic PIR/CNF evidence conditioning".to_string(),
                context: format!(
                    "accepted {} PIR/CNF evidence for {}/{} has no provenance formula; partial \
                     world-view evidence cannot satisfy production conditioning metrics",
                    assumption.evidence_literal(),
                    assumption.predicate,
                    assumption.arity
                ),
            });
        };
        program.evidence.push(Evidence {
            atom,
            value: assumption.value,
        });
        applied += 1;
    }

    if applied == 0 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted probabilistic PIR/CNF evidence conditioning".to_string(),
            context: "PIR/CNF encoding requires at least one accepted epistemic assumption to match existing probabilistic provenance"
                .to_string(),
        });
    }

    Ok(program)
}

#[cfg(feature = "host-io")]
fn condition_program_with_accepted_evidence_using_provenance(
    program: &Program,
    provenance: &Provenance,
    evidence: &AcceptedWorldViewEvidence,
) -> Result<(Program, EpistemicProbConditionedEvidenceCounts)> {
    if evidence.assumptions().is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted probabilistic evidence conditioning".to_string(),
            context: "conditioned exact path requires at least one accepted epistemic assumption"
                .to_string(),
        });
    }

    let mut counts = EpistemicProbConditionedEvidenceCounts::default();
    for assumption in evidence.assumptions() {
        validate_conditioned_assumption_shape(assumption)?;
        record_conditioned_assumption_counts(&mut counts, assumption);
    }
    let mut program = program.clone();
    for assumption in evidence.assumptions() {
        let Some(atom) = conditioned_evidence_atom_for_assumption(provenance, assumption) else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "accepted probabilistic evidence conditioning".to_string(),
                context: format!(
                    "accepted {} exact evidence for {}/{} has no provenance formula; \
                     vacuous evidence cannot satisfy production conditioning metrics",
                    assumption.evidence_literal(),
                    assumption.predicate,
                    assumption.arity
                ),
            });
        };
        program.evidence.push(Evidence {
            atom,
            value: assumption.value,
        });
    }

    Ok((program, counts))
}

fn validate_conditioned_assumption_shape(assumption: &EpistemicAssumption) -> Result<()> {
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
    Ok(())
}

#[cfg(feature = "host-io")]
fn record_conditioned_assumption_counts(
    counts: &mut EpistemicProbConditionedEvidenceCounts,
    assumption: &EpistemicAssumption,
) {
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
}

fn evidence_term_variants(term: &EpistemicEvidenceTerm) -> Vec<(Value, Term)> {
    match term {
        EpistemicEvidenceTerm::Integer(value) => vec![(Value::I64(*value), Term::Integer(*value))],
        EpistemicEvidenceTerm::String(value) => {
            let symbol_id = symbol::intern(value);
            vec![
                (Value::String(value.clone()), Term::String(value.clone())),
                (Value::Symbol(symbol_id), Term::Symbol(symbol_id)),
            ]
        }
        EpistemicEvidenceTerm::Symbol(value) => {
            let string_value = symbol::resolve(*value);
            vec![
                (Value::Symbol(*value), Term::Symbol(*value)),
                (
                    Value::String(string_value.clone()),
                    Term::String(string_value),
                ),
            ]
        }
    }
}

fn conditioned_evidence_atom_for_assumption(
    provenance: &Provenance,
    assumption: &EpistemicAssumption,
) -> Option<Atom> {
    if assumption.terms.is_empty() {
        if provenance
            .query_formula(&assumption.predicate, &[])
            .is_some()
        {
            return Some(Atom {
                predicate: assumption.predicate.clone(),
                terms: Vec::new(),
            });
        }
        return None;
    }

    let variants = assumption
        .terms
        .iter()
        .map(evidence_term_variants)
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(variants.len());
    let mut terms = Vec::with_capacity(variants.len());
    conditioned_evidence_atom_from_variants(
        provenance,
        assumption,
        &variants,
        0,
        &mut values,
        &mut terms,
    )
}

fn conditioned_evidence_atom_from_variants(
    provenance: &Provenance,
    assumption: &EpistemicAssumption,
    variants: &[Vec<(Value, Term)>],
    index: usize,
    values: &mut Vec<Value>,
    terms: &mut Vec<Term>,
) -> Option<Atom> {
    if index == variants.len() {
        if provenance
            .query_formula(&assumption.predicate, values)
            .is_some()
        {
            return Some(Atom {
                predicate: assumption.predicate.clone(),
                terms: terms.clone(),
            });
        }
        return None;
    }

    for (value, term) in &variants[index] {
        values.push(value.clone());
        terms.push(term.clone());
        if let Some(atom) = conditioned_evidence_atom_from_variants(
            provenance,
            assumption,
            variants,
            index + 1,
            values,
            terms,
        ) {
            return Some(atom);
        }
        values.pop();
        terms.pop();
    }
    None
}

fn assumption_has_provenance_formula(
    provenance: &Provenance,
    assumption: &EpistemicAssumption,
) -> bool {
    conditioned_evidence_atom_for_assumption(provenance, assumption).is_some()
}

fn evidence_with_provenance_backed_assumptions(
    evidence: &AcceptedWorldViewEvidence,
    provenance: &Provenance,
) -> Result<AcceptedWorldViewEvidence> {
    let assumptions = evidence
        .assumptions()
        .iter()
        .filter(|assumption| assumption_has_provenance_formula(provenance, assumption))
        .cloned()
        .collect::<Vec<_>>();
    if assumptions.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "accepted probabilistic PIR/CNF evidence conditioning".to_string(),
            context: "PIR/CNF encoding requires at least one accepted epistemic assumption to match existing probabilistic provenance"
                .to_string(),
        });
    }
    Ok(evidence.with_assumptions(assumptions))
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
