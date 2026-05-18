//! Production GPU exact-path adapter for accepted epistemic evidence.
//!
//! This module is intentionally thin. It gates probabilistic execution on
//! accepted world-view evidence, then routes into the existing GPU-native exact
//! provenance path instead of using the bounded epistemic fixture circuit.

use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_logic::ast::Program;
use xlog_runtime::EpistemicGpuExecutionResult;

use crate::epistemic::{AcceptedWorldViewEvidence, EpistemicAssumption};
use crate::exact::{ExactDdnnfProgram, GpuConfig};
#[cfg(feature = "host-io")]
use crate::exact::{ExactResult, ExactResultWithGrads};

/// Trace counters proving the production adapter stayed on the GPU exact path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicProbProductionTrace {
    /// Number of source compiles routed through `ExactDdnnfProgram`.
    pub gpu_exact_source_compiles: u64,
    /// Number of parsed-program compiles routed through `ExactDdnnfProgram`.
    pub gpu_exact_program_compiles: u64,
    /// Number of accepted world-view evidence objects consumed as a gate.
    pub accepted_world_view_evidence_consumed: u64,
    /// Number of GPU exact query evaluations routed through `ExactDdnnfProgram`.
    pub gpu_exact_query_evaluations: u64,
    /// Number of GPU gradient evaluations routed through `ExactDdnnfProgram`.
    pub gpu_exact_gradient_evaluations: u64,
    /// CPU-only probability recomputations performed by this adapter.
    pub cpu_only_probability_recomputations: u64,
    /// Fixture `EpistemicCircuit` evaluations performed by this adapter.
    pub fixture_circuit_evaluations: u64,
}

impl EpistemicProbProductionTrace {
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
}

/// Thin adapter from accepted epistemic evidence to the existing GPU exact path.
pub struct EpistemicProbProductionAdapter {
    config: GpuConfig,
    trace: EpistemicProbProductionTrace,
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
        self.trace.require_zero_cpu_recompute()
    }
}
