// Epistemic evidence handoff: run a `know`/`possible` program on the GPU and
// condition an exact probabilistic query on the accepted world view.
//
// The #[pyclass] struct definitions remain in lib.rs, matching program.rs.

use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "host-io")]
use std::sync::Arc;

use pyo3::prelude::*;
#[cfg(feature = "host-io")]
use pyo3::types::PyDict;

#[cfg(feature = "host-io")]
use xlog_core::{ScalarType, Schema};
#[cfg(feature = "host-io")]
use xlog_cuda::{CudaKernelProvider, DlpackManagedTensor};
#[cfg(feature = "host-io")]
use xlog_prob::epistemic::AcceptedWorldViewEvidence;
#[cfg(feature = "host-io")]
use xlog_prob::epistemic_production::{
    EpistemicProbProductionAdapter, EpistemicProbProductionTrace,
};
#[cfg(feature = "host-io")]
use xlog_prob::exact::{ExactResult, GpuConfig};

#[cfg(feature = "host-io")]
use super::program::atom_to_string;
#[cfg(feature = "host-io")]
use super::{dlpack_capsule_from_tensor, enforce_call_memory_limit};
use super::{
    types, CompiledConditionedProgram, CompiledLogicProgram, EpistemicEvalResult, EpistemicEvidence,
};

#[pymethods]
impl CompiledLogicProgram {
    /// Compile this program's accepted epistemic evidence and `prob_source` once.
    ///
    /// The returned handle can be evaluated repeatedly while independent
    /// probabilistic fact priors are changed atomically. It does not accept
    /// caller-supplied input relations, matching `evaluate_conditioned`.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (prob_source, memory_mb=None))]
    pub fn prepare_conditioned(
        &self,
        py: Python<'_>,
        prob_source: &str,
        memory_mb: Option<u64>,
    ) -> PyResult<CompiledConditionedProgram> {
        enforce_call_memory_limit(&self.provider, memory_mb)?;
        let logic_program = self.program.clone();
        let evidence_provider = self.provider.clone();
        let inputs = HashMap::new();
        let prob_source = prob_source.to_owned();
        let (program, result_provider) = py
            .detach(move || {
                let evidence =
                    logic_program.execute_epistemic_evidence(evidence_provider.clone(), inputs)?;
                let mut config = GpuConfig::default();
                config.device_ordinal = evidence_provider.device().ordinal();
                config.memory_bytes = evidence_provider.memory().budget().device_bytes;
                let mut adapter = EpistemicProbProductionAdapter::new(config);
                let accepted = AcceptedWorldViewEvidence::from_gpu_execution_result(
                    &evidence_provider,
                    &evidence,
                    Vec::new(),
                )?;
                let program = adapter
                    .prepare_conditioned_source_with_accepted_world_view(&prob_source, &accepted)?;
                Ok::<_, xlog_core::XlogError>((program, evidence_provider))
            })
            .map_err(types::xlog_err)?;
        Ok(CompiledConditionedProgram {
            program,
            result_provider,
        })
    }

    #[cfg(not(feature = "host-io"))]
    #[pyo3(signature = (prob_source, memory_mb=None))]
    pub fn prepare_conditioned(
        &self,
        _py: Python<'_>,
        prob_source: &str,
        memory_mb: Option<u64>,
    ) -> PyResult<CompiledConditionedProgram> {
        let _ = (prob_source, memory_mb);
        Err(types::host_io_disabled_pyerr())
    }

    /// Run this epistemic program on the GPU and condition `prob_source` on what it knows.
    ///
    /// The compiled program must contain epistemic operators (`know`, `possible`, ...)
    /// AND lower to a single-component epistemic plan: ordinary Datalog programs are
    /// rejected, because there is no accepted world view to condition on. Only facts
    /// declared in the epistemic program's own source are used to build that world view.
    ///
    /// Both epistemic modes are reachable here. FAEEL programs and non-recursive
    /// G91-compatibility programs (`#pragma epistemic_mode = g91`) both lower to a
    /// single-component epistemic plan and condition normally;
    /// `epistemic_evidence().epistemic_mode` names the mode, and the trace's
    /// `accepted_faeel_world_view_evidence_consumed` /
    /// `accepted_g91_world_view_evidence_consumed` pair says which one actually supplied
    /// the evidence. Only the *recursive* G91 shapes (positive `possible` cycles that
    /// need tuple-level compatibility) compile to a dedicated G91-compatibility plan and
    /// are rejected at plan level, alongside split, stratified and WFS plans.
    ///
    /// LIMITATION: unlike `evaluate`, this method does not accept `dlpack_inputs`.
    /// Caller-supplied input relations are NOT consulted — if the epistemic program
    /// depends on a relation that is normally supplied at call time via
    /// `evaluate(dlpack_inputs=...)`, that relation is empty here and no world view is
    /// accepted. This method then RAISES `RuntimeError` ("Unsupported epistemic
    /// construct: accepted GPU world-view evidence ... probabilistic evidence requires
    /// non-empty accepted GPU final output"); it does NOT fall back to the unconditioned
    /// prior. That is fail-closed by design: a conditioned query that silently became
    /// unconditioned would be indistinguishable from a successful one, which is exactly
    /// the failure the trace counters exist to make visible. To test for the case
    /// without catching an exception, call `epistemic_evidence()` first — it reports
    /// `accepted_world_views == 0` without raising.
    ///
    /// The returned trace must show a non-zero `gpu_conditioned_evidence_facts` —
    /// otherwise the conditioning did not reach the GPU exact path.
    /// `gpu_conditioned_evidence_facts` is the total the
    /// engine itself validates; the per-class counters
    /// (`gpu_conditioned_know_evidence_facts`,
    /// `gpu_conditioned_possible_evidence_facts`,
    /// `gpu_conditioned_not_known_evidence_facts`,
    /// `gpu_conditioned_not_possible_evidence_facts`) break it down. A `possible`-only
    /// or negated-evidence program conditions correctly with the `know` counter at `0`,
    /// so check the total, not the `know` class alone.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (prob_source, memory_mb=None))]
    pub fn evaluate_conditioned(
        &self,
        py: Python<'_>,
        prob_source: &str,
        memory_mb: Option<u64>,
    ) -> PyResult<EpistemicEvalResult> {
        enforce_call_memory_limit(&self.provider, memory_mb)?;
        let program = self.program.clone();
        let provider = self.provider.clone();
        let inputs = HashMap::new();
        let prob_source = prob_source.to_owned();

        let prepared = py
            .detach(move || {
                let evidence = program.execute_epistemic_evidence(provider.clone(), inputs)?;

                // IMPORTANT: not `GpuConfig::default()`. The adapter does not reuse our
                // provider — `ExactDdnnfProgram::compile_provenance_with_gpu` builds its
                // OWN device from `config.device_ordinal` and `config.memory_bytes`.
                //
                // `GpuConfig` is `#[non_exhaustive]`, so it cannot be built with a struct
                // literal naming every field from outside `xlog-prob`; start from
                // `Default::default()` and overwrite the two fields we need to match.
                let mut config = GpuConfig::default();
                config.device_ordinal = provider.device().ordinal();
                config.memory_bytes = provider.memory().budget().device_bytes;
                let mut adapter = EpistemicProbProductionAdapter::new(config);
                let accepted = AcceptedWorldViewEvidence::from_gpu_execution_result(
                    &provider,
                    &evidence,
                    Vec::new(),
                )?;
                let exact = adapter
                    .compile_and_evaluate_conditioned_source_with_accepted_world_view(
                        &prob_source,
                        &accepted,
                    )?;
                let trace = adapter.trace();

                prepare_epistemic_eval_result(&provider, exact, trace)
            })
            .map_err(types::xlog_err)?;

        pack_epistemic_eval_result(py, prepared)
    }

    #[cfg(not(feature = "host-io"))]
    #[pyo3(signature = (prob_source, memory_mb=None))]
    pub fn evaluate_conditioned(
        &self,
        _py: Python<'_>,
        prob_source: &str,
        memory_mb: Option<u64>,
    ) -> PyResult<EpistemicEvalResult> {
        let _ = (prob_source, memory_mb);
        Err(types::host_io_disabled_pyerr())
    }

    /// Run this epistemic program on the GPU and report what it accepted.
    ///
    /// Diagnostic counterpart of `evaluate_conditioned`: it answers whether the
    /// `know`-broadcast happened at all, without involving the probabilistic tier.
    ///
    /// LIMITATION: like `evaluate_conditioned`, this only ever sees facts declared in
    /// the epistemic program's own source; it does not accept caller-supplied input
    /// relations. A program that depends on such a relation reports
    /// `accepted_world_views == 0`, `accepted_candidates == 0` and
    /// `final_output_rows == 0` here — without raising. The operator censuses
    /// (`know_operator_count`, `possible_operator_count`) are read off the plan, not the
    /// execution, so they stay non-zero: it is the accepted/consumed family that goes to
    /// zero, not "every counter". Calling `evaluate_conditioned()` on that same program
    /// RAISES rather than returning an unconditioned result, so this method is the
    /// non-raising way to probe for the case first.
    pub fn epistemic_evidence(&self, py: Python<'_>) -> PyResult<EpistemicEvidence> {
        let program = self.program.clone();
        let provider = self.provider.clone();
        let inputs = HashMap::new();
        let result = py
            .detach(move || program.execute_epistemic_evidence(provider, inputs))
            .map_err(types::xlog_err)?;
        let epistemic_mode = match result.prepared.preflight.epistemic_mode {
            xlog_ir::EirEpistemicMode::G91 => "g91",
            xlog_ir::EirEpistemicMode::Faeel => "faeel",
        }
        .to_string();
        Ok(EpistemicEvidence {
            epistemic_mode,
            know_operator_count: result.prepared.preflight.know_operator_count,
            possible_operator_count: result.prepared.preflight.possible_operator_count,
            accepted_candidates: result.semantic_trace.accepted_candidates,
            rejected_candidates: result.semantic_trace.rejected_candidates,
            accepted_world_views: result.semantic_trace.accepted_world_views,
            final_output_rows: result.final_result_transfer.final_output_rows,
        })
    }
}

#[pymethods]
impl CompiledConditionedProgram {
    /// Evaluate the prepared conditioned circuit without compiling source or program.
    #[cfg(feature = "host-io")]
    pub fn evaluate(&self, py: Python<'_>) -> PyResult<EpistemicEvalResult> {
        let program = self.program.clone();
        let result_provider = self.result_provider.clone();
        let prepared = py
            .detach(move || {
                let (result, trace) = program.evaluate()?;
                prepare_epistemic_eval_result(&result_provider, result, trace)
            })
            .map_err(types::xlog_err)?;
        pack_epistemic_eval_result(py, prepared)
    }

    #[cfg(not(feature = "host-io"))]
    pub fn evaluate(&self, _py: Python<'_>) -> PyResult<EpistemicEvalResult> {
        Err(types::host_io_disabled_pyerr())
    }

    /// Atomically replace independent probabilistic fact priors by CNF variable id.
    #[cfg(feature = "host-io")]
    pub fn set_fact_probabilities(
        &self,
        py: Python<'_>,
        updates: BTreeMap<u32, f64>,
    ) -> PyResult<()> {
        let program = self.program.clone();
        py.detach(move || program.set_fact_probabilities(&updates))
            .map_err(types::xlog_err)
    }

    #[cfg(not(feature = "host-io"))]
    pub fn set_fact_probabilities(
        &self,
        _py: Python<'_>,
        updates: BTreeMap<u32, f64>,
    ) -> PyResult<()> {
        let _ = updates;
        Err(types::host_io_disabled_pyerr())
    }

    /// Describe the current probability assigned to each CNF variable.
    #[cfg(feature = "host-io")]
    pub fn prob_var_map(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        use xlog_prob::exact::ProbVarInfo;

        let program = self.program.clone();
        let entries = py
            .detach(move || program.prob_var_map())
            .map_err(types::xlog_err)?;
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let dict = PyDict::new(py);
            match entry {
                ProbVarInfo::Fact { atom, prob } => {
                    dict.set_item("kind", "fact")?;
                    dict.set_item("atom", atom_to_string(&atom))?;
                    dict.set_item("prob", prob)?;
                }
                ProbVarInfo::Choice {
                    choices,
                    choice_index,
                    prob,
                } => {
                    dict.set_item("kind", "choice")?;
                    dict.set_item(
                        "atoms",
                        choices
                            .iter()
                            .map(|(atom, _)| atom_to_string(atom))
                            .collect::<Vec<_>>(),
                    )?;
                    dict.set_item(
                        "probs",
                        choices.iter().map(|(_, prob)| *prob).collect::<Vec<_>>(),
                    )?;
                    dict.set_item("choice_index", choice_index)?;
                    dict.set_item("prob", prob)?;
                }
                ProbVarInfo::Other => dict.set_item("kind", "other")?,
            }
            out.push(dict.into());
        }
        Ok(out)
    }

    #[cfg(not(feature = "host-io"))]
    pub fn prob_var_map(&self, _py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        Err(types::host_io_disabled_pyerr())
    }
}

#[cfg(feature = "host-io")]
struct PreparedEpistemicEvalResult {
    atoms: Vec<String>,
    prob_tensor: DlpackManagedTensor,
    log_prob_tensor: DlpackManagedTensor,
    log_z_e: f64,
    trace: EpistemicProbProductionTrace,
}

#[cfg(feature = "host-io")]
fn prepare_epistemic_eval_result(
    provider: &Arc<CudaKernelProvider>,
    result: ExactResult,
    trace: EpistemicProbProductionTrace,
) -> xlog_core::Result<PreparedEpistemicEvalResult> {
    let mut atoms: Vec<String> = Vec::with_capacity(result.query_probs.len());
    let mut probs: Vec<f64> = Vec::with_capacity(result.query_probs.len());
    let mut log_probs: Vec<f64> = Vec::with_capacity(result.query_probs.len());

    for q in result.query_probs {
        atoms.push(atom_to_string(&q.atom));
        probs.push(q.prob);
        log_probs.push(q.log_prob);
    }

    let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);
    let prob_buf = provider.create_buffer_from_slice::<f64>(&probs, schema.clone())?;
    let log_prob_buf = provider.create_buffer_from_slice::<f64>(&log_probs, schema)?;
    let prob_tensor = provider.to_dlpack_table(prob_buf).column(0)?;
    let log_prob_tensor = provider.to_dlpack_table(log_prob_buf).column(0)?;

    Ok(PreparedEpistemicEvalResult {
        atoms,
        prob_tensor,
        log_prob_tensor,
        log_z_e: result.log_z_e,
        trace,
    })
}

#[cfg(feature = "host-io")]
fn pack_epistemic_eval_result(
    py: Python<'_>,
    prepared: PreparedEpistemicEvalResult,
) -> PyResult<EpistemicEvalResult> {
    let PreparedEpistemicEvalResult {
        atoms,
        prob_tensor,
        log_prob_tensor,
        log_z_e,
        trace,
    } = prepared;

    let dict = PyDict::new(py);
    dict.set_item(
        "accepted_world_view_evidence_consumed",
        trace.accepted_world_view_evidence_consumed,
    )?;
    dict.set_item(
        "accepted_faeel_world_view_evidence_consumed",
        trace.accepted_faeel_world_view_evidence_consumed,
    )?;
    // Both modes reach this surface: a non-recursive `#pragma epistemic_mode = g91`
    // program conditions through the G91 counter with its FAEEL twin at 0. Exposing
    // only one of the pair would leave the trace unable to prove which mode ran.
    dict.set_item(
        "accepted_g91_world_view_evidence_consumed",
        trace.accepted_g91_world_view_evidence_consumed,
    )?;
    dict.set_item(
        "accepted_evidence_assumptions_consumed",
        trace.accepted_evidence_assumptions_consumed,
    )?;
    dict.set_item(
        "gpu_conditioned_evidence_facts",
        trace.gpu_conditioned_evidence_facts,
    )?;
    // The full evidence-class family. `gpu_conditioned_evidence_facts` above is the
    // total the engine's own `require_conditioned_evidence_trace` validates; the four
    // classes below decompose it. A `possible`-only or negated-evidence program
    // conditions with `gpu_conditioned_know_evidence_facts == 0`, so a caller checking
    // only the `know` class would misread a correct run as unconditioned.
    dict.set_item(
        "gpu_conditioned_know_evidence_facts",
        trace.gpu_conditioned_know_evidence_facts,
    )?;
    dict.set_item(
        "gpu_conditioned_possible_evidence_facts",
        trace.gpu_conditioned_possible_evidence_facts,
    )?;
    dict.set_item(
        "gpu_conditioned_not_known_evidence_facts",
        trace.gpu_conditioned_not_known_evidence_facts,
    )?;
    dict.set_item(
        "gpu_conditioned_not_possible_evidence_facts",
        trace.gpu_conditioned_not_possible_evidence_facts,
    )?;
    dict.set_item(
        "gpu_exact_query_evaluations",
        trace.gpu_exact_query_evaluations,
    )?;
    dict.set_item("gpu_exact_source_compiles", trace.gpu_exact_source_compiles)?;
    dict.set_item(
        "gpu_exact_program_compiles",
        trace.gpu_exact_program_compiles,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_reuses",
        trace.gpu_conditioned_circuit_reuses,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_preparation_compiles",
        trace.gpu_conditioned_circuit_preparation_compiles,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_materializations",
        trace.gpu_conditioned_circuit_materializations,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_disk_cache_restores",
        trace.gpu_conditioned_circuit_disk_cache_restores,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_gpu_cache_hits",
        trace.gpu_conditioned_circuit_gpu_cache_hits,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_generation",
        trace.gpu_conditioned_circuit_generation,
    )?;
    dict.set_item(
        "gpu_conditioned_circuit_cache_slot",
        trace.gpu_conditioned_circuit_cache_slot,
    )?;
    dict.set_item(
        "gpu_knowledge_compilation_end_to_end_runs",
        trace.gpu_knowledge_compilation_end_to_end_runs,
    )?;
    dict.set_item(
        "accepted_gpu_production_path_events",
        trace.accepted_gpu_production_path_events,
    )?;

    Ok(EpistemicEvalResult {
        atoms,
        prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
        log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
        log_z_e,
        trace: dict.into(),
    })
}
