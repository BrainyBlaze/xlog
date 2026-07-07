// Remaining CompiledProgram methods and internal types.
//
// Contains: evaluate/evaluate_device, NLL loss helpers, training control
// methods (zero_grad, optimizer_step, etc.), pack_result helpers, and the
// internal types used by CompiledProgram's implementation (CachedCircuit,
// QuerySignature, InputSource, NeuralGroup, CompiledProbProgram).
//
// The #[pyclass] struct definitions remain in lib.rs.

use cudarc::driver::DeviceSlice;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use xlog_core::{ScalarType, Schema};
use xlog_logic::ast::Term;
use xlog_prob::exact::ExactDdnnfProgram;
#[cfg(feature = "host-io")]
use xlog_prob::exact::{ExactResultWithGrads, QueryProbability};
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};
use xlog_prob::neural_fast_path::GpuWeightSlots;

use super::neural_registry::NeuralPredicateInfo;
use super::{
    dlpack_capsule_from_tensor, enforce_call_memory_limit, provider_memory_stats, types,
    CompiledProgram, EpochStats, EvalResult, McDeviceEvalResult, TrainingHistory,
};

// =========================================================================
// Internal types
// =========================================================================

/// A cached circuit for a specific query template.
///
/// The circuit structure is immutable - only weights change between queries.
/// Weight slots map network outputs to circuit variables.
pub(crate) struct CachedCircuit {
    /// The compiled program containing the GPU circuit
    pub(crate) program: ExactDdnnfProgram,

    /// Device-resident mapping from neural output slots to CNF variable ids.
    pub(crate) slots: GpuWeightSlots,

    /// Ordered target domain for Targeted signatures. Empty for Boolean.
    pub(crate) target_domain: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum InputSource {
    QueryArg(usize),
    ImplicitSlot(usize),
    /// Stage-B real-domain grounding: read row `usize` of the join-domain tensor
    /// source (`nsr_domain`, the per-event feature batch). Used by the per-event
    /// expansions of a neural predicate joined on an existential variable.
    DomainRow(usize),
    /// Stage-B real-domain grounding: an input-independent group (the rule-weight
    /// guard expanded per head-key constant). The forward feeds a dummy row of the
    /// declared input width; only the network's parameters (not the input) matter.
    ConstDummy,
}

#[derive(Debug, Clone)]
pub(crate) struct NeuralGroup {
    pub(crate) info: NeuralPredicateInfo,
    pub(crate) input_source: InputSource,
    /// Stage-B real-domain grounding: when `Some(c)`, the template grounds this
    /// group's neural atom at the real constant `c` (a domain event id or a
    /// head-key edge id) instead of a synthetic placeholder, so the one neural
    /// occurrence expands into one circuit leaf per domain constant.
    pub(crate) ground_const: Option<Term>,
    #[cfg(feature = "host-io")]
    pub(crate) output_var: Option<String>,
}

/// An ordinary-relation body atom in a trainable-rule query treated as a HARD
/// join condition: it gates which query groundings can fire but contributes no
/// probability mass and no gradient. The query probability is
/// `(hard conditions satisfiable?) x (neural prob)`; gradients flow only
/// through the neural predicates x sigma(w), never through these fact atoms.
#[derive(Debug, Clone)]
pub(crate) struct HardFilter {
    /// Ordinary relation name (must hold over the program's facts).
    pub(crate) relation: String,
    /// For each relation argument position, the query HEAD position whose value
    /// it must equal. Current scope: every relation argument is a head
    /// variable; hard conditions that join on existential (non-head) variables
    /// are a documented follow-up.
    pub(crate) arg_head_positions: Vec<usize>,
}

/// Stage-B existential join: the plan for grounding a neural predicate over the
/// REAL join domain inside the circuit (instead of stripping the join relation as
/// a pre-filter). The neural groups are already expanded per domain constant; this
/// records the ordinary relations whose ground facts must stay IN the circuit (so
/// provenance OR-aggregates `OR_event(neural(event) ∧ join(event, head))` at each
/// head binding) and the head-key domain that the query ranges over.
#[derive(Debug, Clone)]
pub(crate) struct JoinPlan {
    /// Ordinary relations kept inside the circuit rule; their ground facts are
    /// added to the template program (read from `self.ast`).
    pub(crate) relations: Vec<String>,
    /// The real head-key domain the query ranges over (e.g. edge ids), in the
    /// same sorted order as the per-edge guard group expansion and the emitted
    /// `prob_queries`. Serves as the `target_domain` for a join signature.
    pub(crate) head_domain: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum QuerySignature {
    Boolean {
        groups: Vec<NeuralGroup>,
        hard_filters: Vec<HardFilter>,
    },
    Targeted {
        target_position: usize,
        groups: Vec<NeuralGroup>,
        hard_filters: Vec<HardFilter>,
        /// `Some` for a Stage-B existential-join signature: the head target
        /// position ranges over the real head-key domain (e.g. edges) and the
        /// groups are real-domain-grounded. `None` for the ordinary targeted path.
        join: Option<JoinPlan>,
    },
}

impl QuerySignature {
    pub(crate) fn groups(&self) -> &[NeuralGroup] {
        match self {
            QuerySignature::Boolean { groups, .. } | QuerySignature::Targeted { groups, .. } => {
                groups
            }
        }
    }

    pub(crate) fn hard_filters(&self) -> &[HardFilter] {
        match self {
            QuerySignature::Boolean { hard_filters, .. }
            | QuerySignature::Targeted { hard_filters, .. } => hard_filters,
        }
    }

    /// The Stage-B join plan, if this is an existential-join signature.
    pub(crate) fn join(&self) -> Option<&JoinPlan> {
        match self {
            QuerySignature::Targeted { join, .. } => join.as_ref(),
            QuerySignature::Boolean { .. } => None,
        }
    }
}

pub(crate) enum CompiledProbProgram {
    Exact(ExactDdnnfProgram),
    Mc(McProgram),
}

impl CompiledProbProgram {
    #[cfg(feature = "host-io")]
    pub(crate) fn num_vars(&self) -> usize {
        match self {
            Self::Exact(p) => p.num_vars(),
            Self::Mc(p) => p.num_vars(),
        }
    }
}

// =========================================================================
// Helper functions
// =========================================================================

#[cfg(feature = "host-io")]
fn atom_to_string(atom: &xlog_prob::provenance::GroundAtom) -> String {
    use xlog_prob::provenance::Value;

    if atom.args.is_empty() {
        return format!("{}()", atom.predicate);
    }

    let mut s = String::new();
    s.push_str(&atom.predicate);
    s.push('(');
    for (i, arg) in atom.args.iter().enumerate() {
        if i != 0 {
            s.push_str(", ");
        }
        match arg {
            Value::I64(v) => s.push_str(&v.to_string()),
            Value::F64(bits) => s.push_str(&f64::from_bits(*bits).to_string()),
            Value::Symbol(sym) => s.push_str(&format!("sym#{}", sym)),
            Value::String(v) => s.push_str(v),
        }
    }
    s.push(')');
    s
}

// =========================================================================
// impl CompiledProgram — private helpers
// =========================================================================

impl CompiledProgram {
    pub(crate) fn parse_sampling_method(s: Option<String>) -> PyResult<Option<McSamplingMethod>> {
        match s.as_deref() {
            None => Ok(None),
            Some("rejection") => Ok(Some(McSamplingMethod::Rejection)),
            Some("evidence_clamping") => Ok(Some(McSamplingMethod::EvidenceClamping)),
            Some(other) => Err(PyValueError::new_err(format!(
                "Unknown sampling_method '{}'. Use 'rejection' or 'evidence_clamping'.",
                other
            ))),
        }
    }

    /// Evaluate probability of a single query by compiling a temporary program.
    pub(crate) fn evaluate_query_probability(&self, query: &str) -> PyResult<f64> {
        let probs = self.evaluate_query_probabilities(&[query.to_string()])?;
        probs
            .into_iter()
            .next()
            .ok_or_else(|| PyRuntimeError::new_err("Query evaluation returned no results"))
    }

    /// Evaluate probabilities for multiple queries by compiling a temporary program.
    pub(crate) fn evaluate_query_probabilities(&self, queries: &[String]) -> PyResult<Vec<f64>> {
        #[cfg(not(feature = "host-io"))]
        {
            let _ = queries;
            return Err(types::host_io_disabled_pyerr());
        }

        #[cfg(feature = "host-io")]
        {
            // Build source with queries appended
            let mut source_with_queries = self._source.clone();
            for query in queries {
                source_with_queries.push_str(&format!("\nquery({}).", query));
            }

            // Compile and evaluate the temporary program
            let result: Vec<QueryProbability> = match self._prob_engine {
                xlog_logic::ast::ProbEngine::ExactDdnnf => {
                    let program = ExactDdnnfProgram::compile_source_with_gpu(
                        &source_with_queries,
                        self._gpu_config,
                    )
                    .map_err(|e| types::gpu_err("Query compilation error", e))?;

                    program
                        .evaluate()
                        .map_err(|e| types::gpu_err("Query evaluation error", e))?
                        .query_probs
                }
                xlog_logic::ast::ProbEngine::Mc => {
                    let program =
                        McProgram::compile_source_with_gpu(&source_with_queries, self._gpu_config)
                            .map_err(|e| types::gpu_err("Query compilation error", e))?;

                    let cfg = McEvalConfig::default();
                    program
                        .evaluate(cfg)
                        .map_err(|e| types::gpu_err("Query evaluation error", e))?
                        .query_estimates
                        .into_iter()
                        .map(|e| QueryProbability {
                            atom: e.atom,
                            prob: e.prob,
                            log_prob: e.log_prob,
                        })
                        .collect()
                }
            };

            // Extract probabilities in query order
            // The results should be in the same order as queries were added
            let probs: Vec<f64> = result.iter().map(|qp| qp.prob).collect();

            if probs.len() != queries.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "Expected {} query results, got {}",
                    queries.len(),
                    probs.len()
                )));
            }

            Ok(probs)
        }
    }

    #[cfg(feature = "host-io")]
    fn pack_result_probs(
        &self,
        py: Python<'_>,
        query_probs: Vec<QueryProbability>,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(query_probs.len());
        let mut probs: Vec<f64> = Vec::with_capacity(query_probs.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(query_probs.len());

        for q in query_probs {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);
        }

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);
        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars: self.program.num_vars(),
            grad_true: None,
            grad_false: None,
            approx: false,
            stderr: None,
            ci_low: None,
            ci_high: None,
            samples: None,
            evidence_samples: None,
            seed: None,
            confidence: None,
            nonmonotone_semantics: None,
            nonmonotone_sccs: None,
            nonmonotone_cycles: None,
            nonmonotone_iteration_limit_hits: None,
            sampling_method: None,
            mc_engine: None,
        })
    }

    #[cfg(feature = "host-io")]
    fn pack_result_with_grads(
        &self,
        py: Python<'_>,
        result: ExactResultWithGrads,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(result.query_grads.len());
        let mut probs: Vec<f64> = Vec::with_capacity(result.query_grads.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(result.query_grads.len());

        let mut grad_true_caps: Vec<PyObject> = Vec::with_capacity(result.query_grads.len());
        let mut grad_false_caps: Vec<PyObject> = Vec::with_capacity(result.query_grads.len());

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);

        let num_vars = self.program.num_vars();
        for q in result.query_grads {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);

            let grad_true_buf = self
                .output_provider
                .create_buffer_from_slice::<f64>(&q.grad_true, schema.clone())
                .map_err(types::xlog_err)?;
            let grad_false_buf = self
                .output_provider
                .create_buffer_from_slice::<f64>(&q.grad_false, schema.clone())
                .map_err(types::xlog_err)?;

            let grad_true_tensor = self
                .output_provider
                .to_dlpack_table(grad_true_buf)
                .column(0)
                .map_err(types::xlog_err)?;
            let grad_false_tensor = self
                .output_provider
                .to_dlpack_table(grad_false_buf)
                .column(0)
                .map_err(types::xlog_err)?;

            grad_true_caps.push(dlpack_capsule_from_tensor(py, grad_true_tensor)?);
            grad_false_caps.push(dlpack_capsule_from_tensor(py, grad_false_tensor)?);
        }

        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars,
            grad_true: Some(grad_true_caps),
            grad_false: Some(grad_false_caps),
            approx: false,
            stderr: None,
            ci_low: None,
            ci_high: None,
            samples: None,
            evidence_samples: None,
            seed: None,
            confidence: None,
            nonmonotone_semantics: None,
            nonmonotone_sccs: None,
            nonmonotone_cycles: None,
            nonmonotone_iteration_limit_hits: None,
            sampling_method: None,
            mc_engine: None,
        })
    }

    #[cfg(feature = "host-io")]
    fn pack_result_mc(
        &self,
        py: Python<'_>,
        result: xlog_prob::mc::McResult,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(result.query_estimates.len());
        let mut probs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut stderrs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut ci_lows: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut ci_highs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());

        for q in &result.query_estimates {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);
            stderrs.push(q.stderr);
            ci_lows.push(q.ci_low);
            ci_highs.push(q.ci_high);
        }

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);
        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema.clone())
            .map_err(types::xlog_err)?;
        let stderr_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&stderrs, schema.clone())
            .map_err(types::xlog_err)?;
        let ci_low_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&ci_lows, schema.clone())
            .map_err(types::xlog_err)?;
        let ci_high_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&ci_highs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let stderr_tensor = self
            .output_provider
            .to_dlpack_table(stderr_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let ci_low_tensor = self
            .output_provider
            .to_dlpack_table(ci_low_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let ci_high_tensor = self
            .output_provider
            .to_dlpack_table(ci_high_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars: self.program.num_vars(),
            grad_true: None,
            grad_false: None,
            approx: true,
            stderr: Some(dlpack_capsule_from_tensor(py, stderr_tensor)?),
            ci_low: Some(dlpack_capsule_from_tensor(py, ci_low_tensor)?),
            ci_high: Some(dlpack_capsule_from_tensor(py, ci_high_tensor)?),
            samples: Some(result.total_samples),
            evidence_samples: Some(result.evidence_samples),
            seed: Some(result.seed),
            confidence: Some(result.confidence),
            nonmonotone_semantics: Some(xlog_prob::mc::NONMONOTONE_SEMANTICS.to_string()),
            nonmonotone_sccs: Some(result.nonmonotone_sccs),
            nonmonotone_cycles: Some(result.nonmonotone_cycles),
            nonmonotone_iteration_limit_hits: Some(result.nonmonotone_iteration_limit_hits),
            sampling_method: Some(match result.sampling_method {
                McSamplingMethod::Rejection => "rejection".to_string(),
                McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
            }),
            mc_engine: Some(result.engine.as_str().to_string()),
        })
    }
}

// =========================================================================
// #[pymethods] impl CompiledProgram — evaluate, NLL, training controls
// =========================================================================

#[pymethods]
impl CompiledProgram {
    #[pyo3(signature = (return_grads=false, samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None, memory_mb=None, allow_cpu_oracle=false))]
    pub fn evaluate(
        &self,
        _py: Python<'_>,
        return_grads: bool,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
        sampling_method: Option<String>,
        memory_mb: Option<u64>,
        allow_cpu_oracle: bool,
    ) -> PyResult<EvalResult> {
        enforce_call_memory_limit(&self.output_provider, memory_mb)?;
        match &self.program {
            CompiledProbProgram::Exact(_program) => {
                if samples.is_some() || seed.is_some() {
                    return Err(PyValueError::new_err(
                        "samples/seed are only supported for prob_engine='mc'",
                    ));
                }
                #[cfg(feature = "host-io")]
                {
                    if return_grads {
                        let result = _program
                            .evaluate_gpu_with_grads()
                            .map_err(types::xlog_err)?;
                        self.pack_result_with_grads(_py, result)
                    } else {
                        let result = _program.evaluate().map_err(types::xlog_err)?;
                        self.pack_result_probs(_py, result.query_probs)
                    }
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = return_grads;
                    Err(types::host_io_disabled_pyerr())
                }
            }
            CompiledProbProgram::Mc(_program) => {
                if return_grads {
                    return Err(PyValueError::new_err(
                        "MC inference does not support gradients (return_grads must be false)",
                    ));
                }

                let mut cfg = McEvalConfig::default();
                cfg.samples = samples.unwrap_or(10000);
                cfg.seed = seed.unwrap_or(0);
                cfg.confidence = confidence;
                cfg.max_nonmonotone_iterations = max_nonmonotone_iterations;
                cfg.sampling_method = Self::parse_sampling_method(sampling_method)?;
                // Fail-closed contract: resident-rejected programs (negation,
                // aggregates, ...) error unless the caller explicitly opts
                // into the labeled CPU oracle.
                cfg.allow_cpu_oracle_fallback = allow_cpu_oracle;
                #[cfg(feature = "host-io")]
                {
                    let result = _program.evaluate(cfg).map_err(types::xlog_err)?;
                    self.pack_result_mc(_py, result)
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = cfg;
                    Err(types::host_io_disabled_pyerr())
                }
            }
        }
    }

    /// Evaluate Monte Carlo programs and return device-only result counts via DLPack.
    ///
    /// This is the primary GPU-native API surface for MC inference. It never performs
    /// device->host reads for result data (only returns device buffers).
    #[pyo3(signature = (samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None, memory_mb=None))]
    pub fn evaluate_device(
        &self,
        py: Python<'_>,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
        sampling_method: Option<String>,
        memory_mb: Option<u64>,
    ) -> PyResult<McDeviceEvalResult> {
        enforce_call_memory_limit(&self.output_provider, memory_mb)?;
        let (
            query_counts,
            evidence_count,
            total_samples,
            seed,
            confidence,
            nonmonotone_sccs,
            nonmonotone_cycles,
            nonmonotone_iteration_limit_hits,
            sampling_method_val,
            no_host,
        ) = match &self.program {
            CompiledProbProgram::Mc(program) => {
                let mut cfg = McEvalConfig::default();
                cfg.samples = samples.unwrap_or(10000);
                cfg.seed = seed.unwrap_or(0);
                cfg.confidence = confidence;
                cfg.max_nonmonotone_iterations = max_nonmonotone_iterations;
                cfg.sampling_method = Self::parse_sampling_method(sampling_method)?;

                let result = program
                    .evaluate_gpu_device_with_provider(cfg, self.output_provider.clone())
                    .map_err(types::xlog_err)?;

                (
                    result.query_counts,
                    result.evidence_count,
                    result.total_samples,
                    result.seed,
                    result.confidence,
                    result.nonmonotone_sccs,
                    result.nonmonotone_cycles,
                    result.nonmonotone_iteration_limit_hits,
                    result.sampling_method,
                    result.no_host,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "evaluate_device is only supported for prob_engine='mc'",
                ))
            }
        };

        // PyTorch does not support unsigned 32-bit types. Export as i32 (bitwise identical for
        // counts < 2^31) for maximum DLPack consumer compatibility.
        let schema_i32 = Schema::new(vec![("col0".to_string(), ScalarType::I32)]);

        let make_count_tensor =
            |counts: xlog_cuda::memory::TrackedCudaSlice<u32>, rows: u64| -> PyResult<PyObject> {
                let rows_u32 = u32::try_from(rows).map_err(|_| {
                    PyValueError::new_err(format!("Row count {} exceeds u32::MAX", rows))
                })?;

                let mut d_num_rows = self
                    .output_provider
                    .memory()
                    .alloc::<u32>(1)
                    .map_err(types::xlog_err)?;
                self.output_provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
                    .map_err(types::xlog_err)?;

                let buffer = xlog_cuda::CudaBuffer::from_columns(
                    vec![counts.into_bytes().into()],
                    rows,
                    d_num_rows,
                    schema_i32.clone(),
                );
                let tensor = self
                    .output_provider
                    .to_dlpack_table(buffer)
                    .column(0)
                    .map_err(types::xlog_err)?;
                dlpack_capsule_from_tensor(py, tensor)
            };

        let query_rows = u64::try_from(query_counts.len())
            .map_err(|_| PyValueError::new_err("query_counts length overflow"))?;
        let query_counts_capsule = make_count_tensor(query_counts, query_rows)?;
        let evidence_count_capsule = make_count_tensor(evidence_count, 1)?;
        let resident_no_host_certified = no_host.is_no_host();

        Ok(McDeviceEvalResult {
            query_counts: query_counts_capsule,
            evidence_count: evidence_count_capsule,
            total_samples,
            seed,
            confidence,
            nonmonotone_semantics: xlog_prob::mc::NONMONOTONE_SEMANTICS.to_string(),
            nonmonotone_sccs,
            nonmonotone_cycles,
            nonmonotone_iteration_limit_hits,
            sampling_method: match sampling_method_val {
                McSamplingMethod::Rejection => "rejection".to_string(),
                McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
            },
            resident_no_host_certified,
            resident_no_host_policy_result: if resident_no_host_certified {
                "certified".to_string()
            } else {
                "failed".to_string()
            },
            resident_no_host_tracked_dtoh_calls: no_host.tracked_dtoh_calls,
            resident_no_host_tracked_htod_calls: no_host.tracked_htod_calls,
            resident_no_host_host_loop_iterations: no_host.host_loop_iterations,
            resident_no_host_per_sample_host_launches: no_host.per_sample_host_launches,
            resident_no_host_untracked_metadata_reads: no_host.untracked_metadata_reads,
            resident_no_host_engine_launches: no_host.engine_launches,
            resident_no_host_host_fixpoint_iterations: no_host.host_fixpoint_iterations,
            resident_no_host_per_operator_host_allocations: no_host.per_operator_host_allocations,
        })
    }

    // =========================================================================
    // NLL Loss Functions
    // =========================================================================

    /// Compute negative log-likelihood loss for a single query.
    ///
    /// NLL loss = -log(P(query))
    ///
    /// This is the fundamental training objective for neural-symbolic programs.
    /// Lower loss means higher probability of the query being true.
    ///
    /// # Arguments
    /// * `query` - Query atom as string, e.g., "digit(0, 5)" or "path(1, 3)"
    ///
    /// # Returns
    /// The NLL loss value (always non-negative, 0 for certain facts)
    fn nll_loss(&self, query: &str) -> PyResult<f64> {
        let prob = self.evaluate_query_probability(query)?;
        Ok(types::nll_loss_value(prob))
    }

    /// Compute sum of NLL losses for a batch of queries.
    ///
    /// Batch loss = Σ -log(P(query_i))
    ///
    /// More efficient than calling nll_loss repeatedly as all queries
    /// are compiled and evaluated together.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings
    ///
    /// # Returns
    /// Sum of individual NLL losses (0.0 for empty batch)
    fn nll_loss_batch(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Ok(0.0);
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        Ok(probs.iter().map(|&p| types::nll_loss_value(p)).sum())
    }

    /// Compute mean NLL loss for a batch of queries.
    ///
    /// Mean loss = (1/n) Σ -log(P(query_i))
    ///
    /// Useful for comparing loss across batches of different sizes.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings (must be non-empty)
    ///
    /// # Returns
    /// Mean of individual NLL losses
    ///
    /// # Errors
    /// Returns error if queries is empty
    fn nll_loss_mean(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot compute mean NLL loss for empty query batch",
            ));
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        let sum: f64 = probs.iter().map(|&p| types::nll_loss_value(p)).sum();
        Ok(sum / probs.len() as f64)
    }

    /// Compute NLL loss and return as PyTorch tensor.
    ///
    /// Returns a scalar tensor that can participate in autograd.
    /// Use this when you need gradients to flow back through the loss.
    ///
    /// # Arguments
    /// * `query` - Query atom as string
    ///
    /// # Returns
    /// PyTorch scalar tensor containing the loss value
    fn nll_loss_tensor(&self, py: Python<'_>, query: &str) -> PyResult<PyObject> {
        let loss = self.nll_loss(query)?;
        types::create_torch_tensor(py, loss)
    }

    /// Compute batch NLL loss and return as PyTorch tensor.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings
    ///
    /// # Returns
    /// PyTorch scalar tensor containing the sum of losses
    fn nll_loss_batch_tensor(&self, py: Python<'_>, queries: Vec<String>) -> PyResult<PyObject> {
        let loss = self.nll_loss_batch(queries)?;
        types::create_torch_tensor(py, loss)
    }

    // =========================================================================
    // Backward Pass / Training Methods
    // =========================================================================

    /// Zero gradients for all registered networks.
    ///
    /// This should be called at the start of each training iteration
    /// to clear accumulated gradients from previous iterations.
    pub fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(optimizer) = handle.optimizer() {
                    optimizer.call_method0(py, "zero_grad")?;
                }
            }
        }
        Ok(())
    }

    /// Perform optimizer step for all registered networks.
    ///
    /// This applies the accumulated gradients to update network parameters.
    /// Should be called after forward_backward().
    pub fn optimizer_step(&self, py: Python<'_>) -> PyResult<()> {
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(optimizer) = handle.optimizer() {
                    optimizer.call_method0(py, "step")?;
                }
            }
        }
        Ok(())
    }

    /// Clip gradient norms for all registered networks.
    ///
    /// Uses `torch.nn.utils.clip_grad_norm_`.
    pub fn clip_grad_norms(&self, py: Python<'_>, max_norm: f64) -> PyResult<()> {
        let clip_fn = py.import("torch.nn.utils")?.getattr("clip_grad_norm_")?;
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(module) = handle.module() {
                    let params = module.call_method0(py, "parameters")?;
                    clip_fn.call1((params, max_norm))?;
                }
            }
        }
        Ok(())
    }

    /// Step the learning rate scheduler.
    ///
    /// PyTorch schedulers expect at least one optimizer step before the first
    /// scheduler step. Call this after `optimizer_step()` (or after a training
    /// path that performs an optimizer step internally).
    ///
    /// If `network_name` is provided, steps only that network's scheduler.
    /// If `None` (default), steps all registered schedulers.
    #[pyo3(signature = (network_name=None))]
    fn scheduler_step(&self, py: Python<'_>, network_name: Option<&str>) -> PyResult<()> {
        match network_name {
            Some(name) => {
                let handle = self.network_registry.get(name).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "No network registered with name '{name}'"
                    ))
                })?;
                if let Some(scheduler) = handle.scheduler() {
                    scheduler.call_method0(py, "step")?;
                }
            }
            None => {
                for name in self.network_registry.names() {
                    if let Some(handle) = self.network_registry.get(name) {
                        if let Some(scheduler) = handle.scheduler() {
                            scheduler.call_method0(py, "step")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the current learning rate for a registered network.
    ///
    /// Reads `optimizer.param_groups[0]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    fn get_lr(&self, py: Python<'_>, network_name: &str) -> PyResult<f64> {
        let handle = self.network_registry.get(network_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "No network registered with name '{network_name}'"
            ))
        })?;
        let optimizer = handle.optimizer().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Network '{network_name}' has no optimizer"
            ))
        })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let group0 = param_groups.call_method1(py, "__getitem__", (0i32,))?;
        let lr = group0.call_method1(py, "__getitem__", ("lr",))?;
        lr.extract(py)
    }

    /// Set the learning rate for a registered network.
    ///
    /// Writes to all `optimizer.param_groups[i]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    /// * `lr` - New learning rate value
    fn set_lr(&self, py: Python<'_>, network_name: &str, lr: f64) -> PyResult<()> {
        let handle = self.network_registry.get(network_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "No network registered with name '{network_name}'"
            ))
        })?;
        let optimizer = handle.optimizer().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Network '{network_name}' has no optimizer"
            ))
        })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let num_groups: usize = param_groups.call_method0(py, "__len__")?.extract(py)?;
        for i in 0..num_groups {
            let group = param_groups.call_method1(py, "__getitem__", (i as i32,))?;
            group.call_method(py, "__setitem__", ("lr", lr), None)?;
        }
        Ok(())
    }

    // =========================================================================
    // Training Methods
    // =========================================================================

    /// Train for one epoch over the given queries.
    ///
    /// This method:
    /// 1. Processes queries in batches
    /// 2. For each batch: zero_grad, forward_backward for each query, optimizer_step
    /// 3. Returns statistics for the epoch
    ///
    /// # Arguments
    /// * `queries` - List of query strings to train on
    /// * `batch_size` - Number of queries per batch (default: 32)
    ///
    /// # Returns
    /// EpochStats with avg_loss, num_batches, total_queries
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_internal(
            py,
            &queries,
            batch_size,
            usize::MAX,
            max_grad_norm,
            &mut history,
        )
    }

    /// Evaluate mean NLL loss over queries without updating parameters.
    ///
    /// Useful for validation/test set evaluation.
    ///
    /// # Arguments
    /// * `queries` - List of query strings to evaluate
    ///
    /// # Returns
    /// Mean NLL loss over all queries
    pub fn evaluate_loss(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Ok(0.0);
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        let total_loss: f64 = probs.iter().map(|&p| types::nll_loss_value(p)).sum();
        Ok(total_loss / queries.len() as f64)
    }

    /// Train for one epoch with GPU-native loss accumulation (no per-query .item()).
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch_tensor(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_tensor_internal(
            py,
            &queries,
            batch_size,
            usize::MAX,
            max_grad_norm,
            &mut history,
        )
    }

    /// Return warmup profiling data as a Python dict (or None if profiling disabled).
    ///
    /// When XLOG_WARMUP_PROFILE=1, returns a dict with:
    ///   - "ptx": PTX load timing breakdown
    ///   - "circuit": circuit compilation timing breakdown
    /// Returns None if profiling is not enabled or no data is available.
    fn warmup_breakdown(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let ptx_profile = self.output_provider.ptx_load_profile();
        // The neural forward path records `self.last_compile_profile`. For a
        // purely probabilistic/deterministic program no neural forward runs,
        // so fall back to the compile profile captured when the program's own
        // circuit was built (the cold D4-compile + CDCL-verify). Without this
        // fallback `warmup_breakdown()` returned None for non-neural programs,
        // hiding the verification-overhead split (EIC W5).
        let circuit_profile = self.last_compile_profile.as_ref().or_else(|| match &self.program {
            CompiledProbProgram::Exact(p) => p.last_compile_profile(),
            CompiledProbProgram::Mc(_) => None,
        });

        // Return None if neither profile is available.
        if ptx_profile.is_none() && circuit_profile.is_none() {
            return Ok(None);
        }

        let result = PyDict::new(py);

        if let Some(ptx) = ptx_profile {
            let ptx_dict = PyDict::new(py);
            ptx_dict.set_item("total_sec", ptx.total_sec)?;
            ptx_dict.set_item("cubin_loaded", ptx.cubin_loaded)?;
            ptx_dict.set_item("ptx_fallback", ptx.ptx_fallback)?;
            let per_module = PyDict::new(py);
            for (name, sec) in &ptx.per_module_sec {
                per_module.set_item(name, *sec)?;
            }
            ptx_dict.set_item("per_module_sec", per_module)?;
            result.set_item("ptx", ptx_dict)?;
        }

        if let Some(circuit) = circuit_profile {
            let circuit_dict = PyDict::new(py);
            circuit_dict.set_item("gpu_cache_hit", circuit.gpu_cache_hit)?;
            circuit_dict.set_item("disk_cache_hit", circuit.disk_cache_hit)?;
            circuit_dict.set_item("d4_compile_sec", circuit.d4_compile_sec)?;
            circuit_dict.set_item("verify_sec", circuit.verify_sec)?;
            circuit_dict.set_item("smooth_sec", circuit.smooth_sec)?;
            circuit_dict.set_item("cache_store_sec", circuit.cache_store_sec)?;
            circuit_dict.set_item("free_var_mask_sec", circuit.free_var_mask_sec)?;
            circuit_dict.set_item("cnf_hash_sec", circuit.cnf_hash_sec)?;
            result.set_item("circuit", circuit_dict)?;
        }

        Ok(Some(result.into()))
    }

    /// Clear the circuit template cache, forcing recompilation on next query.
    /// Used for cache ablation benchmarks.
    fn clear_circuit_cache(&mut self) {
        self.circuit_cache.clear();
    }

    /// Return memory diagnostics including allocated_bytes and memory_limit_bytes.
    pub fn memory_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        provider_memory_stats(py, &self.output_provider)
    }

    pub fn rule_provenance(&self, py: Python<'_>) -> PyResult<PyObject> {
        let provenance = xlog_logic::rule_provenance(&self.ast, None);
        pack_rule_provenance(py, &provenance)
    }

    pub fn proof_traces(&self, py: Python<'_>) -> PyResult<PyObject> {
        let provenance = xlog_logic::rule_provenance(&self.ast, None);
        let traces = xlog_logic::query_proof_traces(&self.ast, &provenance);
        pack_proof_traces(py, &traces)
    }

    pub fn host_transfer_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.output_provider.host_transfer_stats();
        let dict = PyDict::new(py);
        dict.set_item("dtoh_bytes", stats.dtoh_bytes)?;
        dict.set_item("htod_bytes", stats.htod_bytes)?;
        dict.set_item("dtoh_calls", stats.dtoh_calls)?;
        dict.set_item("htod_calls", stats.htod_calls)?;
        Ok(dict.into())
    }

    pub fn reset_host_transfer_stats(&self) {
        self.output_provider.reset_host_transfer_stats()
    }

    pub fn neural_hot_loop_diagnostics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let transfers = self.output_provider.host_transfer_stats();
        let dict = PyDict::new(py);
        dict.set_item("post_load_dtoh_bytes", transfers.dtoh_bytes)?;
        dict.set_item("post_load_htod_bytes", transfers.htod_bytes)?;
        dict.set_item("post_load_dtoh_calls", transfers.dtoh_calls)?;
        dict.set_item("post_load_htod_calls", transfers.htod_calls)?;
        dict.set_item("control_plane_bytes_per_iteration", py.None())?;
        dict.set_item(
            "control_plane_status",
            "unavailable: per-iteration control-plane byte counter is not registered",
        )?;
        dict.set_item("scalar_sync_checks", py.None())?;
        dict.set_item(
            "scalar_sync_status",
            "unavailable: scalar synchronization counter is not registered",
        )?;

        let cuda_graph = PyDict::new(py);
        cuda_graph.set_item(
            "csm_cuda_graph_captures",
            self.output_provider.csm_cuda_graph_captures(),
        )?;
        cuda_graph.set_item(
            "csm_cuda_graph_launches",
            self.output_provider.csm_cuda_graph_launches(),
        )?;
        cuda_graph.set_item(
            "csm_cuda_graph_fallbacks",
            self.output_provider.csm_cuda_graph_fallbacks(),
        )?;
        cuda_graph.set_item(
            "csm_cuda_graph_cache_hits",
            self.output_provider.csm_cuda_graph_cache_hits(),
        )?;
        dict.set_item("cuda_graph", cuda_graph)?;

        let circuit_cache = PyDict::new(py);
        circuit_cache.set_item("circuit_cache_size", self.circuit_cache.len())?;
        circuit_cache.set_item("circuit_cache_hits", self.circuit_cache_hits)?;
        circuit_cache.set_item("circuit_cache_misses", self.circuit_cache_misses)?;
        circuit_cache.set_item("template_compile_count", self.template_compile_count)?;
        circuit_cache.set_item(
            "query_signature_cache_size",
            self.query_signature_cache.len(),
        )?;
        dict.set_item("circuit_cache", circuit_cache)?;

        Ok(dict.into())
    }

    pub fn cuda_graph_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item(
            "csm_cuda_graph_captures",
            self.output_provider.csm_cuda_graph_captures(),
        )?;
        dict.set_item(
            "csm_cuda_graph_launches",
            self.output_provider.csm_cuda_graph_launches(),
        )?;
        dict.set_item(
            "csm_cuda_graph_fallbacks",
            self.output_provider.csm_cuda_graph_fallbacks(),
        )?;
        dict.set_item(
            "csm_cuda_graph_cache_hits",
            self.output_provider.csm_cuda_graph_cache_hits(),
        )?;
        Ok(dict.into())
    }
}

fn pack_rule_provenance(
    py: Python<'_>,
    entries: &[xlog_logic::RuleProvenance],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("rule_id", &entry.rule_id)?;
        dict.set_item("head", &entry.head)?;
        dict.set_item("source_kind", entry.source_kind.as_str())?;
        dict.set_item("source_span", entry.source_span.clone())?;
        dict.set_item("generation_trace_hash", entry.generation_trace_hash.clone())?;
        dict.set_item("support_relation_ids", entry.support_relation_ids.clone())?;
        dict.set_item(
            "counterexample_relation_ids",
            entry.counterexample_relation_ids.clone(),
        )?;
        list.append(dict)?;
    }
    Ok(list.into())
}

fn pack_proof_traces(
    py: Python<'_>,
    entries: &[xlog_logic::QueryProofTrace],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("query_id", &entry.query_id)?;
        dict.set_item("query", &entry.query)?;
        dict.set_item("answer_relation", &entry.answer_relation)?;
        dict.set_item("rule_ids", entry.rule_ids.clone())?;
        dict.set_item("source_facts", entry.source_facts.clone())?;
        dict.set_item("rejected_alternatives", entry.rejected_alternatives.clone())?;
        list.append(dict)?;
    }
    Ok(list.into())
}
