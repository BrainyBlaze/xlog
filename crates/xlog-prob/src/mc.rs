//! Approximate probabilistic inference via Monte Carlo sampling (P3).
//!
//! This engine samples probabilistic facts / annotated disjunction decisions on the GPU and
//! evaluates the deterministic core in each sampled world.
//!
//! For programs with non-monotone recursion (cycles through `not` and/or aggregates), Phase 4
//! requires explicit P3 opt-in. The deterministic evaluation uses a bounded, cycle-aware semantics:
//!
//! - If an SCC reaches a fixpoint under synchronous iteration, that fixpoint is used.
//! - If the SCC enters a cycle, the interpretation is the intersection of all states in the cycle
//!   (skeptical, invariant tuples only). This avoids parity/oscillation dependence on iteration
//!   count while remaining fully deterministic and explicit.

use std::collections::{HashMap, HashSet};
#[cfg(feature = "host-io")]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cudarc::driver::{CudaView, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{MemoryBudget, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_eval_kernels, MC_EVAL_MODULE};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
#[cfg(feature = "host-io")]
use xlog_logic::ast::AggExpr;
#[cfg(feature = "host-io")]
use xlog_logic::ast::Rule;
use xlog_logic::ast::{
    AggOp, AnnotatedDisjunction, Atom, BodyLiteral, Evidence, PredDecl, ProbFact, ProbQuery,
    Program, Term,
};
use xlog_logic::compile::Compiler;
use xlog_logic::stratify::analyze_stratification;
#[cfg(feature = "host-io")]
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering};
use xlog_runtime::Executor;

use crate::exact::GpuConfig;
use crate::provenance::{atom_key_from_ground_atom, validate_prob, GroundAtom, Value};
#[cfg(feature = "host-io")]
use crate::provenance::{eval_arith_expr, eval_comparison, unify_atom, value_from_term};

/// Sampling method for Monte Carlo inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McSamplingMethod {
    /// Sample from prior, discard worlds where evidence is not satisfied.
    Rejection,
    /// Force evidence variables in the sampler; every sample counts.
    EvidenceClamping,
}

/// Phase 4 semantics for non-monotone SCC evaluation inside MC sampling.
pub const NONMONOTONE_SEMANTICS: &str = "Synchronous iteration per SCC; if a fixpoint is reached, use it; if a cycle is detected, use the intersection of all states in the cycle (skeptical tuples only); if the iteration budget is exceeded, use the intersection across all visited states (conservative).";

fn upload_slice<T: cudarc::driver::DeviceRepr>(
    provider: &Arc<CudaKernelProvider>,
    src: &[T],
    dst: &mut TrackedCudaSlice<T>,
    label: &str,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    provider
        .device()
        .inner()
        .htod_sync_copy_into(src, dst)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload {}: {}", label, e)))
}

#[derive(Debug, Clone)]
pub struct McEvalConfig {
    /// Number of Monte Carlo samples.
    pub samples: usize,
    /// RNG seed (deterministic).
    pub seed: u64,
    /// Two-sided confidence level in (0,1) (e.g., 0.95).
    pub confidence: f64,
    /// Maximum SCC iteration steps for non-monotone cycle detection.
    pub max_nonmonotone_iterations: usize,
    /// Sampling method override.  `None` = auto-select (rejection for now).
    pub sampling_method: Option<McSamplingMethod>,
}

impl Default for McEvalConfig {
    fn default() -> Self {
        Self {
            samples: 10000,
            seed: 0,
            confidence: 0.95,
            max_nonmonotone_iterations: 1024,
            sampling_method: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct McQueryEstimate {
    pub atom: GroundAtom,
    pub prob: f64,
    pub log_prob: f64,
    pub stderr: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

#[derive(Debug, Clone)]
pub struct McResult {
    pub total_samples: usize,
    pub evidence_samples: usize,
    pub seed: u64,
    pub confidence: f64,
    pub query_estimates: Vec<McQueryEstimate>,
    pub nonmonotone_sccs: usize,
    pub nonmonotone_cycles: usize,
    pub nonmonotone_iteration_limit_hits: usize,
    pub sampling_method: McSamplingMethod,
}

/// Device-resident Monte Carlo result counts.
pub struct McDeviceResult {
    pub query_counts: TrackedCudaSlice<u32>,
    pub evidence_count: TrackedCudaSlice<u32>,
    pub total_samples: usize,
    pub seed: u64,
    pub confidence: f64,
    pub nonmonotone_sccs: usize,
    pub nonmonotone_cycles: usize,
    pub nonmonotone_iteration_limit_hits: usize,
    pub sampling_method: McSamplingMethod,
}

#[derive(Debug, Clone)]
struct ProbFactSpec {
    var_idx: usize,
    atom: GroundAtom,
}

#[derive(Debug, Clone)]
struct AdSpec {
    decision_vars: Vec<usize>,
    choices: Vec<GroundAtom>,
    has_none: bool,
}

struct GpuMcPlan {
    program: Program,
    plan: xlog_ir::ExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
    query_rel_names: Vec<String>,
    evidence_rel_specs: Vec<(String, bool)>,
    nonmonotone_sccs: HashSet<usize>,
}

struct ProbTableDevice {
    predicate: String,
    buffer: CudaBuffer,
    var_idx: TrackedCudaSlice<u32>,
}

struct AdDecisionDevice {
    decision_vars: TrackedCudaSlice<u32>,
}

struct AdTableDevice {
    predicate: String,
    buffer: CudaBuffer,
    decision_offsets: TrackedCudaSlice<u32>,
    decision_lengths: TrackedCudaSlice<u32>,
    choice_positions: TrackedCudaSlice<u32>,
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Relation {
    tuples: HashSet<Vec<Value>>,
}

#[cfg(feature = "host-io")]
impl Relation {
    fn insert_tuple(&mut self, tuple: Vec<Value>) {
        self.tuples.insert(tuple);
    }

    fn contains(&self, tuple: &[Value]) -> bool {
        self.tuples.contains(tuple)
    }

    fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone)]
enum SccKind {
    MonotoneNonRecursive,
    MonotoneRecursive,
    NonMonotone,
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone)]
struct SccPlan {
    predicates: Vec<String>,
    rules: Vec<Rule>,
    kind: SccKind,
}

#[derive(Debug, Clone, Default)]
struct EvalStats {
    nonmonotone_sccs: usize,
    nonmonotone_cycles: usize,
    nonmonotone_iteration_limit_hits: usize,
}

#[derive(Clone)]
pub struct McProgram {
    gpu_config: GpuConfig,
    program: Program,
    #[cfg(feature = "host-io")]
    base_store: HashMap<String, Relation>,
    #[cfg(feature = "host-io")]
    scc_plans: Vec<SccPlan>,
    queries: Vec<GroundAtom>,
    evidence: Vec<(GroundAtom, bool)>,
    bernoulli_probs: Vec<f32>,
    prob_facts: Vec<ProbFactSpec>,
    annotated_disjunctions: Vec<AdSpec>,
}

impl McProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;
        Self::compile_program(&program)
    }

    pub fn compile_source_with_gpu(source: &str, config: GpuConfig) -> Result<Self> {
        let mut program = Self::compile_source(source)?;
        program.gpu_config = config;
        Ok(program)
    }

    pub fn num_vars(&self) -> usize {
        self.bernoulli_probs.len()
    }

    #[cfg(feature = "host-io")]
    #[cfg(feature = "host-io")]
    pub fn evaluate(&self, cfg: McEvalConfig) -> Result<McResult> {
        let provider = Arc::new(self.provider()?);
        let cfg_clone = cfg.clone();
        let device_result = self.evaluate_gpu_device_with_provider(cfg_clone, provider.clone())?;

        let mut host_counts = vec![0u32; device_result.query_counts.len()];
        if !host_counts.is_empty() {
            provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&device_result.query_counts, &mut host_counts)
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to download MC query counts: {}", e))
                })?;
        }

        let mut host_evidence = [0u32];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&device_result.evidence_count, &mut host_evidence)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to download MC evidence count: {}", e))
            })?;

        let evidence_samples = if self.evidence.is_empty() {
            cfg.samples
        } else {
            host_evidence[0] as usize
        };

        if !self.evidence.is_empty() && evidence_samples == 0 {
            return Err(XlogError::Execution(format!(
                "MC inference error: evidence was never satisfied across {} samples (seed={})",
                cfg.samples, cfg.seed
            )));
        }

        let z = normal_quantile(0.5 + cfg.confidence / 2.0);
        let mut query_estimates: Vec<McQueryEstimate> = Vec::with_capacity(self.queries.len());
        for (i, atom) in self.queries.iter().enumerate() {
            let k = host_counts.get(i).copied().unwrap_or(0) as usize;
            let (p, stderr, ci_low, ci_high) = binomial_estimate(k, evidence_samples, z);
            let log_prob = if p == 0.0 { f64::NEG_INFINITY } else { p.ln() };

            query_estimates.push(McQueryEstimate {
                atom: atom.clone(),
                prob: p,
                log_prob,
                stderr,
                ci_low,
                ci_high,
            });
        }

        Ok(McResult {
            total_samples: cfg.samples,
            evidence_samples,
            seed: cfg.seed,
            confidence: cfg.confidence,
            query_estimates,
            nonmonotone_sccs: device_result.nonmonotone_sccs,
            nonmonotone_cycles: device_result.nonmonotone_cycles,
            nonmonotone_iteration_limit_hits: device_result.nonmonotone_iteration_limit_hits,
            sampling_method: McSamplingMethod::Rejection,
        })
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate_cpu(&self, cfg: McEvalConfig) -> Result<McResult> {
        if cfg.samples == 0 {
            return Err(XlogError::Execution(
                "MC inference requires samples > 0".to_string(),
            ));
        }
        if !(0.0 < cfg.confidence && cfg.confidence < 1.0) || cfg.confidence.is_nan() {
            return Err(XlogError::Execution(format!(
                "MC inference requires 0 < confidence < 1, got {}",
                cfg.confidence
            )));
        }
        if cfg.max_nonmonotone_iterations == 0 {
            return Err(XlogError::Execution(
                "MC inference requires max_nonmonotone_iterations > 0".to_string(),
            ));
        }

        let mut n_evidence: usize = 0;
        let mut n_query_true: Vec<usize> = vec![0; self.queries.len()];
        let mut stats = EvalStats::default();

        let num_vars = self.bernoulli_probs.len();
        let samples_matrix: Vec<u8> = if num_vars == 0 {
            Vec::new()
        } else {
            let total = num_vars
                .checked_mul(cfg.samples)
                .ok_or_else(|| XlogError::Execution("MC sample matrix overflow".to_string()))?;
            let provider = Arc::new(self.provider()?);

            // Allocate zero-filled force arrays (no clamping in evaluate_cpu path)
            let mut d_force_mask = provider.memory().alloc::<u8>(num_vars.max(1))?;
            provider.device().inner().memset_zeros(&mut d_force_mask)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
            let mut d_forced_value = provider.memory().alloc::<u8>(num_vars.max(1))?;
            provider.device().inner().memset_zeros(&mut d_forced_value)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;

            let samples_device = provider.sample_bernoulli_matrix_device(
                &self.bernoulli_probs,
                cfg.samples,
                cfg.seed,
                &d_force_mask.slice(..),
                &d_forced_value.slice(..),
            )?;
            let mut host = vec![0u8; total];
            if !host.is_empty() {
                provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(&samples_device, &mut host)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to download MC samples: {}", e))
                    })?;
            }
            host
        };

        for sample_idx in 0..cfg.samples {
            let sample_bits = if num_vars == 0 {
                &[][..]
            } else {
                let start = sample_idx * num_vars;
                let end = start + num_vars;
                &samples_matrix[start..end]
            };

            let mut store = self.base_store.clone();
            self.apply_sample_facts(&mut store, sample_bits)?;

            let sample_stats = evaluate_program_inplace(
                &self.scc_plans,
                &mut store,
                cfg.max_nonmonotone_iterations,
            )?;
            stats.nonmonotone_sccs += sample_stats.nonmonotone_sccs;
            stats.nonmonotone_cycles += sample_stats.nonmonotone_cycles;
            stats.nonmonotone_iteration_limit_hits += sample_stats.nonmonotone_iteration_limit_hits;

            if !evidence_satisfied(&store, &self.evidence) {
                continue;
            }

            n_evidence += 1;
            for (i, q) in self.queries.iter().enumerate() {
                if atom_holds(&store, q) {
                    n_query_true[i] += 1;
                }
            }
        }

        if !self.evidence.is_empty() && n_evidence == 0 {
            return Err(XlogError::Execution(format!(
                "MC inference error: evidence was never satisfied across {} samples (seed={})",
                cfg.samples, cfg.seed
            )));
        }

        // If there is no evidence, treat all samples as evidence-satisfying.
        let denom = if self.evidence.is_empty() {
            cfg.samples
        } else {
            n_evidence
        };

        let z = normal_quantile(0.5 + cfg.confidence / 2.0);

        let mut query_estimates: Vec<McQueryEstimate> = Vec::with_capacity(self.queries.len());
        for (i, atom) in self.queries.iter().enumerate() {
            let k = n_query_true[i];
            let (p, stderr, ci_low, ci_high) = binomial_estimate(k, denom, z);
            let log_prob = if p == 0.0 { f64::NEG_INFINITY } else { p.ln() };

            query_estimates.push(McQueryEstimate {
                atom: atom.clone(),
                prob: p,
                log_prob,
                stderr,
                ci_low,
                ci_high,
            });
        }

        Ok(McResult {
            total_samples: cfg.samples,
            evidence_samples: denom,
            seed: cfg.seed,
            confidence: cfg.confidence,
            query_estimates,
            nonmonotone_sccs: stats.nonmonotone_sccs,
            nonmonotone_cycles: stats.nonmonotone_cycles,
            nonmonotone_iteration_limit_hits: stats.nonmonotone_iteration_limit_hits,
            sampling_method: McSamplingMethod::Rejection,
        })
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu(&self, cfg: McEvalConfig) -> Result<McResult> {
        self.evaluate(cfg)
    }

    pub fn evaluate_gpu_device(&self, cfg: McEvalConfig) -> Result<McDeviceResult> {
        let provider = Arc::new(self.provider()?);
        self.evaluate_gpu_device_with_provider(cfg, provider)
    }

    pub fn evaluate_gpu_device_with_provider(
        &self,
        cfg: McEvalConfig,
        provider: Arc<CudaKernelProvider>,
    ) -> Result<McDeviceResult> {
        let prob_query_count = self.queries.len();
        let evidence_count = self.evidence.len();
        let prob_query_count_u32 = u32::try_from(prob_query_count).map_err(|_| {
            XlogError::Execution(format!(
                "MC inference requires queries <= u32::MAX (got {})",
                prob_query_count
            ))
        })?;
        let evidence_count_u32 = u32::try_from(evidence_count).map_err(|_| {
            XlogError::Execution(format!(
                "MC inference requires evidence <= u32::MAX (got {})",
                evidence_count
            ))
        })?;

        let mut d_query_counts = provider.memory().alloc::<u32>(prob_query_count)?;
        if prob_query_count > 0 {
            provider
                .device()
                .inner()
                .memset_zeros(&mut d_query_counts)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero MC query counts: {}", e)))?;
        }
        let mut d_evidence_count = provider.memory().alloc::<u32>(1)?;
        provider
            .device()
            .inner()
            .memset_zeros(&mut d_evidence_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero MC evidence count: {}", e)))?;

        let mut d_query_flags = provider.memory().alloc::<u8>(prob_query_count)?;
        let mut d_evidence_ok = provider.memory().alloc::<u8>(1)?;
        let mut d_query_ptrs = provider.memory().alloc::<u64>(prob_query_count)?;
        let mut d_evidence_ptrs = provider.memory().alloc::<u64>(evidence_count)?;
        let mut d_evidence_expected = provider.memory().alloc::<u8>(evidence_count)?;
        let mut d_zero_count = provider.memory().alloc::<u32>(1)?;
        provider
            .device()
            .inner()
            .memset_zeros(&mut d_zero_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero MC zero-count: {}", e)))?;

        if evidence_count > 0 {
            let expected: Vec<u8> = self
                .evidence
                .iter()
                .map(|(_, v)| if *v { 1u8 } else { 0u8 })
                .collect();
            upload_slice(
                &provider,
                &expected,
                &mut d_evidence_expected,
                "MC evidence expected",
            )?;
        }

        let truth_fn = provider
            .device()
            .inner()
            .get_func(
                MC_EVAL_MODULE,
                mc_eval_kernels::MC_EVAL_QUERY_EVIDENCE_TRUTH,
            )
            .ok_or_else(|| {
                XlogError::Kernel("mc_eval_query_evidence_truth kernel not found".to_string())
            })?;
        let accum_fn = provider
            .device()
            .inner()
            .get_func(MC_EVAL_MODULE, mc_eval_kernels::MC_EVAL_ACCUMULATE_COUNTS)
            .ok_or_else(|| {
                XlogError::Kernel("mc_accumulate_counts kernel not found".to_string())
            })?;
        let accum_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let stats =
            self.evaluate_gpu_counts_with(&cfg, provider.clone(), |executor, plan, count| {
                let zero_ptr = *d_zero_count.device_ptr() as u64;

                let mut query_ptrs: Vec<u64> = Vec::with_capacity(count);
                for rel_name in plan.query_rel_names.iter().take(count) {
                    let ptr = executor
                        .store()
                        .get(rel_name)
                        .map(|buf| *buf.num_rows_device().device_ptr() as u64)
                        .unwrap_or(zero_ptr);
                    query_ptrs.push(ptr);
                }
                upload_slice(
                    &provider,
                    &query_ptrs,
                    &mut d_query_ptrs,
                    "MC query count ptrs",
                )?;

                let mut evidence_ptrs: Vec<u64> = Vec::with_capacity(evidence_count);
                for (rel_name, _) in plan.evidence_rel_specs.iter() {
                    let ptr = executor
                        .store()
                        .get(rel_name)
                        .map(|buf| *buf.num_rows_device().device_ptr() as u64)
                        .unwrap_or(zero_ptr);
                    evidence_ptrs.push(ptr);
                }
                upload_slice(
                    &provider,
                    &evidence_ptrs,
                    &mut d_evidence_ptrs,
                    "MC evidence count ptrs",
                )?;

                let block_dim = 128u32;
                let threads = if count == 0 { 1 } else { count as u32 };
                let grid_dim = (threads + block_dim - 1) / block_dim;
                unsafe {
                    truth_fn
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (grid_dim, 1, 1),
                                block_dim: (block_dim, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_query_ptrs,
                                prob_query_count_u32,
                                &d_evidence_ptrs,
                                &d_evidence_expected,
                                evidence_count_u32,
                                &mut d_query_flags,
                                &mut d_evidence_ok,
                            ),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!("mc_eval_query_evidence_truth failed: {}", e))
                        })?;
                }
                unsafe {
                    accum_fn
                        .clone()
                        .launch(
                            accum_config,
                            (
                                &d_query_flags,
                                prob_query_count_u32,
                                &d_evidence_ok,
                                &mut d_query_counts,
                                &mut d_evidence_count,
                            ),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!("mc_accumulate_counts failed: {}", e))
                        })?;
                }
                Ok(())
            })?;

        provider.device().synchronize()?;

        Ok(McDeviceResult {
            query_counts: d_query_counts,
            evidence_count: d_evidence_count,
            total_samples: cfg.samples,
            seed: cfg.seed,
            confidence: cfg.confidence,
            nonmonotone_sccs: stats.nonmonotone_sccs,
            nonmonotone_cycles: stats.nonmonotone_cycles,
            nonmonotone_iteration_limit_hits: stats.nonmonotone_iteration_limit_hits,
            sampling_method: McSamplingMethod::Rejection,
        })
    }

    fn build_gpu_plan(&self) -> Result<GpuMcPlan> {
        let mut plan_program = self.program.clone();
        plan_program.queries.clear();

        for ProbQuery { atom } in &plan_program.prob_queries {
            plan_program
                .queries
                .push(xlog_logic::ast::Query { atom: atom.clone() });
        }

        let evidence_offset = plan_program.queries.len();
        for Evidence { atom, .. } in &plan_program.evidence {
            plan_program
                .queries
                .push(xlog_logic::ast::Query { atom: atom.clone() });
        }

        ensure_predicate_decls(&mut plan_program)?;

        let max_recursion = plan_program.directives.max_recursion_depth.unwrap_or(100);
        let expanded = xlog_logic::expand_program_functions(&plan_program, max_recursion)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&expanded)?;
        let mut schemas = compiler.schemas().clone();
        augment_schemas_for_program(&expanded, &mut schemas);
        let rel_ids = compiler.rel_ids().clone();

        let strat = analyze_stratification(&expanded);
        let nonmonotone_sccs = strat.non_monotone_sccs;

        let mut query_rel_names: Vec<String> = Vec::new();
        for i in 0..expanded.queries.len() {
            query_rel_names.push(format!("__xlog_query_{}", i));
        }

        let mut evidence_rel_specs: Vec<(String, bool)> = Vec::new();
        for (idx, Evidence { value, .. }) in expanded.evidence.iter().enumerate() {
            let rel_idx = evidence_offset + idx;
            evidence_rel_specs.push((format!("__xlog_query_{}", rel_idx), *value));
        }

        Ok(GpuMcPlan {
            program: expanded,
            plan,
            schemas,
            rel_ids,
            query_rel_names,
            evidence_rel_specs,
            nonmonotone_sccs,
        })
    }

    fn evaluate_gpu_counts_with<F>(
        &self,
        cfg: &McEvalConfig,
        provider: Arc<CudaKernelProvider>,
        mut on_sample: F,
    ) -> Result<EvalStats>
    where
        F: FnMut(&Executor, &GpuMcPlan, usize) -> Result<()>,
    {
        if cfg.samples == 0 {
            return Err(XlogError::Execution(
                "MC inference requires samples > 0".to_string(),
            ));
        }
        if !(0.0 < cfg.confidence && cfg.confidence < 1.0) || cfg.confidence.is_nan() {
            return Err(XlogError::Execution(format!(
                "MC inference requires 0 < confidence < 1, got {}",
                cfg.confidence
            )));
        }
        if cfg.max_nonmonotone_iterations == 0 {
            return Err(XlogError::Execution(
                "MC inference requires max_nonmonotone_iterations > 0".to_string(),
            ));
        }
        if cfg.samples > (u32::MAX as usize) {
            return Err(XlogError::Execution(format!(
                "MC inference requires samples <= u32::MAX (got {})",
                cfg.samples
            )));
        }

        let gpu_plan = self.build_gpu_plan()?;
        let prob_query_count = self.queries.len();

        let mut executor = Executor::new(provider.clone());
        executor.set_profiling(false);

        for (name, rel_id) in &gpu_plan.rel_ids {
            executor.register_relation(*rel_id, name);
        }

        for (name, schema) in &gpu_plan.schemas {
            executor.put_relation(name, provider.create_empty_buffer(schema.clone())?);
        }

        load_deterministic_facts(
            &gpu_plan.program,
            &gpu_plan.schemas,
            &provider,
            &mut executor,
        )?;

        let base_store = snapshot_store(&provider, &executor, &gpu_plan.schemas)?;

        let (prob_tables, ad_tables, ad_decisions) =
            build_prob_tables_device(self, &provider, &gpu_plan.schemas)?;

        let num_vars = self.bernoulli_probs.len();

        // Allocate zero-filled force arrays (no clamping by default)
        let mut d_force_mask = provider.memory().alloc::<u8>(num_vars.max(1))?;
        provider.device().inner().memset_zeros(&mut d_force_mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
        let mut d_forced_value = provider.memory().alloc::<u8>(num_vars.max(1))?;
        provider.device().inner().memset_zeros(&mut d_forced_value)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;

        let samples_device = if self.bernoulli_probs.is_empty() || cfg.samples == 0 {
            provider.memory().alloc::<u8>(0)?
        } else {
            provider.sample_bernoulli_matrix_device(
                &self.bernoulli_probs,
                cfg.samples,
                cfg.seed,
                &d_force_mask.slice(..),
                &d_forced_value.slice(..),
            )?
        };

        let mut stats = EvalStats::default();

        for sample_idx in 0..cfg.samples {
            executor.reset_for_mc();
            restore_store(&provider, &base_store, &mut executor)?;

            let start = sample_idx * num_vars;
            let end = start + num_vars;
            let sample_bits = samples_device.slice(start..end);

            let sample_buffers = build_sample_buffers(
                &provider,
                &sample_bits,
                &prob_tables,
                &ad_tables,
                &ad_decisions,
            )?;

            for (pred, buf) in sample_buffers {
                if buf.is_empty() {
                    continue;
                }
                let merged = match executor.store().get(&pred) {
                    Some(existing) if !existing.is_empty() => provider.union_gpu(existing, &buf)?,
                    _ => dedup_relation(&provider, &buf)?,
                };
                executor.put_relation(&pred, merged);
            }

            let sample_stats = evaluate_program_gpu(
                &provider,
                &mut executor,
                &gpu_plan.plan,
                &gpu_plan.nonmonotone_sccs,
                cfg.max_nonmonotone_iterations,
            )?;
            stats.nonmonotone_sccs += sample_stats.nonmonotone_sccs;
            stats.nonmonotone_cycles += sample_stats.nonmonotone_cycles;
            stats.nonmonotone_iteration_limit_hits += sample_stats.nonmonotone_iteration_limit_hits;

            on_sample(&executor, &gpu_plan, prob_query_count)?;
        }

        Ok(stats)
    }

    fn compile_program(program: &Program) -> Result<Self> {
        let mut queries: Vec<GroundAtom> = Vec::new();
        for ProbQuery { atom } in &program.prob_queries {
            queries.push(atom_key_from_ground_atom(atom)?);
        }

        let mut evidence: Vec<(GroundAtom, bool)> = Vec::new();
        for Evidence { atom, value } in &program.evidence {
            evidence.push((atom_key_from_ground_atom(atom)?, *value));
        }

        let mut prob_facts = program.prob_facts.clone();
        extend_prob_facts_with_coin(program, &mut prob_facts)?;
        let (bernoulli_probs, prob_facts, annotated_disjunctions) =
            compile_sampling_plan(&prob_facts, &program.annotated_disjunctions)?;

        #[cfg(feature = "host-io")]
        let mut base_store: HashMap<String, Relation> = HashMap::new();

        // Deterministic facts.
        #[cfg(feature = "host-io")]
        {
            for fact in program.facts() {
                let atom = atom_key_from_ground_atom(&fact.head)?;
                base_store
                    .entry(atom.predicate.clone())
                    .or_default()
                    .insert_tuple(atom.args);
            }
        }

        // Ensure relations exist for all referenced predicates so evaluation treats missing as empty,
        // but never errors due to an unknown predicate (CPU eval path only).
        #[cfg(feature = "host-io")]
        {
            let mut referenced: HashSet<String> = HashSet::new();
            for rule in &program.rules {
                referenced.insert(rule.head.predicate.clone());
                for lit in &rule.body {
                    match lit {
                        BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => {
                            referenced.insert(a.predicate.clone());
                        }
                        BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
                    }
                }
            }
            for pf in &program.prob_facts {
                referenced.insert(pf.atom.predicate.clone());
            }
            for ad in &program.annotated_disjunctions {
                for pf in &ad.choices {
                    referenced.insert(pf.atom.predicate.clone());
                }
            }
            for q in &queries {
                referenced.insert(q.predicate.clone());
            }
            for (e, _) in &evidence {
                referenced.insert(e.predicate.clone());
            }
            for pred in referenced {
                base_store.entry(pred).or_default();
            }
        }

        #[cfg(feature = "host-io")]
        let scc_plans = build_scc_plans(program)?;

        Ok(Self {
            gpu_config: GpuConfig::default(),
            program: program.clone(),
            #[cfg(feature = "host-io")]
            base_store,
            #[cfg(feature = "host-io")]
            scc_plans,
            queries,
            evidence,
            bernoulli_probs,
            prob_facts,
            annotated_disjunctions,
        })
    }

    fn provider(&self) -> Result<CudaKernelProvider> {
        if self.gpu_config.memory_bytes == 0 {
            return Err(XlogError::Kernel(
                "GPU memory budget must be non-zero".to_string(),
            ));
        }

        let device = Arc::new(CudaDevice::new(self.gpu_config.device_ordinal)?);
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(self.gpu_config.memory_bytes),
        ));
        CudaKernelProvider::new(device, memory)
    }

    #[cfg(feature = "host-io")]
    fn apply_sample_facts(&self, store: &mut HashMap<String, Relation>, bits: &[u8]) -> Result<()> {
        for pf in &self.prob_facts {
            if bits.get(pf.var_idx).copied().unwrap_or(0) == 0 {
                continue;
            }
            store
                .entry(pf.atom.predicate.clone())
                .or_default()
                .insert_tuple(pf.atom.args.clone());
        }

        for ad in &self.annotated_disjunctions {
            let mut selected: Option<usize> = None;
            for (idx, &var_idx) in ad.decision_vars.iter().enumerate() {
                if bits.get(var_idx).copied().unwrap_or(0) != 0 {
                    selected = Some(idx);
                    break;
                }
            }

            let outcome = match selected {
                Some(i) => Some(i),
                None => {
                    if ad.has_none {
                        None
                    } else {
                        Some(ad.choices.len().saturating_sub(1))
                    }
                }
            };

            if let Some(outcome_idx) = outcome {
                let atom = ad.choices.get(outcome_idx).ok_or_else(|| {
                    XlogError::Compilation(
                        "Annotated disjunction outcome index out of range".to_string(),
                    )
                })?;
                store
                    .entry(atom.predicate.clone())
                    .or_default()
                    .insert_tuple(atom.args.clone());
            }
        }

        Ok(())
    }

    pub fn compile_evidence_forcing(&self) -> Result<EvidenceForcing> {
        let num_vars = self.bernoulli_probs.len();
        let mut force_mask = vec![0u8; num_vars];
        let mut forced_value = vec![0u8; num_vars];

        if self.evidence.is_empty() {
            return Ok(EvidenceForcing {
                force_mask,
                forced_value,
                forceable: false,
                reason: ForceabilityReason::NoEvidence,
            });
        }

        for (atom, expected) in &self.evidence {
            // Try to match against prob fact specs
            if let Some(spec) = self.prob_facts.iter().find(|s| &s.atom == atom) {
                force_mask[spec.var_idx] = 1;
                forced_value[spec.var_idx] = if *expected { 1 } else { 0 };
                continue;
            }

            // Try to match against AD choice atoms (positive evidence only)
            let mut found_ad = false;
            for ad in &self.annotated_disjunctions {
                if let Some(choice_idx) = ad.choices.iter().position(|c| c == atom) {
                    if !*expected {
                        // evidence(ad_head, false) — not forceable in v0.5.1
                        return Ok(EvidenceForcing {
                            force_mask: vec![0u8; num_vars],
                            forced_value: vec![0u8; num_vars],
                            forceable: false,
                            reason: ForceabilityReason::ContainsNegativeAdHeadEvidence,
                        });
                    }

                    let num_decision_vars = ad.decision_vars.len();
                    if choice_idx < num_decision_vars {
                        // Force d_i = 0 for all i < choice_idx, d_{choice_idx} = 1
                        for i in 0..choice_idx {
                            force_mask[ad.decision_vars[i]] = 1;
                            forced_value[ad.decision_vars[i]] = 0;
                        }
                        force_mask[ad.decision_vars[choice_idx]] = 1;
                        forced_value[ad.decision_vars[choice_idx]] = 1;
                    } else {
                        // Last head (no none branch): force all decision vars to 0
                        for &dv in &ad.decision_vars {
                            force_mask[dv] = 1;
                            forced_value[dv] = 0;
                        }
                    }
                    found_ad = true;
                    break;
                }
            }
            if found_ad {
                continue;
            }

            // Evidence atom not found in prob facts or AD choices → derived/deterministic
            return Ok(EvidenceForcing {
                force_mask: vec![0u8; num_vars],
                forced_value: vec![0u8; num_vars],
                forceable: false,
                reason: ForceabilityReason::ContainsDerivedEvidence,
            });
        }

        Ok(EvidenceForcing {
            force_mask,
            forced_value,
            forceable: true,
            reason: ForceabilityReason::AllForceable,
        })
    }
}

fn load_deterministic_facts(
    program: &Program,
    schemas: &HashMap<String, Schema>,
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
) -> Result<()> {
    let mut rows_by_pred: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    for fact in program.facts() {
        let atom = atom_key_from_ground_atom(&fact.head)?;
        rows_by_pred
            .entry(atom.predicate.clone())
            .or_default()
            .push(atom.args);
    }

    for (pred, rows) in rows_by_pred {
        let schema = schemas.get(&pred).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing schema for deterministic predicate {}",
                pred
            ))
        })?;
        let buffer = build_buffer_from_rows(provider, schema, &rows)?;
        let deduped = dedup_relation(provider, &buffer)?;
        executor.put_relation(&pred, deduped);
    }

    Ok(())
}

fn snapshot_store(
    provider: &Arc<CudaKernelProvider>,
    executor: &Executor,
    schemas: &HashMap<String, Schema>,
) -> Result<HashMap<String, CudaBuffer>> {
    let mut snapshot: HashMap<String, CudaBuffer> = HashMap::new();
    for (pred, schema) in schemas {
        let buffer = executor
            .store()
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing relation {}", pred)))?;
        let cloned = if buffer.is_empty() {
            provider.create_empty_buffer(schema.clone())?
        } else {
            clone_buffer_device(provider, buffer)?
        };
        snapshot.insert(pred.clone(), cloned);
    }
    Ok(snapshot)
}

fn restore_store(
    provider: &Arc<CudaKernelProvider>,
    base_store: &HashMap<String, CudaBuffer>,
    executor: &mut Executor,
) -> Result<()> {
    for (pred, buffer) in base_store {
        let cloned = if buffer.is_empty() {
            provider.create_empty_buffer(buffer.schema().clone())?
        } else {
            clone_buffer_device(provider, buffer)?
        };
        executor.put_relation(pred, cloned);
    }
    Ok(())
}

fn clone_buffer_device(
    provider: &Arc<CudaKernelProvider>,
    buffer: &CudaBuffer,
) -> Result<CudaBuffer> {
    if buffer.is_empty() {
        return provider.create_empty_buffer(buffer.schema().clone());
    }

    let mut result_columns = Vec::with_capacity(buffer.arity());
    for col_idx in 0..buffer.arity() {
        let col_type_size = buffer
            .schema()
            .column_type(col_idx)
            .map(|t| t.size_bytes())
            .unwrap_or(4);
        let bytes = (buffer.num_rows() as usize) * col_type_size;
        let Some(src_col) = buffer.column(col_idx) else {
            continue;
        };
        let mut dst_col = provider.memory().alloc::<u8>(bytes)?;
        if bytes > 0 {
            provider
                .device()
                .inner()
                .dtod_copy(src_col, &mut dst_col)
                .map_err(|e| {
                    XlogError::Execution(format!("Failed to clone column on device: {}", e))
                })?;
        }
        result_columns.push(dst_col.into());
    }

    let mut d_num_rows = provider.memory().alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
        .map_err(|e| XlogError::Execution(format!("Failed to copy row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(
        result_columns,
        buffer.num_rows(),
        d_num_rows,
        buffer.schema().clone(),
    ))
}

fn build_zero_arity_buffer(
    provider: &Arc<CudaKernelProvider>,
    row_count: u32,
    schema: &Schema,
) -> Result<CudaBuffer> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[row_count], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(
        Vec::new(),
        row_count as u64,
        d_num_rows,
        schema.clone(),
    ))
}

fn device_row_count_u32(provider: &Arc<CudaKernelProvider>, buffer: &CudaBuffer) -> Result<u32> {
    let mut host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    Ok(host[0])
}

fn dedup_relation(provider: &Arc<CudaKernelProvider>, buffer: &CudaBuffer) -> Result<CudaBuffer> {
    let rows = device_row_count_u32(provider, buffer)?;
    if rows == 0 {
        return provider.create_empty_buffer(buffer.schema().clone());
    }
    if buffer.arity() == 0 {
        return build_zero_arity_buffer(provider, 1u32, buffer.schema());
    }
    let key_cols: Vec<usize> = (0..buffer.arity()).collect();
    provider.dedup(buffer, &key_cols)
}

fn build_prob_tables_device(
    program: &McProgram,
    provider: &Arc<CudaKernelProvider>,
    schemas: &HashMap<String, Schema>,
) -> Result<(Vec<ProbTableDevice>, Vec<AdTableDevice>, AdDecisionDevice)> {
    let mut prob_rows_by_pred: HashMap<String, Vec<(Vec<Value>, u32)>> = HashMap::new();
    for pf in &program.prob_facts {
        prob_rows_by_pred
            .entry(pf.atom.predicate.clone())
            .or_default()
            .push((pf.atom.args.clone(), pf.var_idx as u32));
    }

    let mut prob_tables: Vec<ProbTableDevice> = Vec::new();
    for (pred, rows) in prob_rows_by_pred {
        let mut tuples: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        let mut var_idx: Vec<u32> = Vec::with_capacity(rows.len());
        for (tuple, idx) in rows {
            tuples.push(tuple);
            var_idx.push(idx);
        }

        let schema = match schemas.get(&pred) {
            Some(schema) => schema.clone(),
            None => infer_schema_from_values(&tuples)?,
        };

        let buffer = build_buffer_from_rows(provider, &schema, &tuples)?;
        let mut d_var_idx = provider.memory().alloc::<u32>(var_idx.len())?;
        upload_slice(provider, &var_idx, &mut d_var_idx, "prob var indices")?;

        prob_tables.push(ProbTableDevice {
            predicate: pred,
            buffer,
            var_idx: d_var_idx,
        });
    }

    #[derive(Debug)]
    struct AdRow {
        args: Vec<Value>,
        offset: u32,
        len: u32,
        pos: u32,
    }

    let mut decision_vars_flat: Vec<u32> = Vec::new();
    let mut ad_rows_by_pred: HashMap<String, Vec<AdRow>> = HashMap::new();

    for ad in &program.annotated_disjunctions {
        let offset = decision_vars_flat.len() as u32;
        let len = ad.decision_vars.len() as u32;
        decision_vars_flat.extend(ad.decision_vars.iter().map(|v| *v as u32));

        let choices_len = ad.choices.len();
        for (idx, atom) in ad.choices.iter().enumerate() {
            let pos = if ad.has_none {
                idx as u32
            } else if idx + 1 == choices_len {
                len
            } else {
                idx as u32
            };
            ad_rows_by_pred
                .entry(atom.predicate.clone())
                .or_default()
                .push(AdRow {
                    args: atom.args.clone(),
                    offset,
                    len,
                    pos,
                });
        }
    }

    let mut ad_tables: Vec<AdTableDevice> = Vec::new();
    for (pred, rows) in ad_rows_by_pred {
        let mut tuples: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        let mut offsets: Vec<u32> = Vec::with_capacity(rows.len());
        let mut lengths: Vec<u32> = Vec::with_capacity(rows.len());
        let mut positions: Vec<u32> = Vec::with_capacity(rows.len());

        for row in rows {
            tuples.push(row.args);
            offsets.push(row.offset);
            lengths.push(row.len);
            positions.push(row.pos);
        }

        let schema = match schemas.get(&pred) {
            Some(schema) => schema.clone(),
            None => infer_schema_from_values(&tuples)?,
        };

        let buffer = build_buffer_from_rows(provider, &schema, &tuples)?;
        let mut d_offsets = provider.memory().alloc::<u32>(offsets.len())?;
        let mut d_lengths = provider.memory().alloc::<u32>(lengths.len())?;
        let mut d_positions = provider.memory().alloc::<u32>(positions.len())?;
        upload_slice(provider, &offsets, &mut d_offsets, "AD offsets")?;
        upload_slice(provider, &lengths, &mut d_lengths, "AD lengths")?;
        upload_slice(provider, &positions, &mut d_positions, "AD positions")?;

        ad_tables.push(AdTableDevice {
            predicate: pred,
            buffer,
            decision_offsets: d_offsets,
            decision_lengths: d_lengths,
            choice_positions: d_positions,
        });
    }

    let mut d_decision_vars = provider.memory().alloc::<u32>(decision_vars_flat.len())?;
    upload_slice(
        provider,
        &decision_vars_flat,
        &mut d_decision_vars,
        "AD decision vars",
    )?;

    Ok((
        prob_tables,
        ad_tables,
        AdDecisionDevice {
            decision_vars: d_decision_vars,
        },
    ))
}

fn build_sample_buffers(
    provider: &Arc<CudaKernelProvider>,
    sample_bits: &CudaView<'_, u8>,
    prob_tables: &[ProbTableDevice],
    ad_tables: &[AdTableDevice],
    ad_decisions: &AdDecisionDevice,
) -> Result<Vec<(String, CudaBuffer)>> {
    if sample_bits.len() == 0 && (!prob_tables.is_empty() || ad_decisions.decision_vars.len() > 0) {
        return Err(XlogError::Execution(
            "MC sample bits empty but probabilistic variables exist".to_string(),
        ));
    }

    let device = provider.device().inner();
    let mut out: Vec<(String, CudaBuffer)> = Vec::new();

    let block_size = 256u32;

    for table in prob_tables {
        if table.buffer.is_empty() {
            continue;
        }
        let n_u64 = table.buffer.num_rows();
        let n: u32 = n_u64.try_into().map_err(|_| {
            XlogError::Execution(format!(
                "Prob table {} rows {} exceed u32",
                table.predicate, n_u64
            ))
        })?;

        let mut d_mask = provider.memory().alloc::<u8>(n as usize)?;
        let num_blocks = (n + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_EVAL_MODULE, mc_eval_kernels::MC_EVAL_MASK_VAR)
            .ok_or_else(|| XlogError::Kernel("mc_eval_mask_var kernel not found".to_string()))?;

        // SAFETY: mc_eval_mask_var(sample_bits, var_idx, n, out_mask)
        unsafe {
            kernel
                .clone()
                .launch(config, (sample_bits, &table.var_idx, n, &mut d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("mc_eval_mask_var failed: {}", e)))?;

        let filtered = provider.compact_buffer_by_device_mask_counted(&table.buffer, &d_mask)?;
        let filtered_rows = device_row_count_u32(provider, &filtered)?;
        if filtered_rows == 0 {
            continue;
        }
        let deduped = dedup_relation(provider, &filtered)?;
        out.push((table.predicate.clone(), deduped));
    }

    for table in ad_tables {
        if table.buffer.is_empty() {
            continue;
        }
        let n_u64 = table.buffer.num_rows();
        let n: u32 = n_u64.try_into().map_err(|_| {
            XlogError::Execution(format!(
                "AD table {} rows {} exceed u32",
                table.predicate, n_u64
            ))
        })?;

        let mut d_mask = provider.memory().alloc::<u8>(n as usize)?;
        let num_blocks = (n + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_EVAL_MODULE, mc_eval_kernels::MC_EVAL_MASK_AD)
            .ok_or_else(|| {
                XlogError::Kernel("mc_eval_mask_ad_choice kernel not found".to_string())
            })?;

        // SAFETY: mc_eval_mask_ad_choice(sample_bits, decision_vars, offsets, lengths, positions, n, out_mask)
        unsafe {
            kernel.clone().launch(
                config,
                (
                    sample_bits,
                    &ad_decisions.decision_vars,
                    &table.decision_offsets,
                    &table.decision_lengths,
                    &table.choice_positions,
                    n,
                    &mut d_mask,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mc_eval_mask_ad_choice failed: {}", e)))?;

        let filtered = provider.compact_buffer_by_device_mask_counted(&table.buffer, &d_mask)?;
        let filtered_rows = device_row_count_u32(provider, &filtered)?;
        if filtered_rows == 0 {
            continue;
        }
        let deduped = dedup_relation(provider, &filtered)?;
        out.push((table.predicate.clone(), deduped));
    }

    Ok(out)
}

fn evaluate_program_gpu(
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
    plan: &xlog_ir::ExecutionPlan,
    nonmonotone_sccs: &HashSet<usize>,
    max_nonmonotone_iterations: usize,
) -> Result<EvalStats> {
    let mut stats = EvalStats::default();

    if plan.strata.is_empty() {
        for (idx, scc) in plan.sccs.iter().enumerate() {
            let rules = plan
                .rules_by_scc
                .get(idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing rules for SCC {}", idx)))?;
            if nonmonotone_sccs.contains(&idx) {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = execute_nonmonotone_scc_gpu(
                    provider,
                    executor,
                    &scc.predicates,
                    rules,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            } else if scc.is_recursive {
                executor.execute_recursive_scc(rules)?;
            } else {
                executor.execute_non_recursive_scc(rules)?;
            }
        }
        return Ok(stats);
    }

    for stratum in &plan.strata {
        for &scc_id in &stratum.sccs {
            let scc_idx = scc_id as usize;
            let scc = plan.sccs.get(scc_idx).ok_or_else(|| {
                XlogError::Execution(format!("Missing SCC metadata for {}", scc_id))
            })?;
            let rules = plan
                .rules_by_scc
                .get(scc_idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing rules for SCC {}", scc_id)))?;

            if nonmonotone_sccs.contains(&scc_idx) {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = execute_nonmonotone_scc_gpu(
                    provider,
                    executor,
                    &scc.predicates,
                    rules,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            } else if scc.is_recursive {
                executor.execute_recursive_scc(rules)?;
            } else {
                executor.execute_non_recursive_scc(rules)?;
            }
        }
    }

    Ok(stats)
}

fn execute_nonmonotone_scc_gpu(
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
    preds: &[String],
    rules: &[xlog_ir::CompiledRule],
    max_iters: usize,
) -> Result<(bool, bool)> {
    let base_state = snapshot_scc_state(provider, executor, preds)?;
    let mut history: Vec<HashMap<String, CudaBuffer>> = Vec::new();
    history.push(clone_state(provider, &base_state)?);
    let mut signatures: Vec<Vec<u64>> = vec![state_signature(&history[0], preds)];

    for _ in 0..max_iters {
        let mut next_state = clone_state(provider, &base_state)?;

        for rule in rules {
            let mut result = executor.execute_node(&rule.body)?;
            if result.is_empty() {
                continue;
            }
            result = dedup_relation(provider, &result)?;
            if result.is_empty() {
                continue;
            }
            if let Some(entry) = next_state.get_mut(&rule.head) {
                if entry.is_empty() {
                    *entry = result;
                } else {
                    let merged = provider.union_gpu(entry, &result)?;
                    *entry = merged;
                }
            } else {
                next_state.insert(rule.head.clone(), result);
            }
        }

        let current = history
            .last()
            .ok_or_else(|| XlogError::Execution("Missing current state".to_string()))?;
        if states_equal(provider, current, &next_state, preds)? {
            apply_state_to_store_move(executor, next_state);
            return Ok((false, false));
        }

        let sig = state_signature(&next_state, preds);
        for (idx, prev_sig) in signatures.iter().enumerate() {
            if *prev_sig != sig {
                continue;
            }
            let candidate = history.get(idx).ok_or_else(|| {
                XlogError::Execution("Nonmonotone history index out of range".to_string())
            })?;
            if states_equal(provider, candidate, &next_state, preds)? {
                let final_state = intersect_states_device(provider, &history[idx..], preds)?;
                apply_state_to_store_move(executor, final_state);
                return Ok((true, false));
            }
        }

        apply_state_to_store_move(executor, next_state);
        history.push(snapshot_scc_state(provider, executor, preds)?);
        signatures.push(sig);
    }

    let final_state = intersect_states_device(provider, &history, preds)?;
    apply_state_to_store_move(executor, final_state);
    Ok((false, true))
}

fn snapshot_scc_state(
    provider: &Arc<CudaKernelProvider>,
    executor: &Executor,
    preds: &[String],
) -> Result<HashMap<String, CudaBuffer>> {
    let mut state = HashMap::new();
    for pred in preds {
        let buf = executor
            .store()
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing relation {}", pred)))?;
        let cloned = if buf.is_empty() {
            provider.create_empty_buffer(buf.schema().clone())?
        } else {
            clone_buffer_device(provider, buf)?
        };
        state.insert(pred.clone(), cloned);
    }
    Ok(state)
}

fn clone_state(
    provider: &Arc<CudaKernelProvider>,
    state: &HashMap<String, CudaBuffer>,
) -> Result<HashMap<String, CudaBuffer>> {
    let mut out = HashMap::new();
    for (pred, buf) in state {
        let cloned = if buf.is_empty() {
            provider.create_empty_buffer(buf.schema().clone())?
        } else {
            clone_buffer_device(provider, buf)?
        };
        out.insert(pred.clone(), cloned);
    }
    Ok(out)
}

fn apply_state_to_store_move(executor: &mut Executor, state: HashMap<String, CudaBuffer>) {
    for (pred, buf) in state {
        executor.put_relation(&pred, buf);
    }
}

fn state_signature(state: &HashMap<String, CudaBuffer>, preds: &[String]) -> Vec<u64> {
    let mut sig = Vec::with_capacity(preds.len());
    for pred in preds {
        let rows = state.get(pred).map(|b| b.num_rows()).unwrap_or(0);
        sig.push(rows);
    }
    sig
}

fn states_equal(
    provider: &Arc<CudaKernelProvider>,
    a: &HashMap<String, CudaBuffer>,
    b: &HashMap<String, CudaBuffer>,
    preds: &[String],
) -> Result<bool> {
    for pred in preds {
        let buf_a = a
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        let buf_b = b
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        if !buffers_equal(provider, buf_a, buf_b)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn buffers_equal(
    provider: &Arc<CudaKernelProvider>,
    a: &CudaBuffer,
    b: &CudaBuffer,
) -> Result<bool> {
    if a.num_rows() != b.num_rows() {
        return Ok(false);
    }
    if a.is_empty() && b.is_empty() {
        return Ok(true);
    }

    let diff_ab = provider.diff_gpu(a, b)?;
    if !diff_ab.is_empty() {
        return Ok(false);
    }
    let diff_ba = provider.diff_gpu(b, a)?;
    Ok(diff_ba.is_empty())
}

fn intersect_states_device(
    provider: &Arc<CudaKernelProvider>,
    states: &[HashMap<String, CudaBuffer>],
    preds: &[String],
) -> Result<HashMap<String, CudaBuffer>> {
    let mut out: HashMap<String, CudaBuffer> = HashMap::new();
    let Some(first) = states.first() else {
        return Ok(out);
    };

    for pred in preds {
        let first_buf = first
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        let mut acc = if first_buf.is_empty() {
            provider.create_empty_buffer(first_buf.schema().clone())?
        } else {
            clone_buffer_device(provider, first_buf)?
        };
        for state in &states[1..] {
            let next = state
                .get(pred)
                .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
            acc = buffer_intersection(provider, &acc, next)?;
            if acc.is_empty() {
                break;
            }
        }
        out.insert(pred.clone(), acc);
    }

    Ok(out)
}

fn buffer_intersection(
    provider: &Arc<CudaKernelProvider>,
    a: &CudaBuffer,
    b: &CudaBuffer,
) -> Result<CudaBuffer> {
    if a.is_empty() || b.is_empty() {
        return provider.create_empty_buffer(a.schema().clone());
    }
    let diff = provider.diff_gpu(a, b)?;
    if diff.is_empty() {
        return clone_buffer_device(provider, a);
    }
    provider.diff_gpu(a, &diff)
}

fn build_buffer_from_rows(
    provider: &Arc<CudaKernelProvider>,
    schema: &Schema,
    rows: &[Vec<Value>],
) -> Result<CudaBuffer> {
    if schema.arity() == 0 {
        for row in rows {
            if !row.is_empty() {
                return Err(XlogError::Execution(
                    "Zero-arity buffer row should be empty".to_string(),
                ));
            }
        }
        if rows.is_empty() {
            return provider.create_empty_buffer(schema.clone());
        }
        let row_count = u32::try_from(rows.len()).map_err(|_| {
            XlogError::Execution(format!(
                "Row count {} exceeds u32::MAX for zero-arity buffer",
                rows.len()
            ))
        })?;
        return build_zero_arity_buffer(provider, row_count, schema);
    }

    if rows.is_empty() {
        return provider.create_empty_buffer(schema.clone());
    }

    let mut columns: Vec<Vec<u8>> = Vec::with_capacity(schema.arity());
    for col_idx in 0..schema.arity() {
        let col_size = schema
            .column_type(col_idx)
            .map(|t| t.size_bytes())
            .unwrap_or(4);
        columns.push(Vec::with_capacity(rows.len() * col_size));
    }

    for row in rows {
        if row.len() != schema.arity() {
            return Err(XlogError::Execution(format!(
                "Row arity {} does not match schema arity {}",
                row.len(),
                schema.arity()
            )));
        }
        for (idx, value) in row.iter().enumerate() {
            let col_type = schema
                .column_type(idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing column type for {}", idx)))?;
            push_value_bytes(&mut columns[idx], col_type, value)?;
        }
    }

    let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
    provider.create_buffer_from_slices(&slices, schema.clone())
}

fn augment_schemas_for_program(program: &Program, schemas: &mut HashMap<String, Schema>) {
    for fact in program.facts() {
        ensure_schema_for_atom(&fact.head, schemas);
    }

    for rule in &program.rules {
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    ensure_schema_for_atom(atom, schemas);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
            }
        }
    }

    for pf in &program.prob_facts {
        ensure_schema_for_atom(&pf.atom, schemas);
    }

    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            ensure_schema_for_atom(&choice.atom, schemas);
        }
    }

    for ProbQuery { atom } in &program.prob_queries {
        ensure_schema_for_atom(atom, schemas);
    }

    for Evidence { atom, .. } in &program.evidence {
        ensure_schema_for_atom(atom, schemas);
    }
}

fn ensure_predicate_decls(program: &mut Program) -> Result<()> {
    let mut declared: HashMap<String, Vec<ScalarType>> = HashMap::new();
    for pred in &program.predicates {
        declared.insert(pred.name.clone(), pred.types.clone());
    }

    let mut inferred: HashMap<String, Vec<ScalarType>> = HashMap::new();

    let mut record_atom = |atom: &Atom| {
        let types: Vec<ScalarType> = atom.terms.iter().map(infer_term_scalar_type).collect();
        match inferred.get(&atom.predicate) {
            Some(existing) if *existing != types => Err(XlogError::Compilation(format!(
                "Inconsistent predicate types for {}",
                atom.predicate
            ))),
            Some(_) => Ok(()),
            None => {
                inferred.insert(atom.predicate.clone(), types);
                Ok(())
            }
        }
    };

    for fact in program.facts() {
        record_atom(&fact.head)?;
    }

    for rule in &program.rules {
        record_atom(&rule.head)?;
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    record_atom(atom)?;
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
            }
        }
    }

    for pf in &program.prob_facts {
        record_atom(&pf.atom)?;
    }

    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            record_atom(&choice.atom)?;
        }
    }

    for ProbQuery { atom } in &program.prob_queries {
        record_atom(atom)?;
    }
    for Evidence { atom, .. } in &program.evidence {
        record_atom(atom)?;
    }

    for (pred, types) in inferred {
        if let Some(existing) = declared.get(&pred) {
            if existing != &types {
                return Err(XlogError::Compilation(format!(
                    "Predicate {} declared with {:?} but inferred {:?}",
                    pred, existing, types
                )));
            }
            continue;
        }
        program.predicates.push(PredDecl {
            name: pred,
            types,
            is_private: false,
        });
    }

    Ok(())
}

fn ensure_schema_for_atom(atom: &Atom, schemas: &mut HashMap<String, Schema>) {
    if schemas.contains_key(&atom.predicate) {
        return;
    }

    let columns: Vec<(String, ScalarType)> = atom
        .terms
        .iter()
        .enumerate()
        .map(|(i, term)| (format!("c{}", i), infer_term_scalar_type(term)))
        .collect();
    schemas.insert(atom.predicate.clone(), Schema::new(columns));
}

fn infer_term_scalar_type(term: &Term) -> ScalarType {
    match term {
        Term::Variable(_) | Term::Anonymous => ScalarType::U64,
        Term::Integer(i) => {
            if *i >= 0 && *i <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Term::Float(_) => ScalarType::F64,
        Term::String(_) | Term::Symbol(_) => ScalarType::Symbol,
        Term::Aggregate(agg) => match agg.op {
            AggOp::Count => ScalarType::U32,
            AggOp::Sum => ScalarType::U64,
            AggOp::Min | AggOp::Max => ScalarType::U32,
            AggOp::LogSumExp => ScalarType::F64,
        },
    }
}

fn infer_schema_from_values(rows: &[Vec<Value>]) -> Result<Schema> {
    if rows.is_empty() {
        return Err(XlogError::Execution(
            "Cannot infer schema from empty rows".to_string(),
        ));
    }
    let arity = rows[0].len();
    let mut types: Vec<Option<ScalarType>> = vec![None; arity];

    for row in rows {
        if row.len() != arity {
            return Err(XlogError::Execution(format!(
                "Row arity {} does not match inferred arity {}",
                row.len(),
                arity
            )));
        }
        for (idx, value) in row.iter().enumerate() {
            let ty = scalar_type_from_value(value);
            match types[idx] {
                Some(existing) if existing != ty => {
                    return Err(XlogError::Execution(format!(
                        "Inconsistent types for column {}: {:?} vs {:?}",
                        idx, existing, ty
                    )))
                }
                None => types[idx] = Some(ty),
                _ => {}
            }
        }
    }

    let columns: Vec<(String, ScalarType)> = types
        .into_iter()
        .enumerate()
        .map(|(i, ty)| (format!("c{}", i), ty.unwrap_or(ScalarType::U64)))
        .collect();
    Ok(Schema::new(columns))
}

fn scalar_type_from_value(value: &Value) -> ScalarType {
    match value {
        Value::I64(v) => {
            if *v >= 0 && *v <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Value::F64(_) => ScalarType::F64,
        Value::Symbol(_) | Value::String(_) => ScalarType::Symbol,
    }
}

fn extend_prob_facts_with_coin(program: &Program, prob_facts: &mut Vec<ProbFact>) -> Result<()> {
    let mut seen: HashSet<GroundAtom> = HashSet::new();
    for pf in prob_facts.iter() {
        seen.insert(atom_key_from_ground_atom(&pf.atom)?);
    }

    for rule in &program.rules {
        for lit in &rule.body {
            let BodyLiteral::Positive(atom) = lit else {
                continue;
            };
            if atom.predicate != "coin" || atom.terms.len() != 1 {
                continue;
            }
            let Term::Float(prob) = atom.terms[0] else {
                continue;
            };
            let key = atom_key_from_ground_atom(atom)?;
            if seen.insert(key) {
                prob_facts.push(ProbFact {
                    prob,
                    atom: atom.clone(),
                });
            }
        }
    }

    Ok(())
}

fn push_value_bytes(out: &mut Vec<u8>, col_type: ScalarType, value: &Value) -> Result<()> {
    match col_type {
        ScalarType::U32 => match value {
            Value::I64(v) => {
                let v_u32 = u32::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for u32", v))
                })?;
                out.extend_from_slice(&v_u32.to_le_bytes());
            }
            Value::Symbol(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for u32".to_string(),
                ))
            }
        },
        ScalarType::U64 => match value {
            Value::I64(v) => {
                let v_u64 = u64::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for u64", v))
                })?;
                out.extend_from_slice(&v_u64.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for u64".to_string(),
                ))
            }
        },
        ScalarType::I32 => match value {
            Value::I64(v) => {
                let v_i32 = i32::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for i32", v))
                })?;
                out.extend_from_slice(&v_i32.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for i32".to_string(),
                ))
            }
        },
        ScalarType::I64 => match value {
            Value::I64(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for i64".to_string(),
                ))
            }
        },
        ScalarType::F32 => match value {
            Value::F64(bits) => {
                let v = f64::from_bits(*bits) as f32;
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::I64(v) => {
                let v = *v as f32;
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected numeric value for f32".to_string(),
                ))
            }
        },
        ScalarType::F64 => match value {
            Value::F64(bits) => {
                let v = f64::from_bits(*bits);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::I64(v) => {
                let v = *v as f64;
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected numeric value for f64".to_string(),
                ))
            }
        },
        ScalarType::Bool => match value {
            Value::I64(v) => {
                let b = match *v {
                    0 => 0u8,
                    1 => 1u8,
                    _ => {
                        return Err(XlogError::Execution(
                            "Boolean value must be 0 or 1".to_string(),
                        ))
                    }
                };
                out.push(b);
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for bool".to_string(),
                ))
            }
        },
        ScalarType::Symbol => match value {
            Value::Symbol(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::String(s) => {
                let id = xlog_core::symbol::intern(s);
                out.extend_from_slice(&id.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected symbol/string value for symbol column".to_string(),
                ))
            }
        },
    }
    Ok(())
}

fn compile_sampling_plan(
    prob_facts: &[ProbFact],
    annotated_disjunctions: &[AnnotatedDisjunction],
) -> Result<(Vec<f32>, Vec<ProbFactSpec>, Vec<AdSpec>)> {
    let mut probs: Vec<f32> = Vec::new();
    let mut fact_specs: Vec<ProbFactSpec> = Vec::new();
    let mut ad_specs: Vec<AdSpec> = Vec::new();

    for pf in prob_facts {
        validate_prob(pf.prob, "probabilistic fact")?;
        let atom = atom_key_from_ground_atom(&pf.atom)?;
        let var_idx = probs.len();
        probs.push(pf.prob as f32);
        fact_specs.push(ProbFactSpec { var_idx, atom });
    }

    for ad in annotated_disjunctions {
        if ad.choices.is_empty() {
            return Err(XlogError::Compilation(
                "Annotated disjunction must contain at least one choice".to_string(),
            ));
        }

        let mut choice_atoms: Vec<GroundAtom> = Vec::with_capacity(ad.choices.len());
        let mut choice_probs: Vec<f64> = Vec::with_capacity(ad.choices.len());
        for pf in &ad.choices {
            validate_prob(pf.prob, "annotated disjunction choice")?;
            choice_atoms.push(atom_key_from_ground_atom(&pf.atom)?);
            choice_probs.push(pf.prob);
        }

        let sum: f64 = choice_probs.iter().copied().sum();
        let eps = 1e-12;
        if sum > 1.0 + eps {
            return Err(XlogError::Compilation(format!(
                "Annotated disjunction probabilities sum to {} (> 1.0)",
                sum
            )));
        }

        let has_none = (1.0 - sum) > eps;
        let mut probs_full: Vec<f64> = choice_probs.clone();
        if has_none {
            probs_full.push((1.0 - sum).max(0.0));
        }

        // Encode categorical choice as a chain of Bernoulli decisions (same as provenance lowering).
        let m = probs_full.len();
        let mut decision_vars: Vec<usize> = Vec::new();
        if m > 1 {
            let mut remaining = 1.0f64;
            for i in 0..(m - 1) {
                let p_i = probs_full[i];
                let cond_true = if remaining <= 0.0 {
                    0.0
                } else {
                    p_i / remaining
                };
                validate_prob(cond_true, "annotated disjunction conditional")?;
                probs.push(cond_true as f32);
                decision_vars.push(probs.len() - 1);
                remaining -= p_i;
            }
        }

        ad_specs.push(AdSpec {
            decision_vars,
            choices: choice_atoms,
            has_none,
        });
    }

    Ok((probs, fact_specs, ad_specs))
}

/// Why evidence may or may not be forceable to root Bernoulli variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForceabilityReason {
    AllForceable,
    ContainsDerivedEvidence,
    ContainsNegativeAdHeadEvidence,
    NoEvidence,
}

/// Compiled evidence forcing for the MC sampler.
#[derive(Debug, Clone)]
pub struct EvidenceForcing {
    pub force_mask: Vec<u8>,
    pub forced_value: Vec<u8>,
    pub forceable: bool,
    pub reason: ForceabilityReason,
}

#[cfg(feature = "host-io")]
fn build_scc_plans(program: &Program) -> Result<Vec<SccPlan>> {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs_for_lowering(&graph);

    let mut rules_by_head: HashMap<String, Vec<Rule>> = HashMap::new();
    for rule in program.proper_rules() {
        rules_by_head
            .entry(rule.head.predicate.clone())
            .or_default()
            .push(rule.clone());
    }

    let mut plans: Vec<SccPlan> = Vec::new();
    for scc in sccs {
        let mut scc_rules: Vec<Rule> = Vec::new();
        for pred in &scc {
            if let Some(rules) = rules_by_head.get(pred) {
                scc_rules.extend(rules.iter().cloned());
            }
        }
        if scc_rules.is_empty() {
            continue;
        }

        let predicate_set: HashSet<String> = scc.iter().cloned().collect();

        let nonmonotone = scc_rules.iter().any(|rule| {
            rule.body.iter().any(|lit| match lit {
                BodyLiteral::Negated(atom) => predicate_set.contains(&atom.predicate),
                BodyLiteral::Positive(atom) if rule.has_aggregation() => {
                    predicate_set.contains(&atom.predicate)
                }
                _ => false,
            })
        });

        let kind = if nonmonotone {
            SccKind::NonMonotone
        } else if is_recursive_scc(&scc, &scc_rules) {
            SccKind::MonotoneRecursive
        } else {
            SccKind::MonotoneNonRecursive
        };

        plans.push(SccPlan {
            predicates: scc,
            rules: scc_rules,
            kind,
        });
    }

    Ok(plans)
}

#[cfg(feature = "host-io")]
fn is_recursive_scc(scc: &[String], rules: &[Rule]) -> bool {
    if scc.len() > 1 {
        return true;
    }
    let Some(only) = scc.first() else {
        return false;
    };
    for rule in rules {
        for lit in &rule.body {
            if let BodyLiteral::Positive(atom) = lit {
                if &atom.predicate == only {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(feature = "host-io")]
fn evaluate_program_inplace(
    scc_plans: &[SccPlan],
    store: &mut HashMap<String, Relation>,
    max_nonmonotone_iterations: usize,
) -> Result<EvalStats> {
    let mut stats = EvalStats::default();

    for plan in scc_plans {
        match plan.kind {
            SccKind::MonotoneNonRecursive => {
                for rule in &plan.rules {
                    let derived = eval_rule(rule, store, &HashMap::new(), None)?;
                    let rel = store.entry(rule.head.predicate.clone()).or_default();
                    for tuple in derived {
                        rel.insert_tuple(tuple);
                    }
                }
            }
            SccKind::MonotoneRecursive => {
                eval_monotone_recursive_scc(&plan.predicates, &plan.rules, store)?;
            }
            SccKind::NonMonotone => {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = eval_nonmonotone_scc(
                    &plan.predicates,
                    &plan.rules,
                    store,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            }
        }
    }

    Ok(stats)
}

#[cfg(feature = "host-io")]
fn eval_monotone_recursive_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
) -> Result<()> {
    const MAX_ITERS: usize = 1024;

    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    let mut full: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        let rel = store.get(pred).cloned().unwrap_or_default();
        full.insert(pred.clone(), rel);
    }

    let mut delta: HashMap<String, Relation> = HashMap::new();
    for rule in rules {
        let derived = eval_rule(rule, store, &full, None)?;
        if derived.is_empty() {
            continue;
        }
        let head = rule.head.predicate.clone();
        let delta_rel = delta.entry(head.clone()).or_default();
        let full_rel = full.entry(head).or_default();
        for tuple in derived {
            if full_rel.tuples.insert(tuple.clone()) {
                delta_rel.insert_tuple(tuple);
            }
        }
    }

    let mut reached_fixpoint = false;
    for _ in 0..MAX_ITERS {
        let any_delta = delta.values().any(|r| !r.is_empty());
        if !any_delta {
            reached_fixpoint = true;
            break;
        }

        let full_prev = full.clone();
        let delta_prev = delta.clone();
        delta.clear();

        for rule in rules {
            let body_indices: Vec<usize> = rule
                .body
                .iter()
                .enumerate()
                .filter_map(|(i, lit)| match lit {
                    BodyLiteral::Positive(atom) if scc_set.contains(atom.predicate.as_str()) => {
                        let non_empty = delta_prev
                            .get(&atom.predicate)
                            .map(|r| !r.is_empty())
                            .unwrap_or(false);
                        non_empty.then_some(i)
                    }
                    _ => None,
                })
                .collect();
            if body_indices.is_empty() {
                continue;
            }

            let mut derived_all: HashSet<Vec<Value>> = HashSet::new();
            for idx in body_indices {
                let derived = eval_rule(rule, store, &full_prev, Some((idx, &delta_prev)))?;
                derived_all.extend(derived);
            }
            if derived_all.is_empty() {
                continue;
            }

            let head = rule.head.predicate.clone();
            let delta_rel = delta.entry(head.clone()).or_default();
            let full_rel = full.entry(head).or_default();
            for tuple in derived_all {
                if full_rel.tuples.insert(tuple.clone()) {
                    delta_rel.insert_tuple(tuple);
                }
            }
        }
    }

    if !reached_fixpoint {
        return Err(XlogError::Execution(format!(
            "Monotone SCC fixpoint iteration limit ({}) exceeded for SCC {:?}",
            MAX_ITERS, scc
        )));
    }

    for (pred, rel) in full {
        store.insert(pred, rel);
    }
    Ok(())
}

#[cfg(feature = "host-io")]
fn eval_nonmonotone_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    max_iters: usize,
) -> Result<(bool, bool)> {
    let mut base: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        base.insert(pred.clone(), store.get(pred).cloned().unwrap_or_default());
    }

    let mut history: Vec<HashMap<String, Relation>> = Vec::new();
    history.push(base.clone());

    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    seen.entry(hash_scc_state(&history[0])).or_default().push(0);

    for _iter in 0..max_iters {
        let current = history.last().expect("history non-empty").clone();

        let mut next: HashMap<String, Relation> = base.clone();

        for rule in rules {
            let derived = eval_rule(rule, store, &current, None)?;
            let rel = next.entry(rule.head.predicate.clone()).or_default();
            for tuple in derived {
                rel.insert_tuple(tuple);
            }
        }

        if next == current {
            for (pred, rel) in next {
                store.insert(pred, rel);
            }
            return Ok((false, false));
        }

        let h = hash_scc_state(&next);
        if let Some(candidates) = seen.get(&h) {
            for &idx in candidates {
                if history.get(idx) == Some(&next) {
                    // Cycle detected: history[idx..] repeats.
                    let cycle_states = &history[idx..];
                    let final_state = intersect_states(cycle_states);
                    for (pred, rel) in final_state {
                        store.insert(pred, rel);
                    }
                    return Ok((true, false));
                }
            }
        }

        let next_index = history.len();
        history.push(next);
        seen.entry(h).or_default().push(next_index);
    }

    // Iteration budget exhausted: fall back to a conservative invariant set.
    let final_state = intersect_states(&history);
    for (pred, rel) in final_state {
        store.insert(pred, rel);
    }
    Ok((false, true))
}

#[cfg(feature = "host-io")]
fn intersect_states(states: &[HashMap<String, Relation>]) -> HashMap<String, Relation> {
    let mut out: HashMap<String, Relation> = HashMap::new();
    let Some(first) = states.first() else {
        return out;
    };

    for (pred, rel0) in first {
        let mut intersection: HashSet<Vec<Value>> = rel0.tuples.clone();
        for state in &states[1..] {
            if let Some(rel) = state.get(pred) {
                intersection.retain(|t| rel.tuples.contains(t));
            } else {
                intersection.clear();
                break;
            }
        }
        out.insert(
            pred.clone(),
            Relation {
                tuples: intersection,
            },
        );
    }

    out
}

#[cfg(feature = "host-io")]
fn hash_scc_state(state: &HashMap<String, Relation>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    let mut preds: Vec<&String> = state.keys().collect();
    preds.sort();
    for pred in preds {
        pred.hash(&mut hasher);
        if let Some(rel) = state.get(pred) {
            let mut tuple_hashes: Vec<u64> = Vec::with_capacity(rel.tuples.len());
            for tuple in &rel.tuples {
                let mut th = DefaultHasher::new();
                tuple.hash(&mut th);
                tuple_hashes.push(th.finish());
            }
            tuple_hashes.sort_unstable();
            tuple_hashes.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Evaluate a single rule and produce derived head tuples.
///
/// `full_scc` is a per-SCC snapshot (used for SCC predicates), and `delta_scc` optionally provides
/// a delta relation for a specific body literal index (semi-naive).
#[cfg(feature = "host-io")]
fn eval_rule(
    rule: &Rule,
    global: &HashMap<String, Relation>,
    full_scc: &HashMap<String, Relation>,
    delta_scc: Option<(usize, &HashMap<String, Relation>)>,
) -> Result<Vec<Vec<Value>>> {
    let mut states: Vec<HashMap<String, Value>> = Vec::new();
    states.push(HashMap::new());

    for (idx, lit) in rule.body.iter().enumerate() {
        let mut next_states: Vec<HashMap<String, Value>> = Vec::new();
        match lit {
            BodyLiteral::Positive(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    for tuple in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            next_states.push(binding2);
                        }
                    }
                }
            }
            BodyLiteral::Negated(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    if negated_atom_holds(atom, rel, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::Comparison(cmp) => {
                for binding in states {
                    if eval_comparison(cmp.op, &cmp.left, &cmp.right, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                for mut binding in states {
                    if binding.contains_key(&is_expr.target) {
                        return Err(XlogError::Compilation(format!(
                            "Is-expression target {} is already bound",
                            is_expr.target
                        )));
                    }
                    let v = eval_arith_expr(&is_expr.expr, &binding)?;
                    binding.insert(is_expr.target.clone(), v);
                    next_states.push(binding);
                }
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }

    if rule.has_aggregation() {
        eval_aggregate_head(&rule.head, states)
    } else {
        let mut out: HashSet<Vec<Value>> = HashSet::new();
        for binding in states {
            let head_tuple = materialize_head_non_aggregate(&rule.head, &binding)?;
            out.insert(head_tuple);
        }
        Ok(out.into_iter().collect())
    }
}

#[cfg(feature = "host-io")]
fn select_relation<'a>(
    atom: &Atom,
    body_index: usize,
    global: &'a HashMap<String, Relation>,
    full_scc: &'a HashMap<String, Relation>,
    delta_scc: Option<(usize, &'a HashMap<String, Relation>)>,
) -> Result<&'a Relation> {
    if let Some((delta_index, delta_map)) = delta_scc {
        if delta_index == body_index {
            return delta_map.get(&atom.predicate).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Missing delta relation for predicate {}",
                    atom.predicate
                ))
            });
        }
    }
    if let Some(rel) = full_scc.get(&atom.predicate) {
        return Ok(rel);
    }
    global
        .get(&atom.predicate)
        .ok_or_else(|| XlogError::Compilation(format!("Unknown predicate {}", atom.predicate)))
}

#[cfg(feature = "host-io")]
fn negated_atom_holds(
    atom: &Atom,
    rel: &Relation,
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    // Safety: all variables in a negated atom must be bound already (otherwise domain is unknown).
    for term in &atom.terms {
        if let Term::Variable(name) = term {
            if !binding.contains_key(name) {
                return Err(XlogError::UnsafeVariable(name.clone()));
            }
        }
    }

    for tuple in &rel.tuples {
        if atom_matches_bound(atom, tuple, binding)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(feature = "host-io")]
fn atom_matches_bound(
    atom: &Atom,
    tuple: &[Value],
    binding: &HashMap<String, Value>,
) -> Result<bool> {
    if atom.terms.len() != tuple.len() {
        return Err(XlogError::Compilation(format!(
            "Arity mismatch for {}: atom has {}, tuple has {}",
            atom.predicate,
            atom.terms.len(),
            tuple.len()
        )));
    }
    for (term, value) in atom.terms.iter().zip(tuple.iter()) {
        match term {
            Term::Variable(name) => {
                let Some(bound) = binding.get(name) else {
                    return Err(XlogError::UnsafeVariable(name.clone()));
                };
                if bound != value {
                    return Ok(false);
                }
            }
            Term::Anonymous => {}
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                if &value_from_term(term)? != value {
                    return Ok(false);
                }
            }
            Term::Aggregate(AggExpr { .. }) => {
                return Err(XlogError::Compilation(
                    "Aggregate not allowed in body atom".to_string(),
                ));
            }
        }
    }
    Ok(true)
}

#[cfg(feature = "host-io")]
fn materialize_head_non_aggregate(
    head: &Atom,
    binding: &HashMap<String, Value>,
) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(head.terms.len());
    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                let v = binding.get(name).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "Unbound head variable {} in {}",
                        name, head.predicate
                    ))
                })?;
                out.push(v.clone());
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(format!(
                    "Anonymous wildcard '_' not allowed in rule head (predicate {})",
                    head.predicate
                )));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                out.push(value_from_term(term)?);
            }
            Term::Aggregate(_) => {
                return Err(XlogError::Compilation(
                    "Aggregate term in non-aggregate rule head".to_string(),
                ));
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
#[cfg(feature = "host-io")]
enum AggState {
    Count(u64),
    SumI128(i128),
    SumF64(f64),
    Min(Option<Value>),
    Max(Option<Value>),
    LogSumExp { max: f64, sumexp: f64, init: bool },
}

#[cfg(feature = "host-io")]
impl AggState {
    fn new(op: AggOp) -> Self {
        match op {
            AggOp::Count => AggState::Count(0),
            AggOp::Sum => AggState::SumI128(0),
            AggOp::Min => AggState::Min(None),
            AggOp::Max => AggState::Max(None),
            AggOp::LogSumExp => AggState::LogSumExp {
                max: f64::NEG_INFINITY,
                sumexp: 0.0,
                init: false,
            },
        }
    }

    fn update(&mut self, op: AggOp, v: &Value) -> Result<()> {
        match op {
            AggOp::Count => match self {
                AggState::Count(c) => {
                    *c = c.saturating_add(1);
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Sum => match self {
                AggState::SumI128(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as i128;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        let f = f64::from_bits(*bits);
                        let acc_f = *acc as f64;
                        *self = AggState::SumF64(acc_f + f);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                AggState::SumF64(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as f64;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        *acc += f64::from_bits(*bits);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Min => match self {
                AggState::Min(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(v, c)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Max => match self {
                AggState::Max(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(c, v)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::LogSumExp => match self {
                AggState::LogSumExp { max, sumexp, init } => {
                    let x = match v {
                        Value::I64(i) => *i as f64,
                        Value::F64(bits) => f64::from_bits(*bits),
                        _ => {
                            return Err(XlogError::Compilation(
                                "logsumexp() aggregate requires numeric input".to_string(),
                            ))
                        }
                    };
                    if x.is_nan() {
                        return Err(XlogError::Compilation(
                            "logsumexp() aggregate encountered NaN".to_string(),
                        ));
                    }
                    if !*init {
                        *max = x;
                        *sumexp = 1.0;
                        *init = true;
                        return Ok(());
                    }
                    if x > *max {
                        *sumexp = *sumexp * (*max - x).exp() + 1.0;
                        *max = x;
                    } else {
                        *sumexp += (x - *max).exp();
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
        }
    }

    fn finish(&self, op: AggOp) -> Result<Value> {
        match (op, self) {
            (AggOp::Count, AggState::Count(c)) => {
                let v: i64 = (*c)
                    .try_into()
                    .map_err(|_| XlogError::Compilation("count() overflowed i64".to_string()))?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumI128(acc)) => {
                let v: i64 = (*acc)
                    .try_into()
                    .map_err(|_| XlogError::Compilation("sum() overflowed i64".to_string()))?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumF64(v)) => Ok(Value::F64(v.to_bits())),
            (AggOp::Min, AggState::Min(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("min() aggregate produced no value".to_string())
            }),
            (AggOp::Max, AggState::Max(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("max() aggregate produced no value".to_string())
            }),
            (AggOp::LogSumExp, AggState::LogSumExp { max, sumexp, init }) => {
                if !*init {
                    return Ok(Value::F64(f64::NEG_INFINITY.to_bits()));
                }
                Ok(Value::F64((max + sumexp.ln()).to_bits()))
            }
            _ => Err(XlogError::Compilation(
                "Internal aggregate state mismatch".to_string(),
            )),
        }
    }
}

#[cfg(feature = "host-io")]
fn value_le(a: &Value, b: &Value) -> Result<bool> {
    match (a, b) {
        (Value::I64(x), Value::I64(y)) => Ok(x <= y),
        (Value::F64(x), Value::F64(y)) => Ok(f64::from_bits(*x) <= f64::from_bits(*y)),
        (Value::Symbol(x), Value::Symbol(y)) => Ok(x <= y),
        (Value::String(x), Value::String(y)) => Ok(x <= y),
        _ => Err(XlogError::Compilation(
            "min/max aggregate requires consistent comparable types".to_string(),
        )),
    }
}

#[cfg(feature = "host-io")]
fn eval_aggregate_head(
    head: &Atom,
    states: Vec<HashMap<String, Value>>,
) -> Result<Vec<Vec<Value>>> {
    // Collect unique group keys (variables) in head order.
    let mut key_vars: Vec<String> = Vec::new();
    let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();

    // Collect unique aggregate specs (op, var) in head order.
    let mut agg_specs: Vec<(AggOp, String)> = Vec::new();
    let mut agg_to_pos: HashMap<(AggOp, String), usize> = HashMap::new();

    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                if !key_var_to_pos.contains_key(name) {
                    let pos = key_vars.len();
                    key_vars.push(name.clone());
                    key_var_to_pos.insert(name.clone(), pos);
                }
            }
            Term::Aggregate(agg) => {
                let key = (agg.op, agg.variable.clone());
                if !agg_to_pos.contains_key(&key) {
                    let pos = agg_specs.len();
                    agg_specs.push(key.clone());
                    agg_to_pos.insert(key, pos);
                }
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(
                    "Anonymous wildcard '_' not allowed in aggregate rule head".to_string(),
                ));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {}
        }
    }

    #[derive(Debug)]
    struct GroupState {
        key: Vec<Value>,
        aggs: Vec<AggState>,
    }

    let mut groups: HashMap<Vec<Value>, GroupState> = HashMap::new();

    for binding in states {
        let mut key: Vec<Value> = Vec::with_capacity(key_vars.len());
        for name in &key_vars {
            let v = binding
                .get(name)
                .ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
            key.push(v.clone());
        }

        let entry = groups.entry(key.clone()).or_insert_with(|| GroupState {
            key,
            aggs: agg_specs.iter().map(|(op, _)| AggState::new(*op)).collect(),
        });

        for (idx, (op, var)) in agg_specs.iter().enumerate() {
            let v = binding
                .get(var)
                .ok_or_else(|| XlogError::UnsafeVariable(var.clone()))?;
            entry.aggs[idx].update(*op, v)?;
        }
    }

    let mut out: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
    for group in groups.values() {
        let mut tuple: Vec<Value> = Vec::with_capacity(head.terms.len());
        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    let pos = *key_var_to_pos.get(name).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Aggregate head variable {} is not a group key",
                            name
                        ))
                    })?;
                    tuple.push(group.key[pos].clone());
                }
                Term::Aggregate(AggExpr { op, variable }) => {
                    let idx = *agg_to_pos
                        .get(&(*op, variable.clone()))
                        .expect("agg_to_pos missing");
                    tuple.push(group.aggs[idx].finish(*op)?);
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    tuple.push(value_from_term(term)?);
                }
                Term::Anonymous => unreachable!(),
            }
        }
        out.push(tuple);
    }

    Ok(out)
}

#[cfg(feature = "host-io")]
fn atom_holds(store: &HashMap<String, Relation>, atom: &GroundAtom) -> bool {
    store
        .get(&atom.predicate)
        .map(|rel| rel.contains(&atom.args))
        .unwrap_or(false)
}

#[cfg(feature = "host-io")]
fn evidence_satisfied(store: &HashMap<String, Relation>, evidence: &[(GroundAtom, bool)]) -> bool {
    for (atom, value) in evidence {
        let holds = atom_holds(store, atom);
        if holds != *value {
            return false;
        }
    }
    true
}

#[cfg(feature = "host-io")]
fn binomial_estimate(k: usize, n: usize, z: f64) -> (f64, f64, f64, f64) {
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n_f = n as f64;
    let p = (k as f64) / n_f;
    let stderr = (p * (1.0 - p) / n_f).sqrt();

    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f) + (z2 / (4.0 * n_f * n_f))).sqrt() / denom;

    let mut lo = center - half;
    let mut hi = center + half;
    if lo < 0.0 {
        lo = 0.0;
    }
    if hi > 1.0 {
        hi = 1.0;
    }
    (p, stderr, lo, hi)
}

// Acklam's inverse normal CDF approximation.
#[cfg(feature = "host-io")]
fn normal_quantile(p: f64) -> f64 {
    if !(0.0 < p && p < 1.0) || p.is_nan() {
        return f64::NAN;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }

    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}
