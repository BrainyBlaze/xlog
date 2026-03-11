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

mod buffers;
mod evidence;
mod results;
mod sampling;

pub use evidence::{EvidenceForcing, ForceabilityReason};

use std::collections::{HashMap, HashSet};
#[cfg(feature = "host-io")]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use xlog_core::{MemoryBudget, RelId, Result, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_eval_kernels, MC_EVAL_MODULE};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
#[cfg(feature = "host-io")]
use xlog_logic::ast::AggExpr;
#[cfg(feature = "host-io")]
use xlog_logic::ast::Rule;
use xlog_logic::ast::{
    Evidence, ProbQuery, Program,
};
use xlog_logic::compile::Compiler;
use xlog_logic::stratify::analyze_stratification;
#[cfg(feature = "host-io")]
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering};
use xlog_runtime::Executor;

use crate::exact::GpuConfig;
use crate::provenance::{atom_key_from_ground_atom, GroundAtom};
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

/// Strategy for counting evidence-satisfied samples in the MC loop.
///
/// In `QueriesOnly` mode (used with evidence clamping), evidence is
/// guaranteed to hold in every sample, so we skip the truth-kernel's
/// evidence check and evidence-side buffer allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McCountStrategy {
    /// Full path: check both queries and evidence each sample.
    QueriesAndEvidence,
    /// Clamped path: evidence is always satisfied; only accumulate query flags.
    QueriesOnly,
}

impl McCountStrategy {
    /// Derive the count strategy from the chosen sampling method.
    pub fn from_method(method: McSamplingMethod) -> Self {
        match method {
            McSamplingMethod::Rejection => Self::QueriesAndEvidence,
            McSamplingMethod::EvidenceClamping => Self::QueriesOnly,
        }
    }
}

/// Breakdown of time spent in each phase of MC evaluation.
/// Gate with `XLOG_MC_PROFILE=1` to print at the end of evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McTimingBreakdown {
    pub sampler_us: u64,
    pub sample_reset_us: u64,
    pub sample_build_us: u64,
    pub eval_us: u64,
    pub count_us: u64,
}

impl McTimingBreakdown {
    pub fn total_us(&self) -> u64 {
        self.sampler_us
            .saturating_add(self.sample_reset_us)
            .saturating_add(self.sample_build_us)
            .saturating_add(self.eval_us)
            .saturating_add(self.count_us)
    }
}

/// Phase 4 semantics for non-monotone SCC evaluation inside MC sampling.
pub const NONMONOTONE_SEMANTICS: &str = "Synchronous iteration per SCC; if a fixpoint is reached, use it; if a cycle is detected, use the intersection of all states in the cycle (skeptical tuples only); if the iteration budget is exceeded, use the intersection across all visited states (conservative).";

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
    /// Sampling method override. `None` = auto-select (EvidenceClamping when forceable, Rejection otherwise).
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
pub(super) struct ProbFactSpec {
    pub(super) var_idx: usize,
    pub(super) atom: GroundAtom,
}

#[derive(Debug, Clone)]
pub(super) struct AdSpec {
    pub(super) decision_vars: Vec<usize>,
    pub(super) choices: Vec<GroundAtom>,
    pub(super) has_none: bool,
}

pub(super) struct GpuMcPlan {
    pub(super) program: Program,
    pub(super) plan: xlog_ir::ExecutionPlan,
    pub(super) schemas: HashMap<String, Schema>,
    pub(super) rel_ids: HashMap<String, RelId>,
    pub(super) query_rel_names: Vec<String>,
    pub(super) evidence_rel_specs: Vec<(String, bool)>,
    pub(super) nonmonotone_sccs: HashSet<usize>,
}

pub(super) struct ProbTableDevice {
    pub(super) predicate: String,
    pub(super) buffer: CudaBuffer,
    pub(super) var_idx: TrackedCudaSlice<u32>,
}

pub(super) struct AdDecisionDevice {
    pub(super) decision_vars: TrackedCudaSlice<u32>,
}

pub(super) struct AdTableDevice {
    pub(super) predicate: String,
    pub(super) buffer: CudaBuffer,
    pub(super) decision_offsets: TrackedCudaSlice<u32>,
    pub(super) decision_lengths: TrackedCudaSlice<u32>,
    pub(super) choice_positions: TrackedCudaSlice<u32>,
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
pub(super) struct EvalStats {
    pub(super) nonmonotone_sccs: usize,
    pub(super) nonmonotone_cycles: usize,
    pub(super) nonmonotone_iteration_limit_hits: usize,
}

#[derive(Clone)]
pub struct McProgram {
    pub(super) gpu_config: GpuConfig,
    pub(super) program: Program,
    #[cfg(feature = "host-io")]
    pub(super) base_store: HashMap<String, Relation>,
    #[cfg(feature = "host-io")]
    pub(super) scc_plans: Vec<SccPlan>,
    pub(super) queries: Vec<GroundAtom>,
    pub(super) evidence: Vec<(GroundAtom, bool)>,
    pub(super) bernoulli_probs: Vec<f32>,
    pub(super) prob_facts: Vec<ProbFactSpec>,
    pub(super) annotated_disjunctions: Vec<AdSpec>,
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

        if device_result.sampling_method != McSamplingMethod::EvidenceClamping
            && !self.evidence.is_empty()
            && evidence_samples == 0
        {
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
            sampling_method: device_result.sampling_method,
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

        let (method, forcing) = self.resolve_sampling_method(cfg.sampling_method)?;
        let is_clamped = method == McSamplingMethod::EvidenceClamping;

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

            // Allocate force arrays: upload actual forcing data in clamped mode, zero-fill otherwise
            let mut d_force_mask = provider.memory().alloc::<u8>(num_vars.max(1))?;
            let mut d_forced_value = provider.memory().alloc::<u8>(num_vars.max(1))?;
            if is_clamped {
                provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&forcing.force_mask, &mut d_force_mask)
                    .map_err(|e| XlogError::Kernel(format!("Failed to upload force_mask: {}", e)))?;
                provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&forcing.forced_value, &mut d_forced_value)
                    .map_err(|e| XlogError::Kernel(format!("Failed to upload forced_value: {}", e)))?;
            } else {
                provider.device().inner().memset_zeros(&mut d_force_mask)
                    .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
                provider.device().inner().memset_zeros(&mut d_forced_value)
                    .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;
            }

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

            // In clamped mode, skip evidence check — all samples count
            if !is_clamped && !evidence_satisfied(&store, &self.evidence) {
                continue;
            }

            n_evidence += 1;
            for (i, q) in self.queries.iter().enumerate() {
                if atom_holds(&store, q) {
                    n_query_true[i] += 1;
                }
            }
        }

        if !is_clamped && !self.evidence.is_empty() && n_evidence == 0 {
            return Err(XlogError::Execution(format!(
                "MC inference error: evidence was never satisfied across {} samples (seed={})",
                cfg.samples, cfg.seed
            )));
        }

        // If there is no evidence (or clamped mode), treat all samples as evidence-satisfying.
        let denom = if self.evidence.is_empty() || is_clamped {
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
            sampling_method: method,
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
        let (method, forcing) = self.resolve_sampling_method(cfg.sampling_method)?;
        let strategy = McCountStrategy::from_method(method);

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
        // In QueriesOnly mode, pass 0 evidence to the truth kernel so it sets evidence_ok = 1
        let effective_evidence_count_u32 = match strategy {
            McCountStrategy::QueriesOnly => 0u32,
            McCountStrategy::QueriesAndEvidence => evidence_count_u32,
        };

        let mut d_query_counts = provider.memory().alloc::<u32>(prob_query_count)?;
        if prob_query_count > 0 {
            provider
                .device()
                .inner()
                .memset_zeros(&mut d_query_counts)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero MC query counts: {}", e)))?;
        }
        let mut d_evidence_count = provider.memory().alloc::<u32>(1)?;
        // Design note (QueriesOnly mode): The spec requires evidence_count ==
        // cfg.samples at the end of inference.  We achieve this without a
        // separate HtoD upload: the truth kernel always sets evidence_ok = 1
        // (because effective_evidence_count = 0 ⇒ all evidence trivially
        // satisfied), so the accumulate kernel atomicAdd's evidence_count once
        // per sample, arriving at exactly cfg.samples after the loop.
        provider
            .device()
            .inner()
            .memset_zeros(&mut d_evidence_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero MC evidence count: {}", e)))?;

        let mut d_query_flags = provider.memory().alloc::<u8>(prob_query_count)?;
        let mut d_evidence_ok = provider.memory().alloc::<u8>(1)?;
        let mut d_query_ptrs = provider.memory().alloc::<u64>(prob_query_count)?;
        let mut d_zero_count = provider.memory().alloc::<u32>(1)?;
        provider
            .device()
            .inner()
            .memset_zeros(&mut d_zero_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero MC zero-count: {}", e)))?;

        // In QueriesOnly mode, skip evidence-side buffer allocations.
        // We still need 1-element sentinel slices for the truth kernel args.
        let evidence_alloc_count = match strategy {
            McCountStrategy::QueriesOnly => 1,
            McCountStrategy::QueriesAndEvidence => evidence_count.max(1),
        };
        let mut d_evidence_ptrs = provider.memory().alloc::<u64>(evidence_alloc_count)?;
        let mut d_evidence_expected = provider.memory().alloc::<u8>(evidence_alloc_count)?;

        if evidence_count > 0 && strategy == McCountStrategy::QueriesAndEvidence {
            let expected: Vec<u8> = self
                .evidence
                .iter()
                .map(|(_, v)| if *v { 1u8 } else { 0u8 })
                .collect();
            buffers::upload_slice(
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

        // Pre-allocate host-side pointer vectors outside the per-sample closure
        // to avoid repeated heap allocation.  Query/evidence relations are dynamic
        // (re-created each sample), so the device pointers themselves are NOT
        // stable -- we still upload every sample -- but the host Vec storage is
        // reused across iterations.
        let mut query_ptrs_buf: Vec<u64> = vec![0u64; prob_query_count];
        let mut evidence_ptrs_buf: Vec<u64> = vec![0u64; evidence_count];

        let stats =
            self.evaluate_gpu_counts_with(&cfg, &forcing, method, provider.clone(), |executor, plan, count| {
                let zero_ptr = *d_zero_count.device_ptr() as u64;

                query_ptrs_buf.clear();
                for rel_name in plan.query_rel_names.iter().take(count) {
                    let ptr = executor
                        .store()
                        .get(rel_name)
                        .map(|buf| *buf.num_rows_device().device_ptr() as u64)
                        .unwrap_or(zero_ptr);
                    query_ptrs_buf.push(ptr);
                }
                buffers::upload_slice(
                    &provider,
                    &query_ptrs_buf,
                    &mut d_query_ptrs,
                    "MC query count ptrs",
                )?;

                if strategy == McCountStrategy::QueriesAndEvidence {
                    evidence_ptrs_buf.clear();
                    for (rel_name, _) in plan.evidence_rel_specs.iter() {
                        let ptr = executor
                            .store()
                            .get(rel_name)
                            .map(|buf| *buf.num_rows_device().device_ptr() as u64)
                            .unwrap_or(zero_ptr);
                        evidence_ptrs_buf.push(ptr);
                    }
                    buffers::upload_slice(
                        &provider,
                        &evidence_ptrs_buf,
                        &mut d_evidence_ptrs,
                        "MC evidence count ptrs",
                    )?;
                }

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
                                effective_evidence_count_u32,
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
            sampling_method: method,
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

        buffers::ensure_predicate_decls(&mut plan_program)?;

        let max_recursion = plan_program.directives.max_recursion_depth.unwrap_or(100);
        let expanded = xlog_logic::expand_program_functions(&plan_program, max_recursion)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&expanded)?;
        let mut schemas = compiler.schemas().clone();
        buffers::augment_schemas_for_program(&expanded, &mut schemas);
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
        forcing: &EvidenceForcing,
        method: McSamplingMethod,
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

        buffers::load_deterministic_facts(
            &gpu_plan.program,
            &gpu_plan.schemas,
            &provider,
            &mut executor,
        )?;

        let (prob_tables, ad_tables, ad_decisions) =
            buffers::build_prob_tables_device(self, &provider, &gpu_plan.schemas)?;

        // Build the targeted reset plan: preserve pure deterministic base
        // relations, clear everything else, and snapshot base facts for
        // predicates that are both deterministic and dynamic.
        let reset_plan =
            buffers::build_sample_reset_plan(&gpu_plan, self, &provider, &executor)?;

        let num_vars = self.bernoulli_probs.len();

        // Allocate force arrays: upload actual forcing data in clamped mode, zero-fill otherwise
        let mut d_force_mask = provider.memory().alloc::<u8>(num_vars.max(1))?;
        let mut d_forced_value = provider.memory().alloc::<u8>(num_vars.max(1))?;
        if method == McSamplingMethod::EvidenceClamping && num_vars > 0 {
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&forcing.force_mask, &mut d_force_mask)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload force_mask: {}", e)))?;
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&forcing.forced_value, &mut d_forced_value)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload forced_value: {}", e)))?;
        } else {
            provider.device().inner().memset_zeros(&mut d_force_mask)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
            provider.device().inner().memset_zeros(&mut d_forced_value)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;
        }

        let mc_profile = std::env::var("XLOG_MC_PROFILE").ok().map(|v| v == "1").unwrap_or(false);
        let mut timing = McTimingBreakdown::default();

        let t0 = std::time::Instant::now();
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
        if mc_profile {
            timing.sampler_us = t0.elapsed().as_micros() as u64;
        }

        let mut stats = EvalStats::default();

        // Pre-compute reference slices for the reset plan (avoids per-sample allocation).
        let preserve_refs: Vec<&str> = reset_plan.preserve.iter().map(|s| s.as_str()).collect();
        let clear_refs: Vec<(&str, Schema)> = reset_plan
            .clear
            .iter()
            .map(|(s, sc)| (s.as_str(), sc.clone()))
            .collect();

        for sample_idx in 0..cfg.samples {
            let t_reset = std::time::Instant::now();
            executor.reset_for_mc_relations(&preserve_refs, &clear_refs)?;
            // Re-load deterministic base facts for predicates that are both
            // deterministic and dynamic (e.g., `a(1). 0.5::a(2).`).
            for (pred, base_buf) in &reset_plan.reload_base {
                let cloned = if base_buf.is_empty() {
                    provider.create_empty_buffer(base_buf.schema().clone())?
                } else {
                    buffers::clone_buffer_device(&provider, base_buf)?
                };
                executor.put_relation(pred, cloned);
            }
            if mc_profile {
                timing.sample_reset_us += t_reset.elapsed().as_micros() as u64;
            }

            let start = sample_idx * num_vars;
            let end = start + num_vars;
            let sample_bits = samples_device.slice(start..end);

            let t_build = std::time::Instant::now();
            let sample_buffers = buffers::build_sample_buffers(
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
                    _ => buffers::dedup_relation(&provider, &buf)?,
                };
                executor.put_relation(&pred, merged);
            }
            if mc_profile {
                timing.sample_build_us += t_build.elapsed().as_micros() as u64;
            }

            let t_eval = std::time::Instant::now();
            let sample_stats = sampling::evaluate_program_gpu(
                &provider,
                &mut executor,
                &gpu_plan.plan,
                &gpu_plan.nonmonotone_sccs,
                cfg.max_nonmonotone_iterations,
            )?;
            if mc_profile {
                timing.eval_us += t_eval.elapsed().as_micros() as u64;
            }
            stats.nonmonotone_sccs += sample_stats.nonmonotone_sccs;
            stats.nonmonotone_cycles += sample_stats.nonmonotone_cycles;
            stats.nonmonotone_iteration_limit_hits += sample_stats.nonmonotone_iteration_limit_hits;

            let t_count = std::time::Instant::now();
            on_sample(&executor, &gpu_plan, prob_query_count)?;
            if mc_profile {
                timing.count_us += t_count.elapsed().as_micros() as u64;
            }
        }

        if mc_profile {
            eprintln!(
                "[MC Profile] samples={} sampler={}us reset={}us build={}us eval={}us count={}us total={}us",
                cfg.samples, timing.sampler_us, timing.sample_reset_us, timing.sample_build_us,
                timing.eval_us, timing.count_us, timing.total_us()
            );
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
        buffers::extend_prob_facts_with_coin(program, &mut prob_facts)?;
        let (bernoulli_probs, prob_facts, annotated_disjunctions) =
            buffers::compile_sampling_plan(&prob_facts, &program.annotated_disjunctions)?;

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
