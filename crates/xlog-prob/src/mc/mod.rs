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
use std::sync::Arc;

#[cfg(feature = "host-io")]
use cudarc::driver::DeviceSlice;
use cudarc::driver::LaunchConfig;
use xlog_core::{MemoryBudget, RelId, Result, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_eval_kernels, MC_EVAL_MODULE};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager, LaunchAsync};
#[cfg(feature = "host-io")]
use xlog_logic::ast::{BodyLiteral, Rule};
use xlog_logic::ast::{Evidence, ProbQuery, Program};
use xlog_logic::compile::Compiler;
use xlog_logic::stratify::analyze_stratification;
use xlog_runtime::Executor;

use crate::exact::GpuConfig;
#[cfg(feature = "host-io")]
use crate::provenance::Value;
use crate::provenance::{atom_key_from_ground_atom, GroundAtom};

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
#[non_exhaustive]
/// Configuration for Monte Carlo probabilistic inference.
///
/// Use [`McEvalConfig::default()`] as a starting point and then update the
/// individual fields you need.
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
pub(super) struct Relation {
    pub(super) tuples: HashSet<Vec<Value>>,
}

#[cfg(feature = "host-io")]
impl Relation {
    pub(super) fn insert_tuple(&mut self, tuple: Vec<Value>) {
        self.tuples.insert(tuple);
    }

    pub(super) fn contains(&self, tuple: &[Value]) -> bool {
        self.tuples.contains(tuple)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone)]
pub(super) enum SccKind {
    MonotoneNonRecursive,
    MonotoneRecursive,
    NonMonotone,
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone)]
pub(super) struct SccPlan {
    pub(super) predicates: Vec<String>,
    pub(super) rules: Vec<Rule>,
    pub(super) kind: SccKind,
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

        let z = results::normal_quantile(0.5 + cfg.confidence / 2.0);
        let mut query_estimates: Vec<McQueryEstimate> = Vec::with_capacity(self.queries.len());
        for (i, atom) in self.queries.iter().enumerate() {
            let k = host_counts.get(i).copied().unwrap_or(0) as usize;
            let (p, stderr, ci_low, ci_high) = results::binomial_estimate(k, evidence_samples, z);
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
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to upload force_mask: {}", e))
                    })?;
                provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&forcing.forced_value, &mut d_forced_value)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to upload forced_value: {}", e))
                    })?;
            } else {
                provider
                    .device()
                    .inner()
                    .memset_zeros(&mut d_force_mask)
                    .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
                provider
                    .device()
                    .inner()
                    .memset_zeros(&mut d_forced_value)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero forced_value: {}", e))
                    })?;
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

            let sample_stats = results::evaluate_program_inplace(
                &self.scc_plans,
                &mut store,
                cfg.max_nonmonotone_iterations,
            )?;
            stats.nonmonotone_sccs += sample_stats.nonmonotone_sccs;
            stats.nonmonotone_cycles += sample_stats.nonmonotone_cycles;
            stats.nonmonotone_iteration_limit_hits += sample_stats.nonmonotone_iteration_limit_hits;

            // In clamped mode, skip evidence check — all samples count
            if !is_clamped && !results::evidence_satisfied(&store, &self.evidence) {
                continue;
            }

            n_evidence += 1;
            for (i, q) in self.queries.iter().enumerate() {
                if results::atom_holds(&store, q) {
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

        let z = results::normal_quantile(0.5 + cfg.confidence / 2.0);

        let mut query_estimates: Vec<McQueryEstimate> = Vec::with_capacity(self.queries.len());
        for (i, atom) in self.queries.iter().enumerate() {
            let k = n_query_true[i];
            let (p, stderr, ci_low, ci_high) = results::binomial_estimate(k, denom, z);
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

        let stats = self.evaluate_gpu_counts_with(
            &cfg,
            &forcing,
            method,
            provider.clone(),
            |executor, plan, count| {
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
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
            },
        )?;

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
        let reset_plan = buffers::build_sample_reset_plan(&gpu_plan, self, &provider, &executor)?;

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
            provider
                .device()
                .inner()
                .memset_zeros(&mut d_force_mask)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
            provider
                .device()
                .inner()
                .memset_zeros(&mut d_forced_value)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;
        }

        let mc_profile = std::env::var("XLOG_MC_PROFILE")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false);
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
                        BodyLiteral::Epistemic(lit) => {
                            referenced.insert(lit.atom.predicate.clone());
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
        let scc_plans = results::build_scc_plans(program)?;

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
}
