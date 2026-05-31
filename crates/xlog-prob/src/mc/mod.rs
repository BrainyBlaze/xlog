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
mod resident;
mod results;
mod sampling;

pub use evidence::{EvidenceForcing, ForceabilityReason};
pub use resident::{
    compile_resident_plan, McNoHostStats, McResidentResult, ResidentPlan, ResidentRejectKind,
    ResidentRejection,
};

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
use xlog_logic::ast::{Directives, Evidence, ProbMethod, ProbQuery, Program};
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

impl McSamplingMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            McSamplingMethod::Rejection => "rejection",
            McSamplingMethod::EvidenceClamping => "evidence_clamping",
        }
    }
}

impl From<ProbMethod> for McSamplingMethod {
    fn from(value: ProbMethod) -> Self {
        match value {
            ProbMethod::Rejection => Self::Rejection,
            ProbMethod::EvidenceClamping => Self::EvidenceClamping,
        }
    }
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

impl McEvalConfig {
    pub fn from_directives(directives: &Directives) -> Result<Self> {
        let mut cfg = Self::default();
        if let Some(samples) = directives.prob_samples {
            cfg.samples = samples;
        }
        if let Some(seed) = directives.prob_seed {
            cfg.seed = seed;
        }
        if let Some(confidence) = directives.prob_confidence {
            cfg.confidence = confidence;
        }
        if let Some(iterations) = directives.prob_max_nonmonotone_iterations {
            cfg.max_nonmonotone_iterations = iterations;
        }
        cfg.sampling_method = directives.prob_method.map(McSamplingMethod::from);
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.samples == 0 {
            return Err(XlogError::Compilation(
                "MC inference requires samples > 0".to_string(),
            ));
        }
        if !(0.0 < self.confidence && self.confidence < 1.0) || self.confidence.is_nan() {
            return Err(XlogError::Compilation(format!(
                "MC inference requires 0 < confidence < 1, got {}",
                self.confidence
            )));
        }
        if self.max_nonmonotone_iterations == 0 {
            return Err(XlogError::Compilation(
                "MC inference requires max_nonmonotone_iterations > 0".to_string(),
            ));
        }
        Ok(())
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

/// **Tracked** (data-plane) host<->device transfers observed *inside* the MC
/// sample/evaluate/count hot loop (i.e. between the static setup uploads and the
/// final-count downloads).
///
/// "Tracked" means transfers recorded by the provider's `HostTransferTracker`
/// (`record_htod`/`record_dtoh`) — the data-plane bytes that define GPU-native
/// execution. Consistent with the rest of the XLOG engine, **bounded
/// control-plane metadata** reads (e.g. reading a relation's `num_rows` scalar
/// via `dtoh_scalar_untracked` after a GPU operator sizes its output) are
/// *intentionally not counted*: they are O(4 bytes) scalars, not data-plane
/// downloads, and they are exempted by the same metadata-vs-data-plane contract
/// the deterministic-D2H gate enforces elsewhere. Such metadata reads still
/// occur per sample (e.g. inside dedup / relational operators).
///
/// For a GPU-native MC run every field here must be zero: static *data-plane*
/// setup uploads happen before the loop and final-count downloads happen after
/// it. This struct is the always-on regression guard for that boundary (K1) —
/// the transfer-budget test asserts `is_zero()`, and any future code that
/// smuggles a *tracked* HtoD/DtoH (a real data-plane transfer) into the loop
/// will trip it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McHotLoopTransfers {
    /// Tracked host-to-device calls during the hot loop.
    pub htod_calls: u64,
    /// Tracked device-to-host calls during the hot loop.
    pub dtoh_calls: u64,
    /// Tracked host-to-device bytes during the hot loop.
    pub htod_bytes: u64,
    /// Tracked device-to-host bytes during the hot loop.
    pub dtoh_bytes: u64,
}

impl McHotLoopTransfers {
    /// True when no tracked host/device transfer occurred inside the hot loop.
    pub fn is_zero(&self) -> bool {
        self.htod_calls == 0 && self.dtoh_calls == 0
    }
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
    /// Tracked host/device transfers observed inside the sample/evaluate/count
    /// hot loop. Zero for a GPU-native run (K1).
    pub hot_loop_transfers: McHotLoopTransfers,
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
    /// Tracked host/device transfer delta measured across the hot loop.
    pub(super) hot_loop_transfers: McHotLoopTransfers,
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

    /// Host-facing MC evaluation: runs the GPU-native device loop
    /// ([`Self::evaluate_gpu_device_with_provider`]) and then **materializes the
    /// result on the host** by downloading the final query/evidence counts
    /// *after* the hot loop. The download is a host-result materialization, not a
    /// hot-loop transfer — the GPU-native zero-transfer property (K1) belongs to
    /// the device loop, not to this convenience wrapper. Use
    /// [`Self::evaluate_gpu_device`] when you want device-resident counts with no
    /// host download at all.
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

    /// CPU **oracle / debug** MC path. Downloads the full sampled-bit matrix to
    /// the host and evaluates every sampled world on a host relation store.
    ///
    /// This is intentionally *not* GPU-native: it performs a large DtoH of the
    /// sample matrix and runs the deterministic core on the CPU. It exists solely
    /// as a deterministic, seed-matched oracle for validating the GPU-native
    /// device counts (the GPU sampler is shared, so for the same program/seed the
    /// two paths see identical samples). It must **never** be used as zero-host /
    /// GPU-native release evidence, and the acceptance matrix excludes it and the
    /// tests that call it (`tests/gpu_mc_vs_cpu.rs`, `tests/mc.rs`).
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
                    .htod_sync_copy_into_tracked(&forcing.force_mask, &mut d_force_mask)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to upload force_mask: {}", e))
                    })?;
                provider
                    .htod_sync_copy_into_tracked(&forcing.forced_value, &mut d_forced_value)
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

    /// Alias for [`Self::evaluate`]: GPU device evaluation followed by host-result
    /// materialization (final-count download after the hot loop). The `_gpu`
    /// suffix denotes that the *compute* runs on the GPU — it does **not** imply a
    /// zero-host result, since it returns a host [`McResult`]. For the
    /// device-resident, no-host-download API use [`Self::evaluate_gpu_device`].
    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu(&self, cfg: McEvalConfig) -> Result<McResult> {
        self.evaluate(cfg)
    }

    /// GPU-native device-resident MC evaluation. Returns [`McDeviceResult`] with
    /// counts left on the device (no host download) and the measured
    /// [`McHotLoopTransfers`]. This is the API whose sample/evaluate/count hot
    /// loop is zero-transfer (K1).
    pub fn evaluate_gpu_device(&self, cfg: McEvalConfig) -> Result<McDeviceResult> {
        let provider = Arc::new(self.provider()?);
        self.evaluate_gpu_device_with_provider(cfg, provider)
    }

    /// GPU-native device-resident MC evaluation using a caller-supplied provider
    /// (enables provider/buffer reuse across calls). Static setup uploads happen
    /// before the sample/evaluate/count hot loop; the hot loop performs zero
    /// tracked HtoD/DtoH transfers (verified by `McDeviceResult::hot_loop_transfers`,
    /// gated by `tests/mc_gpu_native.rs`). Counts remain device-resident; the
    /// caller decides whether/when to download them.
    pub fn evaluate_gpu_device_with_provider(
        &self,
        cfg: McEvalConfig,
        provider: Arc<CudaKernelProvider>,
    ) -> Result<McDeviceResult> {
        // The GPU-resident megakernel engine is the sole MC execution path: there
        // is no host-orchestrated per-sample fallback. It evaluates ALL worlds in
        // a single launch with zero host interaction in the measured region, then
        // returns device-resident counts. Programs outside the supported bounded
        // fragment fail closed (typed `ResidentRejection`).
        let r = self.evaluate_resident_with_provider(cfg, provider)?;
        Ok(McDeviceResult {
            query_counts: r.query_counts,
            evidence_count: r.evidence_count,
            total_samples: r.total_samples,
            seed: r.seed,
            confidence: r.confidence,
            // The bounded-domain dense engine evaluates non-monotone fixpoints on
            // device only when accepted; nonmonotone SCC bookkeeping is not part of
            // this engine's reported state.
            nonmonotone_sccs: 0,
            nonmonotone_cycles: 0,
            nonmonotone_iteration_limit_hits: 0,
            sampling_method: r.sampling_method,
            hot_loop_transfers: McHotLoopTransfers {
                htod_calls: r.no_host.tracked_htod_calls,
                dtoh_calls: r.no_host.tracked_dtoh_calls,
                htod_bytes: 0,
                dtoh_bytes: 0,
            },
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
        buffers::augment_schemas_for_program(&expanded, &mut schemas)?;
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
                .htod_sync_copy_into_tracked(&forcing.force_mask, &mut d_force_mask)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload force_mask: {}", e)))?;
            provider
                .htod_sync_copy_into_tracked(&forcing.forced_value, &mut d_forced_value)
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

        // K1 boundary: snapshot tracked host<->device transfers immediately before
        // the sample/evaluate/count hot loop. Everything above (sampler upload, force
        // arrays, deterministic facts, prob/AD tables, reset plan) is static setup and
        // is allowed to transfer. From here until the loop ends, a GPU-native run must
        // perform *zero* tracked HtoD/DtoH transfers.
        let transfers_pre = provider.host_transfer_stats();

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

        // K1 boundary: sync so all hot-loop kernels and any (disallowed) transfers
        // have been issued, then read the post-loop transfer snapshot. The tracked
        // counters are incremented synchronously at each Rust transfer call site, so
        // the delta is exact regardless of kernel-launch asynchrony.
        provider.device().synchronize()?;
        let transfers_post = provider.host_transfer_stats();
        stats.hot_loop_transfers = McHotLoopTransfers {
            htod_calls: transfers_post
                .htod_calls
                .saturating_sub(transfers_pre.htod_calls),
            dtoh_calls: transfers_post
                .dtoh_calls
                .saturating_sub(transfers_pre.dtoh_calls),
            htod_bytes: transfers_post
                .htod_bytes
                .saturating_sub(transfers_pre.htod_bytes),
            dtoh_bytes: transfers_post
                .dtoh_bytes
                .saturating_sub(transfers_pre.dtoh_bytes),
        };

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
                        BodyLiteral::Comparison(_)
                        | BodyLiteral::IsExpr(_)
                        | BodyLiteral::Univ(_) => {}
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
