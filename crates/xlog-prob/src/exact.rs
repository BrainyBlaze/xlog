//! Exact probabilistic inference via GPU-native Decision-DNNF knowledge compilation
//! and weighted model counting.

#[cfg(feature = "host-io")]
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "host-io")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::LaunchConfig;
use xlog_core::{MemoryBudget, Result, ScalarType, XlogError};
use xlog_cuda::LaunchAsync;
use xlog_logic::ast::Program;

use crate::compilation::gpu_cache::{
    GpuCircuitCache, GpuCircuitCacheConfig, GpuCircuitCacheHandle,
};
use crate::compilation::gpu_cnf::GpuCnfVarTables;
#[cfg(feature = "host-io")]
use crate::compilation::gpu_weights::map_nodes_to_vars_gpu;
use crate::compilation::gpu_weights::{build_evidence_by_var_gpu, build_weights_gpu};
use crate::compilation::{
    compile_gpu_d4_and_verify_cached_with_ledger, encode_cnf_gpu, CircuitCompilationContext,
    CircuitCompilationLedger, CircuitCompileProfile, DeviceRandomVarList, GpuCompileConfig,
    GpuPirGraph, GpuPirRoots,
};
#[cfg(feature = "host-io")]
use crate::logsumexp::{validate_circuit_gradient_values, validate_circuit_value};
use crate::neural_fast_path::{GpuWeightSlots, NeuralFastPathConfig};
use crate::provenance::{
    extract_from_program, extract_from_source, AggregateLiftStatus, GroundAtom, Provenance, Value,
};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{
    arith_kernels, filter_kernels, neural_kernels, weights_kernels, ARITH_MODULE, FILTER_MODULE,
    NEURAL_MODULE, WEIGHTS_MODULE,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

#[derive(Debug, Clone)]
pub struct QueryProbability {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
}

#[derive(Debug, Clone)]
pub struct ExactResult {
    pub log_z_e: f64,
    pub query_probs: Vec<QueryProbability>,
}

#[derive(Debug, Clone)]
pub struct QueryGradients {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
    pub grad_true: Vec<f64>,
    pub grad_false: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ExactResultWithGrads {
    pub log_z_e: f64,
    pub query_grads: Vec<QueryGradients>,
}

#[derive(Debug, Clone)]
struct QuerySpec {
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    atom: GroundAtom,
    var: Option<u32>,
}

fn neural_slot_count_u32(slot_count: usize) -> Result<u32> {
    u32::try_from(slot_count).map_err(|_| {
        XlogError::Compilation(
            "Neural fast-path group slot count exceeds GPU u32 index space".to_string(),
        )
    })
}

fn checked_launch_grid_u32(context: &str, item_count: u32, block_size: u32) -> Result<u32> {
    if block_size == 0 {
        return Err(XlogError::Kernel(format!(
            "{context} launch block size must be non-zero"
        )));
    }
    if item_count == 0 {
        return Ok(0);
    }
    item_count
        .checked_add(block_size - 1)
        .map(|rounded| rounded / block_size)
        .ok_or_else(|| XlogError::Kernel(format!("{context} launch grid overflow")))
}

struct GpuExactState {
    provider: Arc<CudaKernelProvider>,
    cache: Mutex<GpuCircuitCache>,
    handle: GpuCircuitCacheHandle,
    #[cfg(feature = "host-io")]
    circuit_generation: u64,
    #[cfg(feature = "host-io")]
    compilation_ledger: Arc<CircuitCompilationLedger>,
    invalid_reason: OnceLock<String>,
    /// Device-resident batched query-var metadata, keyed by the host vector of
    /// CNF query vars. Lets a warm training loop reuse a single upload instead
    /// of re-uploading (a tracked htod) on every batched force call.
    query_var_batch_cache: Mutex<HashMap<Vec<u32>, Arc<TrackedCudaSlice<u32>>>>,
}

#[cfg(feature = "host-io")]
static NEXT_EXACT_CIRCUIT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Immutable identity of one successfully materialized exact GPU circuit.
///
/// The generation is process-local and allocated only after compilation or
/// cache restoration has produced the handle stored by [`GpuExactState`].
#[cfg(feature = "host-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExactCircuitWitness {
    pub(crate) circuit_generation: u64,
    pub(crate) compiler_invocations: u64,
    pub(crate) materializations: u64,
    pub(crate) disk_cache_restores: u64,
    pub(crate) gpu_cache_hits: u64,
    pub(crate) cache_slot: u32,
}

#[cfg(feature = "host-io")]
fn next_exact_circuit_generation() -> Result<u64> {
    NEXT_EXACT_CIRCUIT_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
            generation.checked_add(1)
        })
        .map_err(|_| {
            XlogError::Compilation(
                "Exact circuit compilation generation counter overflowed".to_string(),
            )
        })
}

/// GPU device selection and memory budget for probabilistic inference.
///
/// Use [`GpuConfig::default()`] and override individual fields as needed.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GpuConfig {
    /// CUDA device ordinal (0-based).
    pub device_ordinal: usize,
    /// Device memory budget in bytes (clamped to available memory at runtime).
    pub memory_bytes: u64,
    /// Host-side Decision-DNNF compiler decision-order hint: renumber leaf/choice
    /// variables by descending structural fanout in the provenance DAG before CNF
    /// encoding, steering the deterministic variable-id tie-breaks of the
    /// (unchanged) GPU-native Decision-DNNF branching heuristic. Query probabilities
    /// are unaffected; only compile-time search shape can differ.
    pub decision_order_hint: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB — clamped to available device memory by GpuMemoryManager at runtime.
            decision_order_hint: false,
        }
    }
}

impl GpuExactState {
    #[cfg(feature = "host-io")]
    fn new(
        provider: Arc<CudaKernelProvider>,
        cache: GpuCircuitCache,
        handle: GpuCircuitCacheHandle,
        compilation_ledger: Arc<CircuitCompilationLedger>,
    ) -> Result<Self> {
        let circuit_generation = next_exact_circuit_generation()?;
        Ok(Self {
            provider,
            cache: Mutex::new(cache),
            handle,
            circuit_generation,
            compilation_ledger,
            invalid_reason: OnceLock::new(),
            query_var_batch_cache: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(not(feature = "host-io"))]
    fn new(
        provider: Arc<CudaKernelProvider>,
        cache: GpuCircuitCache,
        handle: GpuCircuitCacheHandle,
    ) -> Self {
        Self {
            provider,
            cache: Mutex::new(cache),
            handle,
            invalid_reason: OnceLock::new(),
            query_var_batch_cache: Mutex::new(HashMap::new()),
        }
    }

    fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }

    fn handle(&self) -> &GpuCircuitCacheHandle {
        &self.handle
    }

    #[cfg(feature = "host-io")]
    fn compilation_witness(&self) -> ExactCircuitWitness {
        let ledger = self.compilation_ledger.snapshot();
        ExactCircuitWitness {
            circuit_generation: self.circuit_generation,
            compiler_invocations: ledger.compiler_invocations,
            materializations: ledger.materializations,
            disk_cache_restores: ledger.disk_cache_restores,
            gpu_cache_hits: ledger.gpu_cache_hits,
            cache_slot: self.handle.slot_index(),
        }
    }

    #[cfg(feature = "host-io")]
    fn invalidate(&self, reason: String) {
        let _ = self.invalid_reason.set(reason);
    }

    fn ensure_usable(&self) -> Result<()> {
        if let Some(reason) = self.invalid_reason.get() {
            return Err(XlogError::Execution(format!(
                "Exact GPU circuit state is permanently invalid after a failed device rollback: {reason}"
            )));
        }
        Ok(())
    }

    /// Device-resident batched query vars for `query_vars_host`, uploading once
    /// and reusing the cached slice on repeat calls with the same vars. The
    /// upload is a tracked htod; caching it keeps a warm training loop free of
    /// per-step host transfers.
    fn cached_query_var_batch(
        &self,
        query_vars_host: Vec<u32>,
    ) -> Result<Arc<TrackedCudaSlice<u32>>> {
        let mut cache = self
            .query_var_batch_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&query_vars_host) {
            return Ok(Arc::clone(cached));
        }
        let mut query_vars = self.provider.memory().alloc::<u32>(query_vars_host.len())?;
        self.provider
            .htod_sync_copy_into_tracked(&query_vars_host, &mut query_vars)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload batched query vars: {}", e))
            })?;
        let query_vars = Arc::new(query_vars);
        cache.insert(query_vars_host, Arc::clone(&query_vars));
        Ok(query_vars)
    }
}

#[cfg_attr(not(feature = "host-io"), allow(dead_code))]
struct GpuCountLiftQuery {
    atom: GroundAtom,
    target_count: u32,
    leaf_count: u32,
    leaf_probs: TrackedCudaSlice<f64>,
}

#[cfg_attr(not(feature = "host-io"), allow(dead_code))]
struct GpuCountLiftState {
    provider: Arc<CudaKernelProvider>,
    queries: Vec<GpuCountLiftQuery>,
}

impl GpuCountLiftState {
    fn new(provider: Arc<CudaKernelProvider>, queries: Vec<GpuCountLiftQuery>) -> Self {
        Self { provider, queries }
    }

    #[cfg(feature = "host-io")]
    fn evaluate(&self) -> Result<ExactResult> {
        let func = self
            .provider
            .device()
            .inner()
            .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_COUNT_LIFT_EXACT)
            .ok_or_else(|| {
                XlogError::Kernel("weights_count_lift_exact kernel not found".to_string())
            })?;
        let mut query_probs = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let scratch_len = query
                .target_count
                .checked_add(1)
                .ok_or_else(|| XlogError::Compilation("count-lift target overflow".to_string()))?;
            let mut scratch = self.provider.memory().alloc::<f64>(scratch_len as usize)?;
            let mut out = self.provider.memory().alloc::<f64>(1)?;
            unsafe {
                func.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &query.leaf_probs,
                        query.leaf_count,
                        query.target_count,
                        &mut scratch,
                        &mut out,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("weights_count_lift_exact failed: {}", e)))?;
            let mut host = vec![0.0f64; 1];
            self.provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&out, &mut host)
                .map_err(|e| XlogError::Kernel(format!("count-lift result dtoh failed: {}", e)))?;
            let mut prob = host[0];
            if (-1e-12..0.0).contains(&prob) || prob == -1e-12 {
                prob = 0.0;
            } else if prob > 1.0 && (1.0..=1.0 + 1e-12).contains(&prob) {
                prob = 1.0;
            }
            if !prob.is_finite() || !(0.0..=1.0).contains(&prob) {
                return Err(XlogError::Kernel(format!(
                    "count-lift GPU evaluator returned invalid probability {}",
                    prob
                )));
            }
            let log_prob = if prob == 0.0 {
                f64::NEG_INFINITY
            } else {
                prob.ln()
            };
            query_probs.push(QueryProbability {
                atom: query.atom.clone(),
                log_prob,
                prob,
            });
        }
        Ok(ExactResult {
            log_z_e: 0.0,
            query_probs,
        })
    }
}

/// What a CNF variable stands for, in the order of the gradient vectors.
#[derive(Debug, Clone)]
pub enum ProbVarInfo {
    /// A plain probabilistic fact: one atom, one probability. `prob` is
    /// exactly the Bernoulli weight `w_true` stored in the GPU weight table
    /// for this variable, so `p*(1-p)` is the correct Jacobian for
    /// `grad_true`/`grad_false` at this slot.
    Fact { atom: GroundAtom, prob: f64 },
    /// One Bernoulli decision of an annotated disjunction's chain.
    Choice {
        /// Declared heads of the whole disjunction with their *marginal*
        /// probabilities (context/display only — see `prob` below for the
        /// Jacobian-correct parameter of this specific chain variable).
        choices: Arc<[(GroundAtom, f64)]>,
        /// Index of this chain variable's head within `choices`.
        choice_index: usize,
        /// The *conditional* Bernoulli parameter actually assigned to this
        /// CNF variable's weight (`p_i / (1 - sum of earlier heads'
        /// probabilities)`), i.e. the same value stored in
        /// `provenance::Provenance::choice_probs` and used to build the GPU
        /// weight table for this variable. `prob*(1-prob)` — using *this*
        /// `prob`, not `choices[choice_index].1` — is the correct Jacobian
        /// for `grad_true`/`grad_false` at this slot; the two are generally
        /// different values.
        prob: f64,
    },
    /// A variable introduced by compilation that is not a source of randomness.
    Other,
}

#[cfg(feature = "host-io")]
#[derive(Debug, Clone, Copy)]
struct FactWeightChange {
    entry_index: usize,
    var: u32,
    old_prob: f64,
    new_prob: f64,
    evidence: Option<bool>,
}

#[cfg(feature = "host-io")]
fn fact_log_weights(prob: f64, evidence: Option<bool>) -> (f64, f64) {
    let mut log_true = prob.ln();
    let mut log_false = (1.0 - prob).ln();
    match evidence {
        Some(true) => log_false = f64::NEG_INFINITY,
        Some(false) => log_true = f64::NEG_INFINITY,
        None => {}
    }
    (log_true, log_false)
}

#[derive(Clone)]
pub struct ExactDdnnfProgram {
    gpu: Option<Arc<GpuExactState>>,
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    count_lift_gpu: Option<Arc<GpuCountLiftState>>,
    queries: Vec<QuerySpec>,
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    random_vars: Option<Arc<DeviceRandomVarList>>,
    max_var: u32,
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    origin: ExactProgramOrigin,
    #[allow(dead_code)] // retained: config is stored for future re-compilation paths
    gpu_config: GpuConfig,
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    last_compile_profile: Option<CircuitCompileProfile>,
    /// Sparse storage for what each CNF variable stands for: only variables that
    /// were actually assigned to a probabilistic fact or annotated-disjunction
    /// choice are present, as `(var, info)` pairs sorted by `var`. The sort is
    /// by `var` (not construction order) so entries are laid out in the same
    /// order `prob_var_map()` materializes them in, which makes the vector
    /// itself directly inspectable/debuggable as a CNF-var-indexed sequence.
    /// `var` may repeat (a leaf and a choice can be assigned the same CNF
    /// variable in principle); when it does, the *last* matching entry wins
    /// during materialization (see the ordering note on the `sort_by_key` call
    /// in `compile_provenance_with_gpu`, which documents which entry that is).
    /// CNF variables are 1-indexed. Call `prob_var_map()` to materialize the
    /// dense, `grad_true`/`grad_false`-aligned view on demand. Populated only when
    /// compiled with the "host-io" feature (empty otherwise); also empty when
    /// compiled through the GPU count-lift fast path, since that path never
    /// builds a CNF encoding (see `uses_gpu_native_count_lift()` and
    /// `prob_var_map()` below — an empty map there does not mean the program
    /// has no probabilistic facts).
    prob_var_entries: Vec<(u32, ProbVarInfo)>,
    /// Fixed evidence assignments keyed by CNF variable; mutable fact updates
    /// consult this map to preserve the compile-time evidence mask.
    #[cfg(feature = "host-io")]
    fixed_evidence_by_var: BTreeMap<u32, bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactProgramOrigin {
    Source,
    Program,
}

impl ExactDdnnfProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance_with_gpu(
            provenance,
            GpuConfig::default(),
            ExactProgramOrigin::Source,
        )
    }

    pub fn compile_source_with_gpu(source: &str, config: GpuConfig) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance_with_gpu(provenance, config, ExactProgramOrigin::Source)
    }

    /// Compile an already parsed program with the requested GPU configuration.
    ///
    /// Imports must already be resolved and merged. This method does not load
    /// unresolved `use` declarations from the filesystem.
    pub fn compile_from_program(program: &Program, config: GpuConfig) -> Result<Self> {
        let provenance = extract_from_program(program)?;
        Self::compile_provenance_with_gpu(provenance, config, ExactProgramOrigin::Program)
    }

    #[allow(dead_code)] // retained: accessor for future re-compilation paths
    pub(crate) fn gpu_config(&self) -> GpuConfig {
        self.gpu_config
    }

    #[cfg(feature = "host-io")]
    pub(crate) fn origin(&self) -> ExactProgramOrigin {
        self.origin
    }

    pub fn uses_gpu_production_backend(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get the latest circuit compilation profile (populated when XLOG_WARMUP_PROFILE=1).
    pub fn last_compile_profile(&self) -> Option<&CircuitCompileProfile> {
        self.last_compile_profile.as_ref()
    }

    /// Materializes a dense vector describing what each CNF variable stands for.
    ///
    /// The returned vector's length is the CNF encoder's variable *capacity*
    /// (`3 * number of PIR nodes` at compile time, see `compilation/gpu_cnf.rs`)
    /// — **not** the number of CNF variables actually in use, and not the
    /// number of random variables in the program. Slot `v` describes CNF
    /// variable `v` directly when `v` was assigned; slot `0` is always unused
    /// padding (CNF variables are 1-indexed), and so is any other slot with no
    /// variable assigned to it — those padding slots are indistinguishable
    /// from `ProbVarInfo::Other`. Do not treat `len()` of the result as a
    /// variable count or a random-variable count; use
    /// [`Self::random_var_indices`] for that ([`Self::num_vars`] returns this
    /// same capacity, not a count, so it is not a substitute here).
    ///
    /// What *is* guaranteed is alignment with `evaluate_gpu_with_grads`'s
    /// `grad_true`/`grad_false` vectors, which are allocated with the same
    /// capacity: `prob_var_map()[v]` and `grad_true[v]` name and value the
    /// same variable `v`.
    ///
    /// Rebuilds the dense vector on every call from the sparse
    /// `prob_var_entries` storage that actually lives for the lifetime of the
    /// program.
    ///
    /// On the GPU count-lift fast path (count aggregates without evidence or
    /// disjunctions — see [`Self::uses_gpu_native_count_lift`]), no CNF
    /// encoding is ever built, so this returns an **empty** vector even for
    /// programs that do have probabilistic facts. Callers that need to
    /// enumerate a program's probabilistic facts must check
    /// `uses_gpu_native_count_lift()` first and treat an empty map from that
    /// path as "mapping unavailable", not as "no random variables".
    pub fn prob_var_map(&self) -> Vec<ProbVarInfo> {
        let capacity = if self.max_var == 0 {
            0
        } else {
            self.max_var as usize + 1
        };
        let mut dense = vec![ProbVarInfo::Other; capacity];
        for (var, info) in &self.prob_var_entries {
            debug_assert!(
                (*var as usize) < capacity,
                "prob_var_entries contains CNF var {} but capacity is only {} \
                 (max_var {}); entries must never exceed the encoder's own \
                 variable capacity",
                var,
                capacity,
                self.max_var
            );
            if let Some(slot) = dense.get_mut(*var as usize) {
                *slot = info.clone();
            }
        }
        dense
    }

    #[cfg(feature = "host-io")]
    pub(crate) fn checked_prob_var_map(&self) -> Result<Vec<ProbVarInfo>> {
        self.ensure_usable()?;
        Ok(self.prob_var_map())
    }

    /// Atomically replace probabilities for independent probabilistic facts.
    ///
    /// The complete batch is validated before any device write. Annotated-
    /// disjunction choices and compiler-introduced variables are deliberately
    /// immutable through this surface because changing them requires additional
    /// normalization or has no source probability at all.
    #[cfg(feature = "host-io")]
    pub(crate) fn set_fact_probabilities(&mut self, updates: &BTreeMap<u32, f64>) -> Result<()> {
        self.set_fact_probabilities_with_device_failures(updates, None, None)
    }

    #[cfg(feature = "host-io")]
    fn set_fact_probabilities_with_device_failures(
        &mut self,
        updates: &BTreeMap<u32, f64>,
        fail_after_successful_writes: Option<usize>,
        fail_after_successful_rollback_writes: Option<usize>,
    ) -> Result<()> {
        if self.count_lift_gpu.is_some() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "mutable exact fact probabilities".to_string(),
                context: "GPU count-lift exact programs do not expose CNF fact variables"
                    .to_string(),
            });
        }
        let state = self.gpu_state()?;
        state.ensure_usable()?;
        let dense = self.prob_var_map();
        let mut changes = Vec::with_capacity(updates.len());
        for (&var, &new_prob) in updates {
            if var == 0 {
                return Err(XlogError::Compilation(
                    "Cannot update CNF variable 0: exact variables are 1-indexed".to_string(),
                ));
            }
            if var > self.max_var || var as usize >= dense.len() {
                return Err(XlogError::Compilation(format!(
                    "Cannot update CNF variable {var}: valid range is 1..={}",
                    self.max_var
                )));
            }
            if !new_prob.is_finite() || !(0.0..=1.0).contains(&new_prob) {
                return Err(XlogError::Compilation(format!(
                    "Probability for CNF variable {var} must be finite and within [0, 1], got {new_prob}"
                )));
            }
            match &dense[var as usize] {
                ProbVarInfo::Fact { prob, .. } => {
                    let entry_index = self
                        .prob_var_entries
                        .iter()
                        .rposition(|(entry_var, info)| {
                            *entry_var == var && matches!(info, ProbVarInfo::Fact { .. })
                        })
                        .ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "CNF variable {var} fact metadata is unavailable"
                            ))
                        })?;
                    changes.push(FactWeightChange {
                        entry_index,
                        var,
                        old_prob: *prob,
                        new_prob,
                        evidence: self.fixed_evidence_by_var.get(&var).copied(),
                    });
                }
                ProbVarInfo::Choice { .. } => {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "mutable exact fact probabilities".to_string(),
                        context: format!(
                            "CNF variable {var} is an annotated-disjunction choice; only independent probabilistic facts are mutable"
                        ),
                    });
                }
                ProbVarInfo::Other => {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "mutable exact fact probabilities".to_string(),
                        context: format!(
                            "CNF variable {var} is compiler-introduced or unmapped; only independent probabilistic facts are mutable"
                        ),
                    });
                }
            }
        }

        if changes.is_empty() {
            return Ok(());
        }

        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let var_stride = cache.var_stride()? as usize;
        let slot_start = state.handle().slot_index() as usize * var_stride;
        let provider = state.provider.clone();
        let mut successful_writes = 0usize;
        let write_result: Result<()> = (|| {
            for change in &changes {
                let index = slot_start.checked_add(change.var as usize).ok_or_else(|| {
                    XlogError::Compilation("fact weight index overflow".to_string())
                })?;
                let (new_log_true, new_log_false) =
                    fact_log_weights(change.new_prob, change.evidence);
                {
                    let (log_true, _) = cache.var_log_weights_mut();
                    let mut destination = log_true.slice_mut(index..index + 1);
                    provider.htod_sync_copy_into_tracked(&[new_log_true], &mut destination)?;
                }
                successful_writes += 1;
                if fail_after_successful_writes == Some(successful_writes) {
                    return Err(XlogError::Kernel(
                        "injected fact probability device-write failure".to_string(),
                    ));
                }
                {
                    let (_, log_false) = cache.var_log_weights_mut();
                    let mut destination = log_false.slice_mut(index..index + 1);
                    provider.htod_sync_copy_into_tracked(&[new_log_false], &mut destination)?;
                }
                successful_writes += 1;
                if fail_after_successful_writes == Some(successful_writes) {
                    return Err(XlogError::Kernel(
                        "injected fact probability device-write failure".to_string(),
                    ));
                }
            }
            Ok(())
        })();

        if let Err(write_error) = write_result {
            let mut successful_rollback_writes = 0usize;
            let rollback_result: Result<()> = (|| {
                for change in &changes {
                    let index = slot_start.checked_add(change.var as usize).ok_or_else(|| {
                        XlogError::Compilation("fact weight rollback index overflow".to_string())
                    })?;
                    let (old_log_true, old_log_false) =
                        fact_log_weights(change.old_prob, change.evidence);
                    {
                        let (log_true, _) = cache.var_log_weights_mut();
                        let mut destination = log_true.slice_mut(index..index + 1);
                        provider.htod_sync_copy_into_tracked(&[old_log_true], &mut destination)?;
                    }
                    successful_rollback_writes += 1;
                    if fail_after_successful_rollback_writes == Some(successful_rollback_writes) {
                        return Err(XlogError::Kernel(
                            "injected fact probability rollback failure".to_string(),
                        ));
                    }
                    {
                        let (_, log_false) = cache.var_log_weights_mut();
                        let mut destination = log_false.slice_mut(index..index + 1);
                        provider.htod_sync_copy_into_tracked(&[old_log_false], &mut destination)?;
                    }
                    successful_rollback_writes += 1;
                    if fail_after_successful_rollback_writes == Some(successful_rollback_writes) {
                        return Err(XlogError::Kernel(
                            "injected fact probability rollback failure".to_string(),
                        ));
                    }
                }
                Ok(())
            })();
            if let Err(rollback_error) = rollback_result {
                let combined = format!(
                    "Fact probability device update failed ({write_error}); rollback also failed ({rollback_error})"
                );
                state.invalidate(combined.clone());
                return Err(XlogError::Kernel(combined));
            }
            return Err(write_error);
        }

        for change in changes {
            match &mut self.prob_var_entries[change.entry_index].1 {
                ProbVarInfo::Fact { prob, .. } => *prob = change.new_prob,
                _ => unreachable!("validated fact metadata changed while holding exclusive state"),
            }
        }
        Ok(())
    }

    #[cfg(all(test, feature = "host-io"))]
    pub(crate) fn set_fact_probabilities_with_device_failures_for_test(
        &mut self,
        updates: &BTreeMap<u32, f64>,
        fail_after_successful_writes: Option<usize>,
        fail_after_successful_rollback_writes: Option<usize>,
    ) -> Result<()> {
        self.set_fact_probabilities_with_device_failures(
            updates,
            fail_after_successful_writes,
            fail_after_successful_rollback_writes,
        )
    }

    #[cfg(feature = "host-io")]
    pub(crate) fn ensure_usable(&self) -> Result<()> {
        if let Some(state) = &self.gpu {
            state.ensure_usable()?;
        }
        Ok(())
    }

    #[doc(hidden)]
    #[cfg(feature = "host-io")]
    pub fn uses_gpu_native_count_lift(&self) -> bool {
        self.count_lift_gpu.is_some()
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate(&self) -> Result<ExactResult> {
        if let Some(count_lift_gpu) = &self.count_lift_gpu {
            return count_lift_gpu.evaluate();
        }

        // `gpu` is `None` only when compilation found an empty PIR root set
        // (no probabilistic leaves and no derivations reach any query), so
        // every query atom is unprovable and P = 0 is the correct semantics —
        // this is NOT a missing-GPU fallback; a real circuit with an
        // unavailable GPU fails at compile time instead.
        if self.gpu.is_none() {
            let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
            for query in &self.queries {
                query_probs.push(QueryProbability {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                });
            }
            return Ok(ExactResult {
                log_z_e: 0.0,
                query_probs,
            });
        }

        let log_z_e = self.eval_log_z_gpu(None)?;
        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let (log_prob, prob) = match query.var {
                None => (f64::NEG_INFINITY, 0.0),
                Some(var) => {
                    let log_z_eq = self.eval_log_z_gpu(Some(var))?;
                    let log_prob = log_z_eq - log_z_e;
                    let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
                        0.0
                    } else {
                        log_prob.exp()
                    };
                    if prob.is_nan() {
                        return Err(XlogError::Execution(
                            "Exact inference error: NaN probability encountered".to_string(),
                        ));
                    }
                    prob = prob.clamp(0.0, 1.0);
                    (log_prob, prob)
                }
            };

            query_probs.push(QueryProbability {
                atom: query.atom.clone(),
                log_prob,
                prob,
            });
        }

        Ok(ExactResult {
            log_z_e,
            query_probs,
        })
    }

    /// Returns the CNF encoder's variable *capacity* (`max_var + 1`), i.e. the
    /// same quantity as `prob_var_map().len()` — **not** the number of CNF
    /// variables actually assigned, and not the number of random variables in
    /// the program (most CNF variables are auxiliary Tseitin variables with
    /// no probabilistic meaning). Use [`Self::random_var_indices`] to count or
    /// enumerate random variables instead.
    pub fn num_vars(&self) -> usize {
        if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        }
    }

    /// Returns the indices of random (probabilistic) variables in order.
    ///
    /// Random variables are those with non-trivial weights (not (0.0, 0.0)).
    /// These correspond to annotated disjunctions in the source program.
    /// The order matches the order variables were assigned during CNF encoding.
    #[cfg(feature = "host-io")]
    pub fn random_var_indices(&self) -> Vec<u32> {
        let Some(state) = self.gpu.as_ref() else {
            return Vec::new();
        };
        let Some(random_vars) = self.random_vars.as_ref() else {
            return Vec::new();
        };
        if random_vars.is_empty() {
            return Vec::new();
        }
        let count = random_vars.count() as usize;
        let mut host = vec![0u32; count];
        let view = random_vars.list().slice(0..count);
        if let Err(e) = state
            .provider()
            .device()
            .inner()
            .dtoh_sync_copy_into(&view, &mut host)
        {
            eprintln!("Failed to read random var list: {}", e);
            return Vec::new();
        }
        host
    }

    /// CNF variable id for the `idx`-th query formula (DIMACS, 1-based), if present.
    pub(crate) fn query_var(&self, idx: usize) -> Option<u32> {
        self.queries.get(idx).and_then(|q| q.var)
    }

    /// GPU neural fast-path: compute NLL gradients w.r.t. probability tensors (no host reads).
    ///
    /// This implements the design in `docs/design/2026-01-22-gpu-native-compilation-design.md` §5.3:
    /// - Fill AD conditional-chain log-weights from device-resident `p[label]`.
    /// - Run XGCF forward+backward on GPU.
    /// - Scatter gradients back into probability-space via the correct chain rule (uses both grad_true + grad_false).
    ///
    /// The output gradient buffers are updated in-place:
    /// - Base run: `out = dlogZ_base/dp`
    /// - Query-forced run: `out -= dlogZ_query/dp`
    ///   Result: `out = dL/dp` for `L = -log P(query | evidence)` (NLL).
    pub fn neural_backward_nll_buffers(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
    ) -> Result<()> {
        self.neural_backward_nll_buffers_inner(slots, query_idx, probs, out_grads, cfg, None, true)
    }

    /// Same as [`Self::neural_backward_nll_buffers`], but also returns the device-resident scalar NLL loss:
    /// `L = -log P(query | evidence)`.
    ///
    /// The returned slice has length 1 and is written on GPU (no device->host reads).
    pub fn neural_backward_nll_buffers_with_device_loss(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
        expected_true: bool,
    ) -> Result<TrackedCudaSlice<f64>> {
        let state = self.gpu_state()?;
        state.ensure_usable()?;
        let mut loss = state.provider.memory().alloc::<f64>(1)?;
        self.neural_backward_nll_buffers_inner(
            slots,
            query_idx,
            probs,
            out_grads,
            cfg,
            Some(&mut loss),
            expected_true,
        )?;
        Ok(loss)
    }

    /// Batched variant of [`Self::neural_backward_nll_buffers_with_device_loss`].
    ///
    /// Computes NLL gradients for `batch` queries that share one compiled circuit
    /// template and returns a device-resident vector of `batch` scalar losses.
    ///
    /// On circuits that require free-variable correction, this falls back to the
    /// existing per-query path for correctness.
    pub fn neural_backward_nll_buffers_batch_with_device_loss(
        &self,
        slots: &GpuWeightSlots,
        query_indices: &[usize],
        probs_batch: &[Vec<CudaBuffer>],
        out_grads_batch: &mut [Vec<CudaBuffer>],
        cfg: NeuralFastPathConfig,
        expected_true: bool,
    ) -> Result<TrackedCudaSlice<f64>> {
        let batch = query_indices.len();
        if batch == 0 {
            return Err(XlogError::Execution(
                "Neural fast-path batch error: empty query batch".to_string(),
            ));
        }
        if probs_batch.len() != batch || out_grads_batch.len() != batch {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path batch error: query/prob/grad batch mismatch ({}/{}/{})",
                batch,
                probs_batch.len(),
                out_grads_batch.len()
            )));
        }

        let state = self.gpu_state()?;
        let batch_u32 = u32::try_from(batch).map_err(|_| {
            XlogError::Compilation("Neural fast-path batch size exceeds u32".to_string())
        })?;
        let device = state.provider.device().inner();

        // Fallback for circuits that currently require per-query free-var correction.
        {
            let cache = state
                .cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.has_any_free_var_mask() {
                drop(cache);
                let mut losses = state.provider.memory().alloc::<f64>(batch)?;
                for q in 0..batch {
                    let loss_q = self.neural_backward_nll_buffers_with_device_loss(
                        slots,
                        query_indices[q],
                        &probs_batch[q],
                        &mut out_grads_batch[q],
                        cfg,
                        expected_true,
                    )?;
                    let mut dst = losses.slice_mut(q..(q + 1));
                    device.dtod_copy(&loss_q, &mut dst).map_err(|e| {
                        XlogError::Kernel(format!(
                            "Failed to copy fallback batch loss to output: {}",
                            e
                        ))
                    })?;
                }
                return Ok(losses);
            }
        }

        let fill = device
            .get_func(NEURAL_MODULE, neural_kernels::NEURAL_FILL_AD_CHAIN_F32)
            .ok_or_else(|| {
                XlogError::Kernel("neural_fill_ad_chain_f32 kernel not found".to_string())
            })?;
        let scatter = device
            .get_func(
                NEURAL_MODULE,
                neural_kernels::NEURAL_SCATTER_AD_CHAIN_GRADS_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("neural_scatter_ad_chain_grads_f32 kernel not found".to_string())
            })?;
        let binary_f64 = device
            .get_func(ARITH_MODULE, arith_kernels::ARITH_BINARY_F64)
            .ok_or_else(|| XlogError::Kernel("arith_binary_f64 kernel not found".to_string()))?;
        let apply_query_false_batched = device
            .get_func(
                WEIGHTS_MODULE,
                weights_kernels::WEIGHTS_APPLY_QUERY_VARS_FALSE_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "weights_apply_query_vars_false_batched kernel not found".to_string(),
                )
            })?;
        let apply_query_true_batched = device
            .get_func(
                WEIGHTS_MODULE,
                weights_kernels::WEIGHTS_APPLY_QUERY_VARS_TRUE_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "weights_apply_query_vars_true_batched kernel not found".to_string(),
                )
            })?;

        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let var_stride = cache.var_stride()?;
        let var_stride_usize = var_stride as usize;
        let node_stride = cache.node_stride();
        let node_stride_usize = node_stride as usize;

        let mut var_log_true_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut var_log_false_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        cache.copy_slot_weights_to_batch(
            state.handle(),
            &mut var_log_true_batch,
            &mut var_log_false_batch,
            batch_u32,
        )?;

        let mut values_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * node_stride_usize)?;
        let mut adj_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * node_stride_usize)?;
        let mut grad_true_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut grad_false_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut base_roots = state.provider.memory().alloc::<f64>(batch)?;
        let mut query_roots = state.provider.memory().alloc::<f64>(batch)?;
        let mut losses = state.provider.memory().alloc::<f64>(batch)?;
        let mut force_saved = state.provider.memory().alloc::<f64>(batch)?;

        let mut query_vars_host: Vec<u32> = Vec::with_capacity(batch);

        // Fill per-query var weight rows from device-resident probability tensors.
        for q in 0..batch {
            if probs_batch[q].len() != out_grads_batch[q].len() {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: probs len {} != out_grads len {} for query {}",
                    probs_batch[q].len(),
                    out_grads_batch[q].len(),
                    q
                )));
            }
            if probs_batch[q].len() != slots.num_groups_usize() {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: expected {} groups, got {} for query {}",
                    slots.num_groups_usize(),
                    probs_batch[q].len(),
                    q
                )));
            }

            let query_var = self.query_var(query_indices[q]).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Neural fast-path batch error: query {} has no CNF var",
                    query_indices[q]
                ))
            })?;
            if query_var == 0 || query_var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: query var {} out of bounds (max_var={})",
                    query_var, self.max_var
                )));
            }
            query_vars_host.push(query_var);

            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                if prob_buf.arity() != 1 {
                    return Err(XlogError::Compilation(
                        "Neural fast-path expects 1-column prob buffers".to_string(),
                    ));
                }
                let ty = prob_buf.schema().column_type(0).ok_or_else(|| {
                    XlogError::Compilation("Missing prob buffer schema".to_string())
                })?;
                if ty != ScalarType::F32 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path expects prob dtype F32, got {:?}",
                        ty
                    )));
                }

                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = neural_slot_count_u32(slot_vars.len())?;
                if prob_buf.num_rows() != labels as u64 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path prob rows {} != labels {}",
                        prob_buf.num_rows(),
                        labels
                    )));
                }
                if out_grads_batch[q][g].num_rows() != labels as u64 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path grad rows {} != labels {}",
                        out_grads_batch[q][g].num_rows(),
                        labels
                    )));
                }

                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let mut q_true = var_log_true_batch.slice_mut(row_start..row_end);
                let mut q_false = var_log_false_batch.slice_mut(row_start..row_end);

                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    fill.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &mut q_true,
                            &mut q_false,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("neural_fill_ad_chain_f32 failed: {}", e))
                })?;
            }
        }

        // Base pass (all queries): grads = dlogZ_base/dp, roots = logZ_base.
        cache.eval_grads_inplace_fused_batched(
            state.handle(),
            &var_log_true_batch,
            &var_log_false_batch,
            &mut values_batch,
            &mut adj_batch,
            &mut grad_true_batch,
            &mut grad_false_batch,
            batch_u32,
        )?;
        cache.copy_root_batched_from_values(
            state.handle(),
            &values_batch,
            &mut base_roots,
            batch_u32,
        )?;

        // Scatter base gradients into output buffers.
        for q in 0..batch {
            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;
            let q_grad_true = grad_true_batch.slice(row_start..row_end);
            let q_grad_false = grad_false_batch.slice(row_start..row_end);

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = neural_slot_count_u32(slot_vars.len())?;
                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let out_col = out_grads_batch[q][g]
                    .columns_mut()
                    .get_mut(0)
                    .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

                let shared_bytes: u32 = 3u64
                    .checked_mul(labels as u64)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                    })?;

                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    scatter.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &q_grad_true,
                            &q_grad_false,
                            0u8,
                            out_col,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("neural_scatter (base) failed: {}", e)))?;
            }
        }

        // Reuse the device-resident query-var batch (uploaded once and cached),
        // so a warm training loop performs no per-step tracked host transfer here.
        let query_vars = state.cached_query_var_batch(query_vars_host)?;
        let force_grid = checked_launch_grid_u32("gpu exact batched query force", batch_u32, 256)?;
        if force_grid != 0 {
            if expected_true {
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    apply_query_false_batched.clone().launch(
                        LaunchConfig {
                            grid_dim: (force_grid, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            query_vars.as_ref(),
                            batch_u32,
                            self.max_var,
                            var_stride,
                            &mut var_log_false_batch,
                            &mut force_saved,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "weights_apply_query_vars_false_batched failed: {}",
                        e
                    ))
                })?;
            } else {
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    apply_query_true_batched.clone().launch(
                        LaunchConfig {
                            grid_dim: (force_grid, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            query_vars.as_ref(),
                            batch_u32,
                            self.max_var,
                            var_stride,
                            &mut var_log_true_batch,
                            &mut force_saved,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "weights_apply_query_vars_true_batched failed: {}",
                        e
                    ))
                })?;
            }
        }

        // Query-forced pass (all queries): grads = dlogZ_query/dp, roots = logZ_query.
        cache.eval_grads_inplace_fused_batched(
            state.handle(),
            &var_log_true_batch,
            &var_log_false_batch,
            &mut values_batch,
            &mut adj_batch,
            &mut grad_true_batch,
            &mut grad_false_batch,
            batch_u32,
        )?;
        cache.copy_root_batched_from_values(
            state.handle(),
            &values_batch,
            &mut query_roots,
            batch_u32,
        )?;

        let loss_grid = checked_launch_grid_u32("gpu exact batched query loss", batch_u32, 256)?;
        if loss_grid != 0 {
            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
            unsafe {
                binary_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (loss_grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&base_roots, &query_roots, batch_u32, 1u8, &mut losses),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("Failed to compute batched NLL loss: {}", e)))?;
        }

        // Scatter query gradients with subtract mode.
        for q in 0..batch {
            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;
            let q_grad_true = grad_true_batch.slice(row_start..row_end);
            let q_grad_false = grad_false_batch.slice(row_start..row_end);

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = neural_slot_count_u32(slot_vars.len())?;
                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let out_col = out_grads_batch[q][g]
                    .columns_mut()
                    .get_mut(0)
                    .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

                let shared_bytes: u32 = 3u64
                    .checked_mul(labels as u64)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                    })?;

                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    scatter.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &q_grad_true,
                            &q_grad_false,
                            1u8,
                            out_col,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("neural_scatter (query) failed: {}", e)))?;
            }
        }

        Ok(losses)
    }

    #[allow(clippy::too_many_arguments)]
    fn neural_backward_nll_buffers_inner(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
        out_loss: Option<&mut TrackedCudaSlice<f64>>,
        expected_true: bool,
    ) -> Result<()> {
        if self.gpu.is_none() {
            return Err(XlogError::Execution(
                "Neural fast-path error: program has no compiled circuit".to_string(),
            ));
        }

        let query_var = self.query_var(query_idx).ok_or_else(|| {
            XlogError::Execution(format!(
                "Neural fast-path error: query {} has no CNF var",
                query_idx
            ))
        })?;

        if probs.len() != out_grads.len() {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: probs len {} != out_grads len {}",
                probs.len(),
                out_grads.len()
            )));
        }
        if probs.len() != slots.num_groups_usize() {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: expected {} groups, got {}",
                slots.num_groups_usize(),
                probs.len()
            )));
        }

        let state = self.gpu_state()?;
        let device = state.provider.device().inner();

        let fill = device
            .get_func(NEURAL_MODULE, neural_kernels::NEURAL_FILL_AD_CHAIN_F32)
            .ok_or_else(|| {
                XlogError::Kernel("neural_fill_ad_chain_f32 kernel not found".to_string())
            })?;
        let scatter = device
            .get_func(
                NEURAL_MODULE,
                neural_kernels::NEURAL_SCATTER_AD_CHAIN_GRADS_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("neural_scatter_ad_chain_grads_f32 kernel not found".to_string())
            })?;
        let binary_f64 = device
            .get_func(ARITH_MODULE, arith_kernels::ARITH_BINARY_F64)
            .ok_or_else(|| XlogError::Kernel("arith_binary_f64 kernel not found".to_string()))?;

        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let root_idx = state.handle().root() as usize;

        // If the caller requested the scalar loss, keep the base logZ on device so we can compute
        // loss = logZ_base - logZ_query without any host reads.
        let mut base_log_z: Option<TrackedCudaSlice<f64>> = if out_loss.is_some() {
            Some(state.provider.memory().alloc::<f64>(1)?)
        } else {
            None
        };

        // 1) Update AD chain weights from device-resident p[label].
        for (g, prob_buf) in probs.iter().enumerate() {
            if prob_buf.arity() != 1 {
                return Err(XlogError::Compilation(
                    "Neural fast-path expects 1-column prob buffers".to_string(),
                ));
            }
            let ty = prob_buf
                .schema()
                .column_type(0)
                .ok_or_else(|| XlogError::Compilation("Missing prob buffer schema".to_string()))?;
            if ty != ScalarType::F32 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path expects prob dtype F32, got {:?}",
                    ty
                )));
            }

            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = neural_slot_count_u32(slot_vars.len())?;

            if prob_buf.num_rows() != labels as u64 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path prob rows {} != labels {}",
                    prob_buf.num_rows(),
                    labels
                )));
            }

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;

            let (var_log_true, var_log_false) = cache.var_log_weights_mut();

            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
            unsafe {
                fill.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        var_log_true,
                        var_log_false,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_fill_ad_chain_f32 failed: {}", e)))?;
        }

        // 2) Base run: out = dlogZ_base/dp
        cache.eval_grads_inplace_fused(state.handle())?;
        if let Some(base) = base_log_z.as_mut() {
            let root_view = cache.values().slice(root_idx..(root_idx + 1));
            device.dtod_copy(&root_view, base).map_err(|e| {
                XlogError::Kernel(format!("Failed to copy base logZ on GPU: {}", e))
            })?;
        }
        for (g, prob_buf) in probs.iter().enumerate() {
            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = neural_slot_count_u32(slot_vars.len())?;

            let out_buf = out_grads.get_mut(g).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing output grad buffer".to_string())
            })?;
            if out_buf.arity() != 1 {
                return Err(XlogError::Compilation(
                    "Neural fast-path expects 1-column grad buffers".to_string(),
                ));
            }
            let out_ty = out_buf
                .schema()
                .column_type(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad buffer schema".to_string()))?;
            if out_ty != ScalarType::F32 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path expects grad dtype F32, got {:?}",
                    out_ty
                )));
            }
            if out_buf.num_rows() != labels as u64 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path grad rows {} != labels {}",
                    out_buf.num_rows(),
                    labels
                )));
            }

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;
            let out_col = out_buf
                .columns_mut()
                .get_mut(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

            let shared_bytes: u32 = 3u64
                .checked_mul(labels as u64)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                })?;

            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
            unsafe {
                scatter.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        cache.grad_true(),
                        cache.grad_false(),
                        0u8,
                        out_col,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_scatter (base) failed: {}", e)))?;
        }

        // 3) Query run: out -= dlogZ_query/dp
        if query_var == 0 || query_var > self.max_var {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: query var {} out of bounds (max_var={})",
                query_var, self.max_var
            )));
        }

        let mut restore = state.provider.memory().alloc::<f64>(1)?;
        if expected_true {
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, query_var, &mut restore)?;
            }
        } else {
            {
                let (var_log_true, _) = cache.var_log_weights_mut();
                force_query_var_true(state.provider(), var_log_true, query_var, &mut restore)?;
            }
        }

        cache.eval_grads_inplace_fused(state.handle())?;
        if let Some(out) = out_loss {
            let base = base_log_z
                .as_ref()
                .expect("base_log_z allocated when out_loss requested");
            let root_view = cache.values().slice(root_idx..(root_idx + 1));
            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
            unsafe {
                binary_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (base, &root_view, 1u32, 1u8, out),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("Failed to compute NLL loss on GPU: {}", e)))?;
        }
        for (g, prob_buf) in probs.iter().enumerate() {
            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = neural_slot_count_u32(slot_vars.len())?;

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;
            let out_col = out_grads[g]
                .columns_mut()
                .get_mut(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

            let shared_bytes: u32 = 3u64
                .checked_mul(labels as u64)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                })?;

            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
            unsafe {
                scatter.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        cache.grad_true(),
                        cache.grad_false(),
                        1u8,
                        out_col,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_scatter (query) failed: {}", e)))?;
        }
        if expected_true {
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                restore_query_var_false(state.provider(), var_log_false, query_var, &restore)?;
            }
        } else {
            {
                let (var_log_true, _) = cache.var_log_weights_mut();
                restore_query_var_true(state.provider(), var_log_true, query_var, &restore)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads(&self) -> Result<ExactResultWithGrads> {
        if self.gpu.is_none() {
            if self.count_lift_gpu.is_some() {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "GPU exact gradient evaluation".to_string(),
                    context: "GPU count-lift exact backend does not expose gradient evaluation; \
                              gradient production paths require a compiled GPU-native Decision-DNNF exact backend"
                        .to_string(),
                });
            }
            return Ok(ExactResultWithGrads {
                log_z_e: 0.0,
                query_grads: Vec::new(),
            });
        }
        self.ensure_usable()?;

        let weights_len = if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        };

        let (log_z_e, grad_true_e, grad_false_e) = self.eval_log_z_and_grads_gpu_cached(None)?;

        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_grads: Vec<QueryGradients> = Vec::with_capacity(self.queries.len());

        for query in &self.queries {
            let Some(var) = query.var else {
                query_grads.push(QueryGradients {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                    grad_true: vec![0.0; weights_len],
                    grad_false: vec![0.0; weights_len],
                });
                continue;
            };

            let idx = var as usize;
            if idx >= weights_len {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (len={})",
                    var, weights_len
                )));
            }

            let (log_z_eq, grad_true_eq, grad_false_eq) =
                self.eval_log_z_and_grads_gpu_cached(Some(var))?;

            let log_prob = log_z_eq - log_z_e;
            let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
                0.0
            } else {
                log_prob.exp()
            };
            if prob.is_nan() {
                return Err(XlogError::Execution(
                    "Exact inference error: NaN probability encountered".to_string(),
                ));
            }
            prob = prob.clamp(0.0, 1.0);

            if grad_true_eq.len() != grad_true_e.len() || grad_false_eq.len() != grad_false_e.len()
            {
                return Err(XlogError::Execution(
                    "Exact inference error: gradient length mismatch".to_string(),
                ));
            }

            let mut grad_true: Vec<f64> = grad_true_eq;
            let mut grad_false: Vec<f64> = grad_false_eq;
            for i in 0..grad_true.len() {
                grad_true[i] -= grad_true_e[i];
                grad_false[i] -= grad_false_e[i];
            }

            query_grads.push(QueryGradients {
                atom: query.atom.clone(),
                log_prob,
                prob,
                grad_true,
                grad_false,
            });
        }

        Ok(ExactResultWithGrads {
            log_z_e,
            query_grads,
        })
    }

    fn compile_provenance_with_gpu(
        provenance: Provenance,
        config: GpuConfig,
        origin: ExactProgramOrigin,
    ) -> Result<Self> {
        if config.memory_bytes == 0 {
            return Err(XlogError::Kernel(
                "GPU memory budget must be non-zero".to_string(),
            ));
        }

        let provenance = if config.decision_order_hint {
            crate::decision_order::apply_decision_order_hint(provenance)
        } else {
            provenance
        };

        let mut roots_set: HashSet<crate::pir::PirNodeId> = HashSet::new();

        let mut evidence_formulas: Vec<(crate::pir::PirNodeId, bool, GroundAtom)> = Vec::new();
        for (atom, value) in validated_evidence_entries(&provenance)? {
            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            match formula {
                Some(id) => {
                    roots_set.insert(id);
                    evidence_formulas.push((id, value, atom.clone()));
                }
                None => {
                    if value {
                        return Err(XlogError::Execution(format!(
                            "Exact inference error: evidence atom is never derivable: {}",
                            display_atom(atom)
                        )));
                    }
                }
            }
        }

        let mut queries: Vec<QuerySpec> = Vec::new();
        #[cfg(feature = "host-io")]
        let mut query_nodes: Vec<(usize, crate::pir::PirNodeId)> = Vec::new();
        for atom in &provenance.queries {
            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            if let Some(id) = formula {
                roots_set.insert(id);
                #[cfg(feature = "host-io")]
                {
                    query_nodes.push((queries.len(), id));
                }
            }
            queries.push(QuerySpec {
                atom: atom.clone(),
                var: None,
            });
        }

        // Ensure ALL probabilistic variable nodes (Decision, Lit, NegLit) are reachable
        // so they get CNF variables. This is required for the template/neural fast-path
        // where GpuWeightSlots expects one CNF variable per ChoiceVarId/LeafId.
        for (idx, node) in provenance.pir.nodes().iter().enumerate() {
            match node {
                crate::pir::PirNode::Decision { .. }
                | crate::pir::PirNode::Lit { .. }
                | crate::pir::PirNode::NegLit { .. } => {
                    roots_set.insert(crate::pir::PirNodeId::from_u32(idx as u32));
                }
                _ => {}
            }
        }

        let mut roots: Vec<crate::pir::PirNodeId> = roots_set.into_iter().collect();
        roots.sort();

        if roots.is_empty() {
            return Ok(Self {
                gpu: None,
                count_lift_gpu: None,
                queries,
                random_vars: None,
                max_var: 0,
                origin,
                gpu_config: config,
                last_compile_profile: None,
                prob_var_entries: Vec::new(),
                #[cfg(feature = "host-io")]
                fixed_evidence_by_var: BTreeMap::new(),
            });
        }

        let count_lift_gpu = try_build_count_lift_gpu_state(&provenance, &queries, config)?;
        if let Some(count_lift_gpu) = count_lift_gpu {
            // No CNF encoding is built on this path (count aggregates are
            // evaluated by a dedicated GPU kernel instead), so there is no
            // leaf_var/choice_var table to derive a variable map from. Leave
            // `prob_var_entries` empty and `max_var` at 0 — callers must use
            // `uses_gpu_native_count_lift()` to tell this apart from "no random
            // variables in the program" (see the doc on `prob_var_map()`).
            return Ok(Self {
                gpu: None,
                count_lift_gpu: Some(count_lift_gpu),
                queries,
                random_vars: None,
                max_var: 0,
                origin,
                gpu_config: config,
                last_compile_profile: None,
                prob_var_entries: Vec::new(),
                #[cfg(feature = "host-io")]
                fixed_evidence_by_var: BTreeMap::new(),
            });
        }

        let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(config.memory_bytes),
        ));
        let provider = Arc::new(CudaKernelProvider::new(device, memory)?);

        let canonical_cnf_hash = crate::cnf::canonical_pir_hash(&provenance.pir, &roots)?;
        let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider)?;
        let gpu_roots = GpuPirRoots::from_host(&roots, &provider)?;
        let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider)?;
        if encoding.vars.max_var != encoding.cnf.var_cap {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: CNF var_cap {} != vars.max_var {}",
                encoding.cnf.var_cap, encoding.vars.max_var
            )));
        }

        // Which probabilistic fact (or choice) each CNF variable stands for, stored
        // sparsely as `(var, info)` pairs (see the doc on `prob_var_entries`).
        // `leaf_var`/`choice_var` are GPU-resident dense tables keyed by
        // LeafId/ChoiceVarId; a value of 0 means the leaf/choice was not reachable
        // from the compiled roots.
        #[cfg(feature = "host-io")]
        let prob_var_entries = {
            let mut leaf_var_host = vec![0u32; encoding.vars.leaf_var.len()];
            provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&encoding.vars.leaf_var, &mut leaf_var_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read leaf_var table: {}", e)))?;
            let mut choice_var_host = vec![0u32; encoding.vars.choice_var.len()];
            provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&encoding.vars.choice_var, &mut choice_var_host)
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to read choice_var table: {}", e))
                })?;

            let mut entries: Vec<(u32, ProbVarInfo)> = Vec::new();
            for (leaf_idx, &var) in leaf_var_host.iter().enumerate() {
                if var == 0 {
                    continue;
                }
                let leaf = crate::pir::LeafId::new(leaf_idx as u32);
                if let (Some(atom), Some(prob)) = (
                    provenance.leaf_atoms.get(&leaf),
                    provenance.leaf_probs.get(&leaf),
                ) {
                    entries.push((
                        var,
                        ProbVarInfo::Fact {
                            atom: atom.clone(),
                            prob: *prob,
                        },
                    ));
                }
            }
            for (choice_idx, &var) in choice_var_host.iter().enumerate() {
                if var == 0 {
                    continue;
                }
                let choice = crate::pir::ChoiceVarId::new(choice_idx as u32);
                // The map must carry the *conditional* Bernoulli parameter that was
                // actually assigned to this CNF variable's weight (`choice_probs`),
                // not the disjunction's declared marginal probabilities
                // (`ChoiceSource::choices`), which only serve as display context.
                // See ProbVarInfo::Choice::prob's doc for why the two differ.
                match (
                    provenance.choice_sources.get(&choice),
                    provenance.choice_probs.get(&choice),
                ) {
                    (Some(source), Some(&(cond_true, _cond_false))) => {
                        entries.push((
                            var,
                            ProbVarInfo::Choice {
                                choices: source.choices.clone(),
                                choice_index: source.choice_index,
                                prob: cond_true,
                            },
                        ));
                    }
                    _ => {
                        // `choice_sources` and `choice_probs` are populated in
                        // lock-step in `provenance.rs` (see the choice-handling
                        // arm around lines 708-717) and remapped in lock-step in
                        // `decision_order.rs` (lines 115-123); a `choice_var_host`
                        // entry pointing at a `ChoiceVarId` missing from either map
                        // means that invariant broke upstream. Silently falling
                        // back to `ProbVarInfo::Other` would make a real
                        // compilation bug look like "this CNF variable is just an
                        // auxiliary Tseitin variable" to every caller of
                        // `prob_var_map()`, so fail loudly instead.
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: choice_sources/choice_probs are out of \
                             sync for {:?} (CNF var {var}); expected both maps to contain \
                             this ChoiceVarId",
                            choice
                        )));
                    }
                }
            }
            // `entries` is built leaf-first, then choice-second (the two loops
            // above), so on a `var` collision the leaf entry comes first and the
            // choice entry comes second. `sort_by_key` is STABLE, so it preserves
            // that relative order; `prob_var_map()`'s materialization loop then
            // overwrites earlier entries with later ones for the same slot
            // (last-write-wins), so choice wins over leaf on collision. Do not
            // change this to `sort_unstable_by_key`: it makes no such ordering
            // guarantee and would silently flip which entry wins.
            entries.sort_by_key(|(var, _)| *var);
            entries
        };
        #[cfg(not(feature = "host-io"))]
        let prob_var_entries: Vec<(u32, ProbVarInfo)> = Vec::new();

        let (leaf_probs_host, choice_true_host, choice_false_host) =
            build_weight_sources(&provenance)?;

        let leaf_probs = upload_f64(&provider, &leaf_probs_host)?;
        let choice_true = upload_f64(&provider, &choice_true_host)?;
        let choice_false = upload_f64(&provider, &choice_false_host)?;

        let evidence_by_var = if evidence_formulas.is_empty() {
            let mut evidence = provider
                .memory()
                .alloc::<u8>((encoding.vars.max_var as usize) + 1)?;
            provider
                .device()
                .inner()
                .memset_zeros(&mut evidence)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero evidence buffer: {}", e)))?;
            evidence
        } else {
            let mut nodes: Vec<u32> = Vec::with_capacity(evidence_formulas.len());
            let mut vals: Vec<u8> = Vec::with_capacity(evidence_formulas.len());
            for (node, value, _atom) in &evidence_formulas {
                nodes.push(node.as_u32());
                vals.push(if *value { 1u8 } else { 2u8 });
            }
            let evidence_nodes = upload_u32(&provider, &nodes)?;
            let evidence_vals = upload_u8(&provider, &vals)?;
            build_evidence_by_var_gpu(
                &encoding.vars.node_var,
                &evidence_nodes,
                &evidence_vals,
                encoding.vars.max_var,
                &provider,
            )?
        };

        #[cfg(feature = "host-io")]
        let fixed_evidence_by_var = {
            if evidence_formulas.is_empty() {
                BTreeMap::new()
            } else {
                let evidence_nodes = evidence_formulas
                    .iter()
                    .map(|(node, _, _)| node.as_u32())
                    .collect::<Vec<_>>();
                let evidence_vars = map_nodes_to_vars_gpu(
                    &encoding.vars.node_var,
                    &upload_u32(&provider, &evidence_nodes)?,
                    encoding.vars.max_var,
                    &provider,
                )?;
                let mut vars_host = vec![0u32; evidence_vars.len()];
                provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(&evidence_vars, &mut vars_host)
                    .map_err(|error| {
                        XlogError::Kernel(format!(
                            "Failed to read exact evidence CNF variables: {error}"
                        ))
                    })?;
                let mut assignments = BTreeMap::new();
                for (var, (_, value, _)) in vars_host.into_iter().zip(&evidence_formulas) {
                    if let Some(previous) = assignments.insert(var, *value) {
                        if previous != *value {
                            return Err(XlogError::Compilation(format!(
                                "Conflicting exact evidence assignments for CNF variable {var}"
                            )));
                        }
                    }
                }
                assignments
            }
        };

        let weights = build_weights_gpu(
            &encoding.vars,
            &leaf_probs,
            &choice_true,
            &choice_false,
            &evidence_by_var,
            &provider,
        )?;
        let random_var_count = leaf_probs_host
            .len()
            .checked_add(choice_true_host.len())
            .ok_or_else(|| XlogError::Compilation("random var count overflow".to_string()))?;
        let random_var_count = u32::try_from(random_var_count)
            .map_err(|_| XlogError::Compilation("random var count exceeds u32".to_string()))?;
        let num_leaf_probs = u32::try_from(leaf_probs_host.len())
            .map_err(|_| XlogError::Compilation("leaf_probs count exceeds u32".to_string()))?;
        let num_choice_probs = u32::try_from(choice_true_host.len())
            .map_err(|_| XlogError::Compilation("choice_probs count exceeds u32".to_string()))?;
        let (random_var_list, actual_random_var_count) = collect_random_vars_device(
            &provider,
            &encoding.vars,
            num_leaf_probs,
            num_choice_probs,
            random_var_count,
        )?;
        let random_vars =
            DeviceRandomVarList::from_device(random_var_list, actual_random_var_count)?;

        let compile_config = default_compile_config(&encoding.cnf, config.memory_bytes)?;
        let cache_config = default_cache_config(&encoding.cnf, &compile_config)?;

        let mut cache = GpuCircuitCache::new(&provider, cache_config)?;
        let compilation_ledger = Arc::new(CircuitCompilationLedger::new());
        let (handle, compile_profile) = compile_gpu_d4_and_verify_cached_with_ledger(
            &encoding.cnf,
            &encoding.decision_var_limit,
            &provider,
            &compile_config,
            &mut cache,
            &random_vars,
            CircuitCompilationContext {
                canonical_cnf_hash: Some(canonical_cnf_hash),
                ledger: compilation_ledger.as_ref(),
            },
        )?;
        cache.store_weights(&handle, &weights.log_true, &weights.log_false)?;

        #[cfg(feature = "host-io")]
        if !query_nodes.is_empty() {
            let mut node_ids: Vec<u32> = Vec::with_capacity(query_nodes.len());
            for (_idx, node) in &query_nodes {
                node_ids.push(node.as_u32());
            }
            let node_ids_device = upload_u32(&provider, &node_ids)?;
            let vars_device = map_nodes_to_vars_gpu(
                &encoding.vars.node_var,
                &node_ids_device,
                encoding.vars.max_var,
                &provider,
            )?;

            let mut vars_host = vec![0u32; vars_device.len()];
            provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&vars_device, &mut vars_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read query vars: {}", e)))?;

            for (i, (query_idx, _)) in query_nodes.iter().enumerate() {
                let var = vars_host[i];
                queries[*query_idx].var = Some(var);
            }
        }

        #[cfg(feature = "host-io")]
        let state = GpuExactState::new(provider, cache, handle, compilation_ledger)?;
        #[cfg(not(feature = "host-io"))]
        let state = GpuExactState::new(provider, cache, handle);

        Ok(Self {
            gpu: Some(Arc::new(state)),
            count_lift_gpu: None,
            queries,
            random_vars: Some(Arc::new(random_vars)),
            max_var: encoding.vars.max_var,
            origin,
            gpu_config: config,
            last_compile_profile: compile_profile,
            prob_var_entries,
            #[cfg(feature = "host-io")]
            fixed_evidence_by_var,
        })
    }

    #[cfg(feature = "host-io")]
    fn eval_log_z_gpu(&self, query_true: Option<u32>) -> Result<f64> {
        let state = self.gpu_state()?;
        state.ensure_usable()?;
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(var) = query_true {
            if var == 0 || var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (max_var={})",
                    var, self.max_var
                )));
            }
        }

        let mut restore = None;
        if let Some(var) = query_true {
            let mut buf = state.provider.memory().alloc::<f64>(1)?;
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, var, &mut buf)?;
            }
            restore = Some((var, buf));
        }

        let mut out_log_z = state.provider.memory().alloc::<f64>(1)?;
        let eval_result = cache.eval_log_wmc_device_inplace(state.handle(), &mut out_log_z);

        if let Some((var, buf)) = restore {
            let (_, var_log_false) = cache.var_log_weights_mut();
            let restore_result =
                restore_query_var_false(state.provider(), var_log_false, var, &buf);
            if let Err(err) = eval_result {
                restore_result?;
                return Err(err);
            }
            restore_result?;
        } else {
            eval_result?;
        }

        let mut host = [0.0f64];
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&out_log_z, &mut host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read logZ: {}", e)))?;
        validate_circuit_value(host[0])
    }

    fn gpu_state(&self) -> Result<Arc<GpuExactState>> {
        self.gpu.clone().ok_or_else(|| {
            XlogError::Execution(
                "Exact inference GPU error: program has no compiled circuit".to_string(),
            )
        })
    }

    #[cfg(feature = "host-io")]
    pub(crate) fn circuit_witness(&self) -> Result<ExactCircuitWitness> {
        let state = self.gpu_state()?;
        state.ensure_usable()?;
        Ok(state.compilation_witness())
    }

    #[cfg(feature = "host-io")]
    fn eval_log_z_and_grads_gpu_cached(
        &self,
        query_true: Option<u32>,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let state = self.gpu_state()?;
        state.ensure_usable()?;
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(var) = query_true {
            if var == 0 || var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (max_var={})",
                    var, self.max_var
                )));
            }
        }

        let mut restore = None;
        if let Some(var) = query_true {
            let mut buf = state.provider.memory().alloc::<f64>(1)?;
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, var, &mut buf)?;
            }
            restore = Some((var, buf));
        }

        let eval_result = cache.eval_grads_inplace(state.handle());

        if let Some((var, buf)) = restore {
            let (_, var_log_false) = cache.var_log_weights_mut();
            let restore_result =
                restore_query_var_false(state.provider(), var_log_false, var, &buf);
            if let Err(err) = eval_result {
                restore_result?;
                return Err(err);
            }
            restore_result?;
        } else {
            eval_result?;
        }

        let weights_len = if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        };

        let device = state.provider.device().inner();
        let mut host_grad_true: Vec<f64> = vec![0.0; weights_len];
        let mut host_grad_false: Vec<f64> = vec![0.0; weights_len];

        let root_idx = state.handle().root() as usize;
        let root_view = cache.values().slice(root_idx..(root_idx + 1));
        let mut log_z = [0.0_f64];
        device
            .dtoh_sync_copy_into(&root_view, &mut log_z)
            .map_err(|e| XlogError::Kernel(format!("Failed to read logZ: {}", e)))?;
        let log_z = validate_circuit_value(log_z[0])?;

        // Gradient buffers are multi-slot: [slot0_var0..slot0_varN, slot1_var0..].
        // Slice into the correct slot to download only this circuit's gradients.
        let var_stride = cache.var_stride()? as usize;
        let slot = state.handle().slot_index() as usize;
        let grad_start = slot * var_stride;
        let grad_end = grad_start + weights_len;
        let grad_true_slot = cache.grad_true().slice(grad_start..grad_end);
        let grad_false_slot = cache.grad_false().slice(grad_start..grad_end);
        device
            .dtoh_sync_copy_into(&grad_true_slot, &mut host_grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
        device
            .dtoh_sync_copy_into(&grad_false_slot, &mut host_grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;
        validate_circuit_gradient_values(&host_grad_true, &host_grad_false)?;

        Ok((log_z, host_grad_true, host_grad_false))
    }
}

fn try_build_count_lift_gpu_state(
    provenance: &Provenance,
    queries: &[QuerySpec],
    config: GpuConfig,
) -> Result<Option<Arc<GpuCountLiftState>>> {
    if queries.is_empty() || !provenance.evidence.is_empty() || !provenance.choice_probs.is_empty()
    {
        return Ok(None);
    }

    let fired_count_predicates: HashSet<&str> = provenance
        .aggregate_lifting
        .iter()
        .filter(|entry| {
            entry.status == AggregateLiftStatus::Fired
                && entry.operator.as_str() == "count"
                && entry.deterministic_rows == 0
        })
        .map(|entry| entry.predicate.as_str())
        .collect();
    if fired_count_predicates.is_empty() {
        return Ok(None);
    }
    if queries
        .iter()
        .any(|query| !fired_count_predicates.contains(query.atom.predicate.as_str()))
    {
        return Ok(None);
    }

    let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(config.memory_bytes),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory)?);
    let mut gpu_queries = Vec::with_capacity(queries.len());
    for query in queries {
        let target_count = match count_lift_query_target(query)? {
            Some(target) => target,
            None => return Ok(None),
        };
        let root = match provenance.query_formula(&query.atom.predicate, &query.atom.args) {
            Some(root) => root,
            None => return Ok(None),
        };
        let mut leaves = HashSet::new();
        collect_count_lift_leaves(provenance, root, &mut leaves)?;
        if leaves.is_empty() || leaves.len() > 64 {
            return Ok(None);
        }
        if target_count > leaves.len() as u32 {
            return Ok(None);
        }
        let mut leaves: Vec<_> = leaves.into_iter().collect();
        leaves.sort_by_key(|leaf| leaf.as_u32());
        let mut leaf_probs_host = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let p = *provenance.leaf_probs.get(&leaf).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Count-lift GPU evaluator missing probability for leaf {}",
                    leaf.as_u32()
                ))
            })?;
            leaf_probs_host.push(p);
        }
        let leaf_count = u32::try_from(leaf_probs_host.len())
            .map_err(|_| XlogError::Compilation("count-lift leaf count exceeds u32".to_string()))?;
        let leaf_probs = upload_f64(&provider, &leaf_probs_host)?;
        gpu_queries.push(GpuCountLiftQuery {
            atom: query.atom.clone(),
            target_count,
            leaf_count,
            leaf_probs,
        });
    }
    Ok(Some(Arc::new(GpuCountLiftState::new(
        provider,
        gpu_queries,
    ))))
}

fn count_lift_query_target(query: &QuerySpec) -> Result<Option<u32>> {
    match query.atom.args.last() {
        Some(Value::I64(value)) if *value >= 0 => u32::try_from(*value)
            .map(Some)
            .map_err(|_| XlogError::Compilation("count-lift target exceeds u32".to_string())),
        _ => Ok(None),
    }
}

fn collect_count_lift_leaves(
    provenance: &Provenance,
    node: crate::pir::PirNodeId,
    leaves: &mut HashSet<crate::pir::LeafId>,
) -> Result<()> {
    let pir_node = provenance.pir.node(node).ok_or_else(|| {
        XlogError::Compilation(format!(
            "Count-lift GPU evaluator saw invalid PIR node {}",
            node.as_u32()
        ))
    })?;
    match pir_node {
        crate::pir::PirNode::Const(_) => Ok(()),
        crate::pir::PirNode::Lit { leaf } | crate::pir::PirNode::NegLit { leaf } => {
            leaves.insert(*leaf);
            Ok(())
        }
        crate::pir::PirNode::And { children } | crate::pir::PirNode::Or { children } => {
            for child in children {
                collect_count_lift_leaves(provenance, *child, leaves)?;
            }
            Ok(())
        }
        crate::pir::PirNode::Decision { .. } => Err(XlogError::Compilation(
            "Count-lift GPU evaluator does not support annotated-disjunction choices".to_string(),
        )),
    }
}

fn force_query_var_false(
    provider: &Arc<CudaKernelProvider>,
    log_false: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FORCE_VAR_FALSE)
        .ok_or_else(|| XlogError::Kernel("weights_force_var_false kernel not found".to_string()))?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_false, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_force_var_false failed: {}", e)))?;
    Ok(())
}

fn restore_query_var_false(
    provider: &Arc<CudaKernelProvider>,
    log_false: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_RESTORE_VAR_FALSE)
        .ok_or_else(|| {
            XlogError::Kernel("weights_restore_var_false kernel not found".to_string())
        })?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_false, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_restore_var_false failed: {}", e)))?;
    Ok(())
}

fn force_query_var_true(
    provider: &Arc<CudaKernelProvider>,
    log_true: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FORCE_VAR_TRUE)
        .ok_or_else(|| XlogError::Kernel("weights_force_var_true kernel not found".to_string()))?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_true, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_force_var_true failed: {}", e)))?;
    Ok(())
}

fn restore_query_var_true(
    provider: &Arc<CudaKernelProvider>,
    log_true: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_RESTORE_VAR_TRUE)
        .ok_or_else(|| {
            XlogError::Kernel("weights_restore_var_true kernel not found".to_string())
        })?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_true, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_restore_var_true failed: {}", e)))?;
    Ok(())
}

pub(crate) fn default_compile_config(
    cnf: &xlog_solve::GpuCnf,
    memory_bytes: u64,
) -> Result<GpuCompileConfig> {
    // Must match the default GPU-native Decision-DNNF compiler configuration expected
    // by the Python training paths.
    // Sizing is conservative and strictly bounded by `GpuCompileConfig::{smooth_node_cap,smooth_edge_cap}`.
    let frontier_depth: u16 = 6;

    let var_cap = cnf.var_cap.max(1);
    let trail_bytes_per_item = (var_cap as u64)
        .checked_add(1)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| XlogError::Compilation("trail size overflow".to_string()))?;
    let denom = trail_bytes_per_item
        .checked_mul(8)
        .ok_or_else(|| XlogError::Compilation("trail memory denominator overflow".to_string()))?;
    if memory_bytes
        < denom.checked_mul(8).ok_or_else(|| {
            XlogError::Compilation("minimum frontier memory requirement overflow".to_string())
        })?
    {
        return Err(XlogError::Compilation(format!(
            "memory budget {} cannot hold the minimum GPU-native Decision-DNNF frontier allocation",
            memory_bytes
        )));
    }
    let max_items_by_trail = memory_bytes / denom;
    let max_frontier_items = max_items_by_trail.min(4096).min(u64::from(u32::MAX)) as u32;

    // The GPU-native Decision-DNNF compiler emits one leaf circuit per frontier item;
    // caps must scale with the maximum frontier size (up to 2^frontier_depth,
    // bounded by max_frontier_items).
    let frontier_cap_factor = (1u64
        .checked_shl(frontier_depth as u32)
        .unwrap_or(u64::from(u32::MAX)))
    .min(u64::from(max_frontier_items)) as u32;

    let per_item_nodes = cnf
        .var_cap
        .checked_mul(5)
        .ok_or_else(|| XlogError::Compilation("smooth_node_cap overflow".to_string()))?
        .max(1024);
    let smooth_node_cap = per_item_nodes
        .checked_mul(frontier_cap_factor)
        .ok_or_else(|| XlogError::Compilation("smooth_node_cap overflow".to_string()))?;

    // Edge capacity scales with node capacity; AND/OR fanout grows edges but stays within a small
    // multiple of nodes for the compiler's frontier emission patterns.
    let mut smooth_edge_cap = smooth_node_cap
        .checked_mul(2)
        .ok_or_else(|| XlogError::Compilation("smooth_edge_cap overflow".to_string()))?;
    if smooth_edge_cap < max_frontier_items {
        smooth_edge_cap = max_frontier_items;
    }

    // The verifier's UNSAT certificate (resolution trace) can be large even when the source CNF
    // is small, because equivalence checking builds CNF(C) with many Tseitin variables/clauses.
    // Allocate a larger share of the budget to the GPU CDCL arenas to avoid deterministic
    // overflow errors in production verifier paths.
    let mut cdcl_learned_bytes = memory_bytes / 8;
    if cdcl_learned_bytes < 4 * 1024 * 1024 {
        cdcl_learned_bytes = 4 * 1024 * 1024;
    }

    let config = GpuCompileConfig {
        frontier_depth,
        max_frontier_items,
        max_depth: 128,
        smooth_node_cap,
        smooth_edge_cap,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };
    Ok(config)
}

pub(crate) fn default_cache_config(
    cnf: &xlog_solve::GpuCnf,
    compile: &GpuCompileConfig,
) -> Result<GpuCircuitCacheConfig> {
    if compile.smooth_node_cap == 0 || compile.smooth_edge_cap == 0 {
        return Err(XlogError::Compilation(
            "GPU cache config requires non-zero smoothing caps".to_string(),
        ));
    }
    Ok(GpuCircuitCacheConfig {
        num_slots: 4, // Hold 4 circuit templates; power-of-2 hash table.
        table_size: 8,
        node_cap: compile.smooth_node_cap,
        edge_cap: compile.smooth_edge_cap,
        level_cap: compile.smooth_node_cap,
        var_cap: cnf.var_cap,
    })
}

pub(crate) fn build_weight_sources(
    provenance: &Provenance,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let max_leaf = provenance.leaf_probs.keys().map(|leaf| leaf.as_u32()).max();
    let leaf_len = max_leaf.map(|v| v as usize + 1).unwrap_or(0);
    let mut leaf_probs = vec![0.0f64; leaf_len];
    let mut leaf_seen = vec![false; leaf_len];
    for (leaf, p) in &provenance.leaf_probs {
        let idx = leaf.as_u32() as usize;
        if idx >= leaf_len {
            return Err(XlogError::Compilation(
                "leaf probability index out of bounds".to_string(),
            ));
        }
        leaf_probs[idx] = *p;
        leaf_seen[idx] = true;
    }
    if let Some((idx, _)) = leaf_seen.iter().enumerate().find(|(_, seen)| !**seen) {
        return Err(XlogError::Compilation(format!(
            "missing probability for leaf {}",
            idx
        )));
    }

    let max_choice = provenance
        .choice_probs
        .keys()
        .map(|choice| choice.as_u32())
        .max();
    let choice_len = max_choice.map(|v| v as usize + 1).unwrap_or(0);
    let mut choice_true = vec![0.0f64; choice_len];
    let mut choice_false = vec![0.0f64; choice_len];
    let mut choice_seen = vec![false; choice_len];
    for (choice, (pt, pf)) in &provenance.choice_probs {
        let idx = choice.as_u32() as usize;
        if idx >= choice_len {
            return Err(XlogError::Compilation(
                "choice probability index out of bounds".to_string(),
            ));
        }
        choice_true[idx] = *pt;
        choice_false[idx] = *pf;
        choice_seen[idx] = true;
    }
    if let Some((idx, _)) = choice_seen.iter().enumerate().find(|(_, seen)| !**seen) {
        return Err(XlogError::Compilation(format!(
            "missing probability for choice {}",
            idx
        )));
    }

    Ok((leaf_probs, choice_true, choice_false))
}

pub(crate) fn upload_u32(
    provider: &Arc<CudaKernelProvider>,
    host: &[u32],
) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u32>(host.len())?;
    provider
        .htod_sync_copy_into_tracked(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload u32 buffer: {}", e)))?;
    Ok(buf)
}

pub(crate) fn upload_u8(
    provider: &Arc<CudaKernelProvider>,
    host: &[u8],
) -> Result<TrackedCudaSlice<u8>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u8>(host.len())?;
    provider
        .htod_sync_copy_into_tracked(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload u8 buffer: {}", e)))?;
    Ok(buf)
}

pub(crate) fn upload_f64(
    provider: &Arc<CudaKernelProvider>,
    host: &[f64],
) -> Result<TrackedCudaSlice<f64>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<f64>(host.len())?;
    provider
        .htod_sync_copy_into_tracked(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload f64 buffer: {}", e)))?;
    Ok(buf)
}

fn capture_compact_count_device(
    provider: &Arc<CudaKernelProvider>,
    prefix_sum: &TrackedCudaSlice<u32>,
    mask: &TrackedCudaSlice<u8>,
    n: u32,
) -> Result<TrackedCudaSlice<u32>> {
    let mut out = provider.memory().alloc::<u32>(1)?;
    let device = provider.device().inner();
    let capture_fn = device
        .get_func(FILTER_MODULE, filter_kernels::CAPTURE_COMPACT_COUNT)
        .ok_or_else(|| XlogError::Kernel("capture_compact_count kernel not found".to_string()))?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        capture_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (prefix_sum, mask, n, &mut out),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("capture_compact_count failed: {}", e)))?;
    Ok(out)
}

pub(crate) fn collect_random_vars_device(
    provider: &Arc<CudaKernelProvider>,
    vars: &GpuCnfVarTables,
    num_leaf_probs: u32,
    num_choice_probs: u32,
    _expected_count: u32,
) -> Result<(TrackedCudaSlice<u32>, u32)> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let mask_len = vars
        .max_var
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("random var mask_len overflow".to_string()))?;
    let mask_len_usize = usize::try_from(mask_len)
        .map_err(|_| XlogError::Compilation("random var mask_len exceeds usize".to_string()))?;

    let mut mask = memory.alloc::<u8>(mask_len_usize)?;
    device
        .memset_zeros(&mut mask)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero random var mask: {}", e)))?;

    let mut iota = memory.alloc::<u32>(mask_len_usize)?;
    let fill_iota = device
        .get_func(FILTER_MODULE, filter_kernels::FILL_U32_IOTA)
        .ok_or_else(|| XlogError::Kernel("fill_u32_iota kernel not found".to_string()))?;
    let block_size = 256u32;
    let grid = checked_launch_grid_u32("fill random-var iota", mask_len, block_size)?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        fill_iota.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut iota, mask_len, 0u32),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("fill_u32_iota failed: {}", e)))?;

    // Only iterate over the probabilistic entries — leaf_var and choice_var are allocated
    // to num_nodes but only the first num_leaf_probs / num_choice_probs entries correspond
    // to variables with actual probabilities. Non-probabilistic PIR leaf nodes also get
    // CNF variables but must NOT be marked as random.
    let leaf_len = num_leaf_probs;
    let choice_len = num_choice_probs;

    let mark_kernel = device
        .get_func(FILTER_MODULE, filter_kernels::MARK_RANDOM_VARS)
        .ok_or_else(|| XlogError::Kernel("mark_random_vars kernel not found".to_string()))?;
    let mark_n = leaf_len.max(choice_len);
    if mark_n > 0 {
        let grid = checked_launch_grid_u32("mark random vars", mark_n, block_size)?;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            mark_kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &vars.leaf_var,
                    &vars.choice_var,
                    leaf_len,
                    choice_len,
                    &mut mask,
                    mask_len,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mark_random_vars failed: {}", e)))?;
    }

    let prefix_sum = provider.scan_u8_mask_device(&mask, mask_len)?;
    let count_device = capture_compact_count_device(provider, &prefix_sum, &mask, mask_len)?;

    // Read the actual random var count from device (the GPU scan result is authoritative).
    // The host-side expected_count can be wrong when some ChoiceVarIds are unreachable
    // from query/evidence roots and don't get assigned CNF variables.
    let actual_count = {
        let mut buf = vec![0u32; 1];
        device
            .dtoh_sync_copy_into(&count_device, &mut buf)
            .map_err(|e| XlogError::Kernel(format!("dtoh count_device failed: {}", e)))?;
        buf[0]
    };

    if actual_count == 0 {
        // No random variables in the circuit — return empty list.
        let out = provider.memory().alloc::<u32>(0)?;
        return Ok((out, 0));
    }

    let mut out = memory.alloc::<u32>(mask_len_usize)?;
    let compact_fn = device
        .get_func(FILTER_MODULE, filter_kernels::COMPACT_U32_BY_MASK)
        .ok_or_else(|| XlogError::Kernel("compact_u32_by_mask kernel not found".to_string()))?;
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        compact_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&iota, &mask, &prefix_sum, mask_len, &mut out),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("compact_u32_by_mask failed: {}", e)))?;

    Ok((out, actual_count))
}

fn display_atom(atom: &GroundAtom) -> String {
    if atom.args.is_empty() {
        format!("{}()", atom.predicate)
    } else {
        format!("{}({} args)", atom.predicate, atom.args.len())
    }
}

pub(crate) fn validated_evidence_entries(
    provenance: &Provenance,
) -> Result<Vec<(&GroundAtom, bool)>> {
    let mut evidence_atoms: HashMap<GroundAtom, bool> = HashMap::new();
    let mut entries = Vec::with_capacity(provenance.evidence.len());
    for (atom, value) in &provenance.evidence {
        let canonical_atom = provenance.canonical_atom(atom)?;
        if let Some(previous) = evidence_atoms.insert(canonical_atom, *value) {
            if previous != *value {
                return Err(XlogError::Execution(format!(
                    "Exact inference error: conflicting evidence for {}",
                    display_atom(atom)
                )));
            }
            continue;
        }
        entries.push((atom, *value));
    }
    Ok(entries)
}

#[cfg(all(test, feature = "host-io"))]
mod tests {
    use super::*;
    use xlog_cuda::CudaDevice;

    #[test]
    fn fact_weight_updates_match_compile_kernel_and_preserve_fixed_evidence() {
        let tiny = 1e-17;
        let (log_true, log_false) = fact_log_weights(tiny, None);
        assert_eq!(log_true.to_bits(), tiny.ln().to_bits());
        assert_eq!(log_false.to_bits(), (1.0 - tiny).ln().to_bits());

        let (fixed_true, impossible_false) = fact_log_weights(0.25, Some(true));
        assert_eq!(fixed_true.to_bits(), 0.25f64.ln().to_bits());
        assert_eq!(impossible_false, f64::NEG_INFINITY);

        let (impossible_true, fixed_false) = fact_log_weights(0.25, Some(false));
        assert_eq!(impossible_true, f64::NEG_INFINITY);
        assert_eq!(fixed_false.to_bits(), (1.0f64 - 0.25).ln().to_bits());
    }

    #[test]
    fn exact_evidence_entries_deduplicate_schema_equivalent_values() {
        let provenance = extract_from_source(
            "0.5::gate(\"alpha\").\n\
             evidence(gate(\"alpha\"), true).\n\
             evidence(gate(alpha), true).\n\
             query(gate(\"alpha\")).\n",
        )
        .expect("extract equivalent symbol evidence");

        let evidence = validated_evidence_entries(&provenance).expect("validate evidence");
        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].1);
    }

    #[test]
    fn test_exact_negation_probability() {
        let _gpu_guard = crate::test_gpu_lock::lock();
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }
        // 0.3::rain(). dry() :- not rain().
        // P(dry) = P(not rain) = 1 - 0.3 = 0.7
        let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let dry_prob = result.query_probs[0].prob;
        assert!(
            (dry_prob - 0.7).abs() < 1e-6,
            "P(dry) should be 0.7, got {}",
            dry_prob
        );
    }

    #[test]
    fn test_exact_multi_layer_negation() {
        let _gpu_guard = crate::test_gpu_lock::lock();
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }
        // 0.4::c(). b() :- not c(). a() :- not b().
        // P(b) = P(not c) = 0.6
        // P(a) = P(not b) = 0.4
        let source = r#"
0.4::c().
b() :- not c().
a() :- not b().
query(a()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let a_prob = result.query_probs[0].prob;
        assert!(
            (a_prob - 0.4).abs() < 1e-6,
            "P(a) should be 0.4, got {}",
            a_prob
        );
    }

    #[test]
    fn test_eval_log_z_changes_for_sprinkler_given_wet() {
        let _gpu_guard = crate::test_gpu_lock::lock();
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }

        let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let log_z_e = program.eval_log_z_gpu(None).unwrap();
        let sprinkler_var = program.query_var(1).unwrap();
        let log_z_eq = program.eval_log_z_gpu(Some(sprinkler_var)).unwrap();

        let state = program.gpu_state().unwrap();
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (_, var_log_false) = cache.var_log_weights_mut();

        let mut before = [0.0f64];
        let view = var_log_false.slice(sprinkler_var as usize..(sprinkler_var as usize + 1));
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&view, &mut before)
            .unwrap();

        let mut restore = state.provider.memory().alloc::<f64>(1).unwrap();
        force_query_var_false(state.provider(), var_log_false, sprinkler_var, &mut restore)
            .unwrap();

        let mut after = [0.0f64];
        let view_after = var_log_false.slice(sprinkler_var as usize..(sprinkler_var as usize + 1));
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&view_after, &mut after)
            .unwrap();

        restore_query_var_false(state.provider(), var_log_false, sprinkler_var, &restore).unwrap();

        assert!(
            before[0].is_finite(),
            "expected finite log_false before forcing"
        );
        assert!(
            after[0].is_infinite() && after[0].is_sign_negative(),
            "expected -inf log_false after forcing, got {}",
            after[0]
        );
        assert!(
            log_z_eq < log_z_e,
            "conditioning on sprinkler should reduce logZ (log_z_e={}, log_z_eq={})",
            log_z_e,
            log_z_eq
        );
    }

    #[test]
    fn cached_gradient_readback_rejects_non_finite_device_output() {
        let _gpu_guard = crate::test_gpu_lock::lock();
        match CudaDevice::new(0) {
            Ok(_) => {}
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA runtime initialization failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {error}");
                return;
            }
        }

        let program = ExactDdnnfProgram::compile_source(
            "0.5::rain().\n\
             query(rain()).\n",
        )
        .unwrap();
        let rain_var = program.query_var(0).unwrap() as usize;
        let state = program.gpu_state().unwrap();

        {
            let mut cache = state
                .cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let var_stride = cache.var_stride().unwrap() as usize;
            let slot_start = state.handle().slot_index() as usize * var_stride;
            let (var_log_true, _) = cache.var_log_weights_mut();
            let mut rain_true =
                var_log_true.slice_mut((slot_start + rain_var)..(slot_start + rain_var + 1));
            state
                .provider
                .htod_sync_copy_into_tracked(&[f64::INFINITY], &mut rain_true)
                .unwrap();
        }

        let error = program
            .eval_log_z_and_grads_gpu_cached(None)
            .expect_err("non-finite downloaded gradients must be rejected");
        assert!(
            matches!(error, XlogError::Compilation(ref message) if message.contains("gradient") && message.contains("non-finite")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn fact_probability_update_rolls_back_metadata_weights_and_fixed_evidence() {
        let _gpu_guard = crate::test_gpu_lock::lock();
        match CudaDevice::new(0) {
            Ok(_) => {}
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA runtime initialization failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {error}");
                return;
            }
        }

        let mut program = ExactDdnnfProgram::compile_source(
            "0.5::rain().\n\
             evidence(rain(), true).\n\
             query(rain()).\n",
        )
        .unwrap();
        let rain_var = program
            .prob_var_map()
            .iter()
            .enumerate()
            .find_map(|(var, info)| {
                matches!(info, ProbVarInfo::Fact { atom, .. } if atom.predicate == "rain")
                    .then_some(var as u32)
            })
            .unwrap();
        let before = program.evaluate().unwrap();
        assert_eq!(before.query_probs.len(), 1);
        assert!((before.query_probs[0].prob - 1.0).abs() < 1e-12);

        let error = program
            .set_fact_probabilities_with_device_failures(
                &BTreeMap::from([(rain_var, 0.9)]),
                Some(1),
                None,
            )
            .expect_err("failure after the first device write must be reported");
        assert!(error.to_string().contains("injected"));

        assert!(matches!(
            &program.prob_var_map()[rain_var as usize],
            ProbVarInfo::Fact { prob, .. } if (*prob - 0.5).abs() < 1e-12
        ));
        let after = program.evaluate().unwrap();
        assert!((after.log_z_e - before.log_z_e).abs() < 1e-12);
        assert_eq!(after.query_probs.len(), 1);
        assert!((after.query_probs[0].prob - 1.0).abs() < 1e-12);
        assert!((after.query_probs[0].prob - before.query_probs[0].prob).abs() < 1e-12);
        assert!((after.query_probs[0].log_prob - before.query_probs[0].log_prob).abs() < 1e-12);
    }
}
