//! Query executor for RIR nodes
//!
//! The executor interprets RIR (Relational IR) nodes using the CUDA kernel provider
//! to execute GPU-accelerated relational operations.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
use xlog_core::ScalarType;
use xlog_core::{RelId, Result, RuntimeConfig, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
#[cfg(test)]
use xlog_ir::{CompareOp, ConstValue, Stratum};
use xlog_ir::{ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::ilp_registry::{IlpRegistry, IlpTaggedResult};
use crate::profiler::{ExecutionStats, Profiler};
use crate::RelationStore;

mod delta;
mod epistemic_workspace;
mod expression;
mod join_cache;
mod node_dispatch;
mod recursive;
mod rewrite;
mod wcoj_cost_model;
mod wcoj_dispatch;
#[cfg(feature = "wcoj-phase-timing")]
pub mod wcoj_phase_timing;
pub use epistemic_workspace::{
    EpistemicGpuBatchExecutionResult, EpistemicGpuBatchExecutionTrace,
    EpistemicGpuCandidateGenerationTrace, EpistemicGpuCandidateValidationTrace,
    EpistemicGpuConstraintValidationTrace, EpistemicGpuConstraintWorldViewValidationTrace,
    EpistemicGpuExecutionResult, EpistemicGpuFinalResultMaterializationTrace,
    EpistemicGpuFinalResultTransferTrace, EpistemicGpuFinalTupleMaterializationTrace,
    EpistemicGpuKernelTimingTrace, EpistemicGpuMaterializationTrace,
    EpistemicGpuModelMembershipSource, EpistemicGpuModelMembershipTrace,
    EpistemicGpuPreparedExecution, EpistemicGpuPropagationTrace, EpistemicGpuProviderIdentity,
    EpistemicGpuRejectionReason, EpistemicGpuRuntimeCounters, EpistemicGpuRuntimePreflight,
    EpistemicGpuRuntimeTrace, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuTransferBudgetTrace, EpistemicGpuWorkspace, EpistemicGpuWorkspaceCapacities,
    EpistemicGpuWorkspaceLayout, EpistemicGpuWorkspaceResetTrace,
    EpistemicGpuWorldViewValidationTrace,
};
use join_cache::JoinIndexCache;
pub use join_cache::JoinIndexCacheStats;

/// Incremental update for a base relation.
pub struct RelationDelta {
    /// Tuples to insert (if any).
    pub insert: Option<CudaBuffer>,
    /// Tuples to delete (if any).
    pub delete: Option<CudaBuffer>,
}

impl RelationDelta {
    /// Create a new incremental update.
    pub fn new(insert: Option<CudaBuffer>, delete: Option<CudaBuffer>) -> Self {
        Self { insert, delete }
    }
}

/// Runtime summary for a delta recomputation pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaRecomputeStats {
    /// Number of base relations changed by the delta map.
    pub changed_relations: usize,
    /// True when at least one relation supplied delete rows.
    pub has_deletes: bool,
    /// Number of SCCs whose dependency closure was affected.
    pub affected_sccs: usize,
    /// Number of affected SCCs that were cleared and fully recomputed.
    pub recomputed_sccs: usize,
    /// Number of affected SCCs updated without clearing prior output.
    pub incremental_sccs: usize,
}

/// Runtime common subexpression elimination telemetry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommonSubexpressionStats {
    /// Number of safe subplan cache hits.
    pub hits: u64,
    /// Number of safe subplans evaluated and inserted into the cache.
    pub misses: u64,
    /// Number of subplans rejected because they cross an unsafe boundary.
    pub unsafe_rejections: u64,
    /// Rejection reason labels observed during the current executor lifetime.
    pub rejection_reasons: Vec<String>,
}

/// Runtime join observation used by adaptive re-optimization decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveJoinObservation {
    /// Left relation ID observed at the join boundary.
    pub left_rel: RelId,
    /// Right relation ID observed at the join boundary.
    pub right_rel: RelId,
    /// Estimated output rows before the join executed.
    pub estimated_output_rows: u64,
    /// Actual output rows observed after the join executed.
    pub actual_output_rows: u64,
    /// Absolute row-count delta between estimate and observation.
    pub cardinality_delta_abs: u64,
    /// Estimated selectivity before execution.
    pub estimated_selectivity: f64,
    /// Actual selectivity observed after execution.
    pub actual_selectivity: f64,
    /// Absolute selectivity delta.
    pub selectivity_delta_abs: f64,
    /// Runtime heat for the left relation.
    pub left_heat: f32,
    /// Runtime heat for the right relation.
    pub right_heat: f32,
    /// Absolute heat delta between the join inputs.
    pub heat_delta_abs: f32,
    /// Multiplicative mis-plan ratio, always at least 1.0.
    pub misplan_ratio: f64,
}

/// Deterministic adaptive re-optimization decision action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveReoptimizationAction {
    /// Adaptive re-optimization is explicitly disabled.
    Disabled,
    /// Telemetry did not cross the deterministic adaptation threshold.
    Skipped,
    /// Telemetry crossed the threshold and the candidate should be attempted.
    AttemptCandidate,
    /// Candidate output matched the baseline and was adopted.
    Adopted,
    /// Candidate failed or diverged and the baseline snapshot was restored.
    RolledBack,
}

/// Typed adaptive re-optimization decision.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveReoptimizationDecision {
    /// Decision action.
    pub action: AdaptiveReoptimizationAction,
    /// Stable reason label.
    pub reason: String,
    /// Maximum observed mis-plan ratio used by the decision.
    pub max_misplan_ratio: f64,
    /// Minimum threshold required to attempt a candidate.
    pub min_misplan_ratio: f64,
}

/// Typed adaptive re-optimization diagnostic kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveReoptimizationDiagnosticKind {
    /// Candidate execution returned an error.
    CandidateExecutionFailed,
    /// Candidate completed but did not produce baseline-equivalent outputs.
    CandidateOutputMismatch,
    /// Baseline relation snapshot was restored after an adverse candidate.
    RollbackRestoredBaseline,
}

/// Typed adaptive re-optimization diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdaptiveReoptimizationDiagnostic {
    /// Diagnostic kind.
    pub kind: AdaptiveReoptimizationDiagnosticKind,
    /// Stable reason label or error text.
    pub message: String,
}

/// Runtime adaptive re-optimization telemetry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdaptiveReoptimizationStats {
    /// Number of adaptive entry-point invocations.
    pub invocations: u64,
    /// Number of invocations skipped because the feature was disabled.
    pub disabled: u64,
    /// Number of enabled invocations that did not attempt a candidate.
    pub skipped: u64,
    /// Number of candidate plans adopted.
    pub adopted: u64,
    /// Number of candidate plans rolled back.
    pub rolled_back: u64,
    /// Last deterministic decision.
    pub last_decision: Option<AdaptiveReoptimizationDecision>,
    /// Baseline join observations from the most recent execution.
    pub last_observations: Vec<AdaptiveJoinObservation>,
    /// Typed diagnostics emitted by the adaptive entry point.
    pub diagnostics: Vec<AdaptiveReoptimizationDiagnostic>,
    /// Tracked data-plane DTOH calls added during the most recent adaptive path.
    pub data_plane_dtoh_calls: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CommonSubexpressionKey {
    Scan {
        rel: RelId,
        generation: u64,
    },
    Filter {
        input: Box<CommonSubexpressionKey>,
        predicate: String,
    },
    Project {
        input: Box<CommonSubexpressionKey>,
        columns: Vec<String>,
    },
    Join {
        left: Box<CommonSubexpressionKey>,
        right: Box<CommonSubexpressionKey>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
    },
    Union {
        inputs: Vec<CommonSubexpressionKey>,
    },
    Distinct {
        input: Box<CommonSubexpressionKey>,
        key_cols: Vec<usize>,
    },
}

/// Query executor that interprets RIR nodes using GPU kernels
///
/// The executor processes execution plans by iterating through strata and
/// executing RIR node trees. It maintains a relation store for intermediate
/// and final results.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use xlog_runtime::Executor;
/// use xlog_cuda::CudaKernelProvider;
///
/// let provider = Arc::new(CudaKernelProvider::new(device, memory)?);
/// let mut executor = Executor::new(provider);
///
/// // Execute a plan
/// let result = executor.execute_plan(&plan)?;
/// ```
pub struct Executor {
    /// CUDA kernel provider for GPU operations
    provider: Arc<CudaKernelProvider>,
    /// Storage for named relations
    store: RelationStore,
    /// Mapping from RelId to relation name
    rel_names: HashMap<RelId, String>,
    /// Mapping from relation name to RelId
    name_to_rel: HashMap<String, RelId>,
    /// Runtime statistics for adaptive optimization
    stats: StatsManager,
    /// Cached build-side join indexes (adaptive indexing)
    join_index_cache: JoinIndexCache,
    /// Per-execution CSE cache for safe deterministic subplans.
    common_subexpression_cache: HashMap<CommonSubexpressionKey, CudaBuffer>,
    /// Runtime CSE telemetry for evidence and diagnostics.
    common_subexpression_stats: CommonSubexpressionStats,
    /// Runtime adaptive re-optimization telemetry for adoption/rollback evidence.
    adaptive_reoptimization_stats: AdaptiveReoptimizationStats,
    /// Per-plan join observations captured before `StatsManager` is updated.
    adaptive_join_observations: Vec<AdaptiveJoinObservation>,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Performance profiler for --stats output
    profiler: Profiler,
    /// ILP tensor mask registry
    ilp_registry: IlpRegistry,
    /// Last ILP tagged result metadata
    ilp_last_result: Option<IlpTaggedResult>,
    /// Number of times the env-gated WCOJ triangle dispatch
    /// (`XLOG_USE_WCOJ_TRIANGLE_U32` / `RuntimeConfig::wcoj_triangle_dispatch`)
    /// produced a result and the executor installed it. Tests use this
    /// counter to assert that the WCOJ path actually fired vs. silently
    /// falling back to the binary-join chain with the same answer.
    wcoj_triangle_dispatch_count: u64,
    /// v0.6.5 slice 2 — count of times `try_dispatch_wcoj_4cycle`
    /// produced a result and the executor installed it. Tracks
    /// 4-cycle dispatches separately from triangle.
    pub(super) wcoj_4cycle_dispatch_count: u64,
    /// Goal-039 G_W63_CHAIN — count of times the chain
    /// dispatcher produced a result and the executor installed it.
    pub(super) w63_chain_dispatch_count: u64,
    /// W3.2 — count of times `try_dispatch_wcoj_clique5` produced
    /// a result and the executor installed it. Public accessor:
    /// `Executor::wcoj_clique5_dispatch_count(&self) -> u64`.
    pub(super) wcoj_clique5_dispatch_count: u64,
    /// W3.2 — count of times `try_dispatch_wcoj_clique6` produced
    /// a result and the executor installed it.
    pub(super) wcoj_clique6_dispatch_count: u64,
    /// W6.4 — count of times `try_dispatch_wcoj_clique7` produced
    /// a result and the executor installed it.
    pub(super) wcoj_clique7_dispatch_count: u64,
    /// W6.4 — count of times `try_dispatch_wcoj_clique8` produced
    /// a result and the executor installed it.
    pub(super) wcoj_clique8_dispatch_count: u64,
    /// Authorization 5 G_HIST_KC — number of recursive Merge-phase
    /// K-clique histogram refresh boundaries observed.
    pub(super) kclique_histogram_refresh_count: u64,
    /// Authorization 5 G_HIST_KC — cumulative nanoseconds spent in
    /// recursive Merge-phase K-clique histogram refresh accounting.
    pub(super) kclique_histogram_refresh_nanos: u128,
    /// W4.2 — count of times `execute_join` routed an inner-join
    /// to the nested-loop provider entry point
    /// (`CudaKernelProvider::nested_loop_join_v2_inner_u32_1key`)
    /// because the eligibility predicate + Cartesian-product
    /// threshold both held. Tests use this counter to assert that
    /// the W4.2 path actually fired vs. silently falling back to
    /// hash. Public accessor:
    /// `Executor::nested_loop_dispatch_count(&self) -> u64`.
    pub(super) nested_loop_dispatch_count: u64,
    /// Counts WCOJ pipeline errors (layout or kernel failures) that were
    /// converted into binary-join declines instead of propagating. Healthy
    /// dispatch keeps this at 0; nonzero values expose regressed kernels
    /// that would otherwise hide behind the silent-fallback contract.
    /// `XLOG_WCOJ_STRICT=1` propagates the error instead. Public accessor:
    /// `Executor::wcoj_error_decline_count(&self) -> u64`.
    pub(super) wcoj_error_decline_count: u64,
    /// D1 — count of fused group-by-root count dispatches that produced the
    /// installed result. Public accessor:
    /// `Executor::wcoj_groupby_fusion_dispatch_count(&self) -> u64`.
    pub(super) wcoj_groupby_fusion_dispatch_count: u64,
    /// D2 — count of generalized Free Join dispatches that produced the
    /// installed result. Public accessor:
    /// `Executor::free_join_dispatch_count(&self) -> u64`.
    pub(super) free_join_dispatch_count: u64,
    /// D3 — count of factorized recursive-delta dispatches that produced
    /// an installed novel set. Public accessor:
    /// `Executor::factorized_delta_dispatch_count(&self) -> u64`.
    pub(super) factorized_delta_dispatch_count: u64,
    /// Cached non-default stream for the WCOJ triangle dispatch hook.
    /// Acquired lazily on first dispatch and reused thereafter — mirrors
    /// [`xlog_cuda::CudaKernelProvider::recorded_op_stream`] for the
    /// same reason: the device-runtime
    /// [`xlog_cuda::device_runtime::StreamPool`] is grow-only with a
    /// hard cap (default 16). Acquiring per-invocation would silently
    /// drain the pool on long-lived runtimes (benchmarks, soak tests,
    /// any program with >16 matching WCOJ-eligible rules) and route
    /// subsequent dispatches through the binary-join fallback,
    /// invalidating the dispatch counter and the gate-on path.
    ///
    /// **Shared across WCOJ shapes** (v0.6.5 slice 2): triangle and
    /// 4-cycle dispatch both acquire and reuse this single stream.
    /// Renamed from `wcoj_triangle_stream` when 4-cycle dispatch
    /// landed.
    wcoj_dispatch_stream: OnceLock<xlog_cuda::device_runtime::StreamId>,
    /// Diagnostic-only: per-dispatch WCOJ triangle phase
    /// timings, populated by `try_dispatch_wcoj_triangle` when
    /// the `wcoj-phase-timing` Cargo feature is on. Read by the
    /// `wcoj_phase_report` binary in xlog-integration. Field is
    /// absent under feature-off so production builds have zero
    /// overhead.
    #[cfg(feature = "wcoj-phase-timing")]
    pub(super) last_wcoj_phase_timing:
        std::sync::Mutex<Option<wcoj_phase_timing::WcojDispatchPhaseTiming>>,
    /// W2.3: per-iteration recursive-SCC stats trace, populated
    /// by `execute_recursive_scc` after each Phase 2 (delta) and
    /// Phase 4 (full) cardinality update site. Field + types +
    /// accessor + populating call sites are gated on the
    /// `recursive-stats-trace` Cargo feature (default OFF) so
    /// production builds carry zero trace overhead — no field,
    /// no populating call site, no symbol. The W2.3 acceptance
    /// test target declares this feature in its
    /// `required-features`, so it is only built when the
    /// feature is enabled.
    #[cfg(feature = "recursive-stats-trace")]
    pub(super) last_recursive_stats_trace: RecursiveStatsTrace,
}

/// W2.3 step 7 acceptance gate — recursive-SCC stats trace.
///
/// Captures one entry per `(iteration, predicate)` boundary
/// at which `execute_recursive_scc` updates `StatsManager` for
/// a recursive predicate's `(full_rel, delta_rel)` RelIds.
/// Used by Part A + Part B tests to assert per-iteration
/// cardinality evolution + binary-join estimate evolution
/// without intrusive instrumentation.
#[cfg(feature = "recursive-stats-trace")]
#[derive(Debug, Default, Clone)]
#[allow(missing_docs)]
pub struct RecursiveStatsTrace {
    pub entries: Vec<RecursiveStatsTraceEntry>,
}

/// One entry per `(iteration, pred)` boundary.
///
/// `iteration == 0` is the seed pass; `iteration >= 1` is the
/// fixpoint loop. `phase` distinguishes the Phase 2 delta-
/// recording site from the Phase 4 full-recording site so Part
/// A's strict `>` assertions on `full_rows` only see Phase 4
/// snapshots (full_rel actually advanced) and Part A's
/// delta-evolves assertions only see Phase 2 snapshots.
#[cfg(feature = "recursive-stats-trace")]
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct RecursiveStatsTraceEntry {
    pub iteration: usize,
    pub pred: String,
    pub full_rel: RelId,
    pub delta_rel: RelId,
    pub full_rows: u64,
    pub delta_rows: u64,
    pub phase: RecursiveStatsPhase,
    /// Optional binary-join estimate the cost model would use
    /// for the variant body's first binary hop. Triangle:
    /// `(delta_e1_rel, e2_rel, &[1], &[0])`. 4-cycle: same
    /// `(delta_e1_rel, e2_rel, &[1], &[0])` (slot 0 → slot 1
    /// adjacency on the X variable).
    pub binary_est_for_variant: Option<u64>,
}

#[cfg(feature = "recursive-stats-trace")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum RecursiveStatsPhase {
    /// Seed pass — full_rel + delta_rel both updated; trace
    /// entry contains both row counts. iteration == 0.
    Seed,
    /// Fixpoint loop Phase 2 — delta_rel updated; full_rel
    /// holds the previous iteration's value. Trace entry
    /// reports `full_rows` as the previous-iter card it sees.
    Phase2Delta,
    /// Fixpoint loop Phase 4 — full_rel updated post-merge.
    /// Trace entry reports the new full row count + the
    /// delta_rel value Phase 2 just recorded.
    Phase4Full,
}

impl Executor {
    /// Create a new executor with the given kernel provider
    ///
    /// # Arguments
    /// * `provider` - The CUDA kernel provider for GPU operations
    pub fn new(provider: Arc<CudaKernelProvider>) -> Self {
        Self::new_with_config(provider, RuntimeConfig::default())
    }

    /// Create a new executor with the given kernel provider and runtime config
    pub fn new_with_config(provider: Arc<CudaKernelProvider>, config: RuntimeConfig) -> Self {
        const DEFAULT_JOIN_INDEX_CACHE_BYTES: u64 = 256 * 1024 * 1024;
        let max_index_cache_bytes =
            (provider.memory().budget().device_bytes / 4).min(DEFAULT_JOIN_INDEX_CACHE_BYTES);
        Self {
            provider: provider.clone(),
            store: RelationStore::new(provider.clone()),
            rel_names: HashMap::new(),
            name_to_rel: HashMap::new(),
            stats: StatsManager::new(),
            join_index_cache: JoinIndexCache::new(max_index_cache_bytes),
            common_subexpression_cache: HashMap::new(),
            common_subexpression_stats: CommonSubexpressionStats::default(),
            adaptive_reoptimization_stats: AdaptiveReoptimizationStats::default(),
            adaptive_join_observations: Vec::new(),
            config,
            profiler: Profiler::default(),
            ilp_registry: IlpRegistry::new(),
            ilp_last_result: None,
            wcoj_triangle_dispatch_count: 0,
            wcoj_4cycle_dispatch_count: 0,
            w63_chain_dispatch_count: 0,
            wcoj_clique5_dispatch_count: 0,
            wcoj_clique6_dispatch_count: 0,
            wcoj_clique7_dispatch_count: 0,
            wcoj_clique8_dispatch_count: 0,
            kclique_histogram_refresh_count: 0,
            kclique_histogram_refresh_nanos: 0,
            nested_loop_dispatch_count: 0,
            wcoj_error_decline_count: 0,
            wcoj_groupby_fusion_dispatch_count: 0,
            free_join_dispatch_count: 0,
            factorized_delta_dispatch_count: 0,
            wcoj_dispatch_stream: OnceLock::new(),
            #[cfg(feature = "wcoj-phase-timing")]
            last_wcoj_phase_timing: std::sync::Mutex::new(None),
            #[cfg(feature = "recursive-stats-trace")]
            last_recursive_stats_trace: RecursiveStatsTrace::default(),
        }
    }

    /// W2.3 Part A + Part B test seam — return the most recent
    /// recursive-SCC stats trace populated by
    /// `execute_recursive_scc`. Gated on the
    /// `recursive-stats-trace` Cargo feature; default OFF.
    #[cfg(feature = "recursive-stats-trace")]
    pub fn last_recursive_stats_trace(&self) -> &RecursiveStatsTrace {
        &self.last_recursive_stats_trace
    }

    /// Take the most recent WCOJ triangle dispatch's per-phase
    /// timing breakdown. Reading clears the slot — designed for
    /// one-shot consumption by the `wcoj_phase_report` binary.
    /// Returns `None` if no triangle has dispatched since the
    /// last read (or since construction).
    ///
    /// Compiled in only with the `wcoj-phase-timing` Cargo
    /// feature; production builds have no such method.
    #[cfg(feature = "wcoj-phase-timing")]
    pub fn take_wcoj_phase_timing(&self) -> Option<wcoj_phase_timing::WcojDispatchPhaseTiming> {
        self.last_wcoj_phase_timing
            .lock()
            .ok()
            .and_then(|mut g| g.take())
    }

    /// Enable or disable the performance profiler
    ///
    /// When enabled, execution statistics will be collected for --stats output.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiler = Profiler::new(enabled);
        if enabled {
            self.profiler
                .set_memory_budget(self.provider.memory().budget().device_bytes);
        }
    }

    /// Check if profiling is enabled
    pub fn is_profiling(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Get execution statistics
    ///
    /// Returns collected statistics if profiling was enabled.
    pub fn execution_stats(&self, total_output_rows: u64) -> ExecutionStats {
        self.profiler.execution_stats(total_output_rows)
    }

    /// Get a reference to the relation store
    pub fn store(&self) -> &RelationStore {
        &self.store
    }

    /// Get a mutable reference to the relation store
    pub fn store_mut(&mut self) -> &mut RelationStore {
        &mut self.store
    }

    /// Get a mutable reference to the ILP registry (RD-35).
    pub fn ilp_registry_mut(&mut self) -> &mut IlpRegistry {
        &mut self.ilp_registry
    }

    /// Get a shared reference to the ILP registry.
    pub fn ilp_registry(&self) -> &IlpRegistry {
        &self.ilp_registry
    }

    /// Get the last ILP tagged result (RD-35).
    pub fn ilp_last_result(&self) -> Option<&IlpTaggedResult> {
        self.ilp_last_result.as_ref()
    }

    /// Store a relation buffer and invalidate join indices.
    pub fn put_relation(&mut self, name: &str, buffer: CudaBuffer) {
        self.store_put(name, buffer);
    }

    /// Get a reference to the runtime statistics manager
    pub fn stats(&self) -> &StatsManager {
        &self.stats
    }

    /// Return persistent join-index manager telemetry.
    pub fn join_index_cache_stats(&self) -> JoinIndexCacheStats {
        self.join_index_cache.stats()
    }

    /// Reset executor state for Monte Carlo sampling.
    ///
    /// Clears relation storage and join index cache while preserving relation registrations.
    pub fn reset_for_mc(&mut self) {
        self.store.clear();
        self.join_index_cache.clear();
        self.common_subexpression_cache.clear();
        self.adaptive_join_observations.clear();
    }

    /// Targeted MC reset: preserve base/static relations and clear dynamic ones.
    ///
    /// Unlike [`Self::reset_for_mc`] which drops all relations, this method keeps the
    /// relations listed in `preserve` untouched, removes every other relation,
    /// then re-creates the relations specified in `clear_to_empty` as empty
    /// GPU buffers with the given schemas.  The join-index cache is fully
    /// invalidated because dynamic relations have changed.
    ///
    /// # Arguments
    /// * `preserve` - Relation names to keep as-is (base/static facts).
    /// * `clear_to_empty` - `(name, schema)` pairs for dynamic relations that
    ///   should be present but empty after the reset.
    pub fn reset_for_mc_relations(
        &mut self,
        preserve: &[&str],
        clear_to_empty: &[(&str, Schema)],
    ) -> Result<()> {
        let preserve_set: HashSet<&str> = preserve.iter().copied().collect();
        let existing_names: Vec<String> = self.store.names().map(|s| s.to_string()).collect();

        for name in &existing_names {
            if !preserve_set.contains(name.as_str()) {
                self.store.remove(name);
            }
        }

        for (name, schema) in clear_to_empty {
            let empty = self.provider.create_empty_buffer(schema.clone())?;
            self.store.put(name, empty);
        }

        self.join_index_cache.clear();
        self.common_subexpression_cache.clear();
        self.adaptive_join_observations.clear();
        Ok(())
    }

    /// Reset executor state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks + tagged results), relation storage,
    /// join index cache, stats, and profiler. Preserves relation name
    /// registrations (rel_names, name_to_rel) since those are immutable
    /// compile artifacts.
    pub fn reset_for_ilp(&mut self) {
        self.ilp_registry.clear();
        self.ilp_last_result = None;
        self.store.clear();
        self.join_index_cache.clear();
        self.common_subexpression_cache.clear();
        self.adaptive_join_observations.clear();
        self.stats = StatsManager::new();
        self.profiler = Profiler::default();
    }

    /// Get a mutable reference to the runtime statistics manager
    pub fn stats_mut(&mut self) -> &mut StatsManager {
        &mut self.stats
    }

    /// Capture a runtime statistics snapshot, including predicate name mappings.
    ///
    /// Use this snapshot to seed the compiler/optimizer on subsequent compilations.
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        let mut snapshot = self.stats.snapshot();
        snapshot.rel_names = self
            .rel_names
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        snapshot
    }

    /// Return runtime CSE telemetry for evidence and diagnostics.
    pub fn common_subexpression_stats(&self) -> &CommonSubexpressionStats {
        &self.common_subexpression_stats
    }

    /// Return adaptive re-optimization telemetry for evidence and diagnostics.
    pub fn adaptive_reoptimization_stats(&self) -> &AdaptiveReoptimizationStats {
        &self.adaptive_reoptimization_stats
    }

    /// Replay the deterministic adaptive decision against captured telemetry.
    pub fn replay_adaptive_reoptimization_decision(
        &self,
        observations: &[AdaptiveJoinObservation],
    ) -> AdaptiveReoptimizationDecision {
        self.adaptive_reoptimization_decision(observations)
    }

    fn common_subexpression_enabled(&self) -> bool {
        self.config.resolved_common_subexpression_elimination()
    }

    fn adaptive_reoptimization_enabled(&self) -> bool {
        self.config.resolved_adaptive_reoptimization()
    }

    fn adaptive_reoptimization_decision(
        &self,
        observations: &[AdaptiveJoinObservation],
    ) -> AdaptiveReoptimizationDecision {
        let min_misplan_ratio = self
            .config
            .resolved_adaptive_reoptimization_min_misplan_ratio();
        let max_misplan_ratio = observations
            .iter()
            .map(|observation| observation.misplan_ratio)
            .fold(1.0_f64, f64::max);

        if !self.adaptive_reoptimization_enabled() {
            return AdaptiveReoptimizationDecision {
                action: AdaptiveReoptimizationAction::Disabled,
                reason: "adaptive_reoptimization_disabled".to_string(),
                max_misplan_ratio,
                min_misplan_ratio,
            };
        }

        if observations.is_empty() {
            return AdaptiveReoptimizationDecision {
                action: AdaptiveReoptimizationAction::Skipped,
                reason: "no_join_telemetry".to_string(),
                max_misplan_ratio,
                min_misplan_ratio,
            };
        }

        if max_misplan_ratio >= min_misplan_ratio {
            AdaptiveReoptimizationDecision {
                action: AdaptiveReoptimizationAction::AttemptCandidate,
                reason: "misplan_threshold_crossed".to_string(),
                max_misplan_ratio,
                min_misplan_ratio,
            }
        } else {
            AdaptiveReoptimizationDecision {
                action: AdaptiveReoptimizationAction::Skipped,
                reason: "misplan_threshold_not_crossed".to_string(),
                max_misplan_ratio,
                min_misplan_ratio,
            }
        }
    }

    fn record_adaptive_join_observation(
        &mut self,
        left_rel: RelId,
        right_rel: RelId,
        left_keys: &[usize],
        right_keys: &[usize],
        input_rows: u64,
        actual_output_rows: u64,
    ) {
        let estimated_output_rows = self
            .stats
            .estimate_join_cardinality(left_rel, right_rel, left_keys, right_keys);
        let estimated_selectivity = if input_rows > 0 {
            estimated_output_rows as f64 / input_rows as f64
        } else {
            0.0
        };
        let actual_selectivity = if input_rows > 0 {
            actual_output_rows as f64 / input_rows as f64
        } else {
            0.0
        };
        let cardinality_delta_abs = estimated_output_rows.abs_diff(actual_output_rows);
        let selectivity_delta_abs = (estimated_selectivity - actual_selectivity).abs();
        let left_heat = self
            .stats
            .get_relation_stats(left_rel)
            .map(|stats| stats.heat)
            .unwrap_or(0.0);
        let right_heat = self
            .stats
            .get_relation_stats(right_rel)
            .map(|stats| stats.heat)
            .unwrap_or(0.0);
        let heat_delta_abs = (left_heat - right_heat).abs();
        let smaller = estimated_output_rows.min(actual_output_rows);
        let larger = estimated_output_rows.max(actual_output_rows);
        let misplan_ratio = if smaller == 0 {
            if larger == 0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            (larger as f64 / smaller as f64).max(1.0)
        };

        self.adaptive_join_observations
            .push(AdaptiveJoinObservation {
                left_rel,
                right_rel,
                estimated_output_rows,
                actual_output_rows,
                cardinality_delta_abs,
                estimated_selectivity,
                actual_selectivity,
                selectivity_delta_abs,
                left_heat,
                right_heat,
                heat_delta_abs,
                misplan_ratio,
            });
    }

    fn plan_head_names(plan: &ExecutionPlan) -> Vec<String> {
        let mut names = Vec::new();
        for stratum in &plan.strata {
            for scc_id in &stratum.sccs {
                if let Some(rules) = plan.rules_by_scc.get(*scc_id as usize) {
                    for rule in rules {
                        if !names.iter().any(|name| name == &rule.head) {
                            names.push(rule.head.clone());
                        }
                    }
                }
            }
        }

        if names.is_empty() {
            for rules in &plan.rules_by_scc {
                for rule in rules {
                    if !names.iter().any(|name| name == &rule.head) {
                        names.push(rule.head.clone());
                    }
                }
            }
        }

        names
    }

    fn clone_store_snapshot(&self) -> Result<HashMap<String, CudaBuffer>> {
        let names: Vec<String> = self.store.names().map(|name| name.to_string()).collect();
        let mut snapshot = HashMap::with_capacity(names.len());
        for name in names {
            if let Some(buffer) = self.store.get(&name) {
                snapshot.insert(name, self.clone_buffer(buffer)?);
            }
        }
        Ok(snapshot)
    }

    fn restore_store_snapshot(&mut self, snapshot: HashMap<String, CudaBuffer>) {
        let snapshot_names: HashSet<String> = snapshot.keys().cloned().collect();
        let existing_names: Vec<String> = self.store.names().map(|name| name.to_string()).collect();
        for name in existing_names {
            if !snapshot_names.contains(&name) {
                self.store.remove(&name);
            }
        }
        for (name, buffer) in snapshot {
            self.store.put(&name, buffer);
        }
    }

    fn restore_stats_snapshot(&mut self, snapshot: &StatsSnapshot) {
        self.stats.clear();
        self.stats.merge_snapshot(snapshot);
    }

    fn clone_final_plan_output(&self, plan: &ExecutionPlan) -> Result<CudaBuffer> {
        let head_names = Self::plan_head_names(plan);
        if let Some(name) = head_names.last() {
            let output = self.store.get(name).ok_or_else(|| {
                XlogError::Execution(format!("adaptive reoptimization output missing: {name}"))
            })?;
            return self.clone_buffer(output);
        }

        self.provider.create_empty_buffer(Schema::new(vec![]))
    }

    fn plan_outputs_match(
        &self,
        head_names: &[String],
        baseline_snapshot: &HashMap<String, CudaBuffer>,
    ) -> Result<bool> {
        for name in head_names {
            let Some(baseline) = baseline_snapshot.get(name) else {
                return Ok(false);
            };
            let Some(candidate) = self.store.get(name) else {
                return Ok(false);
            };
            if !self.buffers_gpu_set_equivalent(baseline, candidate)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn buffers_gpu_set_equivalent(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<bool> {
        if left.schema() != right.schema() {
            return Ok(false);
        }
        let left_rows = self.provider.device_row_count(left)?;
        let right_rows = self.provider.device_row_count(right)?;
        if left_rows != right_rows {
            return Ok(false);
        }

        let left_minus_right = self.provider.diff_full_row(left, right)?;
        if self.provider.device_row_count(&left_minus_right)? != 0 {
            return Ok(false);
        }
        let right_minus_left = self.provider.diff_full_row(right, left)?;
        Ok(self.provider.device_row_count(&right_minus_left)? == 0)
    }

    fn record_adaptive_dtoh_delta(&mut self, before_dtoh_calls: u64) {
        let after_dtoh_calls = self.provider.host_transfer_stats().dtoh_calls;
        self.adaptive_reoptimization_stats.data_plane_dtoh_calls =
            after_dtoh_calls.saturating_sub(before_dtoh_calls);
    }

    fn is_common_subexpression_cacheable(node: &RirNode) -> bool {
        !matches!(node, RirNode::Unit | RirNode::Scan { .. })
    }

    fn record_common_subexpression_rejection(&mut self, reason: &'static str) {
        self.common_subexpression_stats.unsafe_rejections = self
            .common_subexpression_stats
            .unsafe_rejections
            .saturating_add(1);
        if !self
            .common_subexpression_stats
            .rejection_reasons
            .iter()
            .any(|seen| seen == reason)
        {
            self.common_subexpression_stats
                .rejection_reasons
                .push(reason.to_string());
        }
    }

    fn common_subexpression_key(&mut self, node: &RirNode) -> Option<CommonSubexpressionKey> {
        match node {
            RirNode::Unit => None,
            RirNode::Scan { rel } => {
                let generation = self
                    .get_rel_name(*rel)
                    .and_then(|name| self.store.version(name))
                    .unwrap_or(0);
                Some(CommonSubexpressionKey::Scan {
                    rel: *rel,
                    generation,
                })
            }
            RirNode::Filter { input, predicate } => {
                let input = self.common_subexpression_key(input)?;
                Some(CommonSubexpressionKey::Filter {
                    input: Box::new(input),
                    predicate: Self::expr_cse_key(predicate),
                })
            }
            RirNode::Project { input, columns } => {
                let input = self.common_subexpression_key(input)?;
                Some(CommonSubexpressionKey::Project {
                    input: Box::new(input),
                    columns: columns.iter().map(Self::project_expr_cse_key).collect(),
                })
            }
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                if *join_type != JoinType::Inner {
                    self.record_common_subexpression_rejection("negation_or_outer_join_boundary");
                    return None;
                }
                let left = self.common_subexpression_key(left)?;
                let right = self.common_subexpression_key(right)?;
                Some(CommonSubexpressionKey::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    left_keys: left_keys.clone(),
                    right_keys: right_keys.clone(),
                })
            }
            RirNode::Union { inputs } => {
                let mut input_keys = Vec::with_capacity(inputs.len());
                for input in inputs {
                    input_keys.push(self.common_subexpression_key(input)?);
                }
                Some(CommonSubexpressionKey::Union { inputs: input_keys })
            }
            RirNode::Distinct { input, key_cols } => {
                let input = self.common_subexpression_key(input)?;
                Some(CommonSubexpressionKey::Distinct {
                    input: Box::new(input),
                    key_cols: key_cols.clone(),
                })
            }
            RirNode::Diff { .. } => {
                self.record_common_subexpression_rejection("negation_or_difference_boundary");
                None
            }
            RirNode::GroupBy { .. } => {
                self.record_common_subexpression_rejection("aggregate_boundary");
                None
            }
            RirNode::Fixpoint { .. } => {
                self.record_common_subexpression_rejection("recursive_or_mutable_boundary");
                None
            }
            RirNode::TensorMaskedJoin { .. } => {
                self.record_common_subexpression_rejection("provenance_or_tensor_boundary");
                None
            }
            RirNode::MultiWayJoin { .. } | RirNode::ChainJoin { .. } => {
                self.record_common_subexpression_rejection("specialized_dispatch_boundary");
                None
            }
        }
    }

    fn expr_cse_key(expr: &Expr) -> String {
        match expr {
            Expr::Column(idx) => format!("col:{idx}"),
            Expr::Const(value) => format!("const:{}", Self::const_cse_key(value)),
            Expr::Compare { left, op, right } => format!(
                "cmp:{}:{}:{}",
                Self::expr_cse_key(left),
                Self::compare_op_cse_key(*op),
                Self::expr_cse_key(right)
            ),
            Expr::And(items) => format!(
                "and:[{}]",
                items
                    .iter()
                    .map(Self::expr_cse_key)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Expr::Or(items) => format!(
                "or:[{}]",
                items
                    .iter()
                    .map(Self::expr_cse_key)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Expr::Not(inner) => format!("not:{}", Self::expr_cse_key(inner)),
            Expr::Add(left, right) => {
                format!(
                    "add:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Sub(left, right) => {
                format!(
                    "sub:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Mul(left, right) => {
                format!(
                    "mul:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Div(left, right) => {
                format!(
                    "div:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Mod(left, right) => {
                format!(
                    "mod:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Abs(inner) => format!("abs:{}", Self::expr_cse_key(inner)),
            Expr::Min(left, right) => {
                format!(
                    "min:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Max(left, right) => {
                format!(
                    "max:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Pow(left, right) => {
                format!(
                    "pow:{}:{}",
                    Self::expr_cse_key(left),
                    Self::expr_cse_key(right)
                )
            }
            Expr::Cast(inner, ty) => format!("cast:{:?}:{}", ty, Self::expr_cse_key(inner)),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => format!(
                "if:{}:{}:{}",
                Self::expr_cse_key(condition),
                Self::expr_cse_key(then_expr),
                Self::expr_cse_key(else_expr)
            ),
        }
    }

    fn project_expr_cse_key(expr: &ProjectExpr) -> String {
        match expr {
            ProjectExpr::Column(idx) => format!("col:{idx}"),
            ProjectExpr::Computed(expr, ty) => {
                format!("computed:{:?}:{}", ty, Self::expr_cse_key(expr))
            }
        }
    }

    fn const_cse_key(value: &xlog_ir::ConstValue) -> String {
        match value {
            xlog_ir::ConstValue::U32(value) => format!("u32:{value}"),
            xlog_ir::ConstValue::U64(value) => format!("u64:{value}"),
            xlog_ir::ConstValue::I32(value) => format!("i32:{value}"),
            xlog_ir::ConstValue::I64(value) => format!("i64:{value}"),
            xlog_ir::ConstValue::F32(value) => format!("f32:{:08x}", value.to_bits()),
            xlog_ir::ConstValue::F64(value) => format!("f64:{:016x}", value.to_bits()),
            xlog_ir::ConstValue::Bool(value) => format!("bool:{value}"),
            xlog_ir::ConstValue::Symbol(value) => format!("symbol:{value:?}"),
        }
    }

    fn compare_op_cse_key(op: xlog_ir::CompareOp) -> &'static str {
        match op {
            xlog_ir::CompareOp::Eq => "eq",
            xlog_ir::CompareOp::Ne => "ne",
            xlog_ir::CompareOp::Lt => "lt",
            xlog_ir::CompareOp::Le => "le",
            xlog_ir::CompareOp::Gt => "gt",
            xlog_ir::CompareOp::Ge => "ge",
        }
    }

    fn store_put(&mut self, name: &str, buffer: CudaBuffer) {
        self.common_subexpression_cache.clear();
        self.store.put(name, buffer);
        if let Some(&rel_id) = self.name_to_rel.get(name) {
            self.join_index_cache.invalidate_rel(rel_id);
        }
    }

    fn store_remove(&mut self, name: &str) -> Option<CudaBuffer> {
        self.common_subexpression_cache.clear();
        if let Some(&rel_id) = self.name_to_rel.get(name) {
            self.join_index_cache.invalidate_rel(rel_id);
        }
        self.store.remove(name)
    }

    /// Register a relation name for a RelId
    ///
    /// This mapping is used when executing Scan nodes to look up relations
    /// by their RelId.
    ///
    /// # Arguments
    /// * `rel_id` - The relation identifier
    /// * `name` - The name to associate with the relation
    pub fn register_relation(&mut self, rel_id: RelId, name: &str) {
        self.rel_names.insert(rel_id, name.to_string());
        self.name_to_rel.insert(name.to_string(), rel_id);
        self.stats.register_relation(rel_id);
    }

    /// W2.3: reverse-lookup a RelId by predicate name. Used by
    /// `execute_recursive_scc` to resolve a recursive predicate's
    /// full-rel RelId for `StatsManager::update_cardinality`
    /// calls at iteration boundaries. Returns `None` for
    /// unregistered names (defensive — production callers
    /// register IDB heads before `execute_plan`; tests that
    /// omit registration get a no-op stats update).
    fn name_to_rel_id(&self, name: &str) -> Option<RelId> {
        self.name_to_rel.get(name).copied()
    }

    /// Get the relation name for a RelId
    fn get_rel_name(&self, rel_id: RelId) -> Option<&str> {
        self.rel_names.get(&rel_id).map(|s| s.as_str())
    }

    /// Execute a baseline plan and conditionally adopt a compiler-supplied
    /// re-optimized candidate plan.
    ///
    /// The baseline runs through the normal [`Self::execute_plan`] path first,
    /// producing runtime join telemetry. If deterministic mis-plan thresholds
    /// fire and adaptive re-optimization is enabled, the candidate also runs
    /// through [`Self::execute_plan`]. Candidate outputs are compared on the GPU
    /// with deterministic full-row set difference; divergent or failing
    /// candidates roll back to the baseline relation/statistics snapshot.
    pub fn execute_plan_with_adaptive_candidate(
        &mut self,
        baseline_plan: &ExecutionPlan,
        candidate_plan: &ExecutionPlan,
    ) -> Result<CudaBuffer> {
        self.adaptive_reoptimization_stats.invocations = self
            .adaptive_reoptimization_stats
            .invocations
            .saturating_add(1);
        self.adaptive_reoptimization_stats.diagnostics.clear();
        let before_dtoh_calls = self.provider.host_transfer_stats().dtoh_calls;

        self.execute_plan(baseline_plan)?;
        let baseline_observations = self.adaptive_join_observations.clone();
        self.adaptive_reoptimization_stats.last_observations = baseline_observations.clone();
        let decision = self.adaptive_reoptimization_decision(&baseline_observations);
        self.adaptive_reoptimization_stats.last_decision = Some(decision.clone());

        match decision.action {
            AdaptiveReoptimizationAction::Disabled => {
                self.adaptive_reoptimization_stats.disabled = self
                    .adaptive_reoptimization_stats
                    .disabled
                    .saturating_add(1);
                self.record_adaptive_dtoh_delta(before_dtoh_calls);
                return self.clone_final_plan_output(baseline_plan);
            }
            AdaptiveReoptimizationAction::Skipped => {
                self.adaptive_reoptimization_stats.skipped =
                    self.adaptive_reoptimization_stats.skipped.saturating_add(1);
                self.record_adaptive_dtoh_delta(before_dtoh_calls);
                return self.clone_final_plan_output(baseline_plan);
            }
            AdaptiveReoptimizationAction::AttemptCandidate => {}
            AdaptiveReoptimizationAction::Adopted | AdaptiveReoptimizationAction::RolledBack => {
                unreachable!("decision replay never returns terminal adaptive actions")
            }
        }

        let head_names = Self::plan_head_names(baseline_plan);
        let baseline_snapshot = self.clone_store_snapshot()?;
        let baseline_stats_snapshot = self.stats_snapshot();

        if let Err(err) = self.execute_plan(candidate_plan) {
            self.restore_store_snapshot(baseline_snapshot);
            self.restore_stats_snapshot(&baseline_stats_snapshot);
            self.adaptive_reoptimization_stats.rolled_back = self
                .adaptive_reoptimization_stats
                .rolled_back
                .saturating_add(1);
            self.adaptive_reoptimization_stats
                .diagnostics
                .push(AdaptiveReoptimizationDiagnostic {
                    kind: AdaptiveReoptimizationDiagnosticKind::CandidateExecutionFailed,
                    message: err.to_string(),
                });
            self.adaptive_reoptimization_stats
                .diagnostics
                .push(AdaptiveReoptimizationDiagnostic {
                    kind: AdaptiveReoptimizationDiagnosticKind::RollbackRestoredBaseline,
                    message: "baseline_snapshot_restored".to_string(),
                });
            self.adaptive_reoptimization_stats.last_observations = baseline_observations;
            self.adaptive_reoptimization_stats.last_decision =
                Some(AdaptiveReoptimizationDecision {
                    action: AdaptiveReoptimizationAction::RolledBack,
                    reason: "candidate_execution_failed".to_string(),
                    max_misplan_ratio: decision.max_misplan_ratio,
                    min_misplan_ratio: decision.min_misplan_ratio,
                });
            self.record_adaptive_dtoh_delta(before_dtoh_calls);
            return self.clone_final_plan_output(baseline_plan);
        }

        if !self.plan_outputs_match(&head_names, &baseline_snapshot)? {
            self.restore_store_snapshot(baseline_snapshot);
            self.restore_stats_snapshot(&baseline_stats_snapshot);
            self.adaptive_reoptimization_stats.rolled_back = self
                .adaptive_reoptimization_stats
                .rolled_back
                .saturating_add(1);
            self.adaptive_reoptimization_stats
                .diagnostics
                .push(AdaptiveReoptimizationDiagnostic {
                    kind: AdaptiveReoptimizationDiagnosticKind::CandidateOutputMismatch,
                    message: "candidate_output_mismatch".to_string(),
                });
            self.adaptive_reoptimization_stats
                .diagnostics
                .push(AdaptiveReoptimizationDiagnostic {
                    kind: AdaptiveReoptimizationDiagnosticKind::RollbackRestoredBaseline,
                    message: "baseline_snapshot_restored".to_string(),
                });
            self.adaptive_reoptimization_stats.last_observations = baseline_observations;
            self.adaptive_reoptimization_stats.last_decision =
                Some(AdaptiveReoptimizationDecision {
                    action: AdaptiveReoptimizationAction::RolledBack,
                    reason: "candidate_output_mismatch".to_string(),
                    max_misplan_ratio: decision.max_misplan_ratio,
                    min_misplan_ratio: decision.min_misplan_ratio,
                });
            self.record_adaptive_dtoh_delta(before_dtoh_calls);
            return self.clone_final_plan_output(baseline_plan);
        }

        self.adaptive_reoptimization_stats.adopted =
            self.adaptive_reoptimization_stats.adopted.saturating_add(1);
        self.adaptive_reoptimization_stats.last_observations = baseline_observations;
        self.adaptive_reoptimization_stats.last_decision = Some(AdaptiveReoptimizationDecision {
            action: AdaptiveReoptimizationAction::Adopted,
            reason: "candidate_adopted".to_string(),
            max_misplan_ratio: decision.max_misplan_ratio,
            min_misplan_ratio: decision.min_misplan_ratio,
        });
        self.record_adaptive_dtoh_delta(before_dtoh_calls);
        self.clone_final_plan_output(candidate_plan)
    }

    /// Execute a complete execution plan
    ///
    /// Iterates through strata in order, executing each one.
    /// Returns the result of the final query if present, or an empty buffer.
    ///
    /// # Arguments
    /// * `plan` - The execution plan to execute
    ///
    /// # Returns
    /// The result buffer from executing the plan
    ///
    /// # Errors
    /// Returns an error if any stratum or query execution fails
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<CudaBuffer> {
        self.adaptive_join_observations.clear();
        self.common_subexpression_cache.clear();
        // Opt-in deterministic-Datalog D2H gate. Enabled only for the
        // duration of this call; the provider is shared so we restore the
        // prior state on every exit path (including errors). This PR ships
        // the gate as opt-in only — known violating relational paths
        // (set difference, binary-join count/materialize) are scheduled for
        // replacement before the default flips.
        let gate = self.config.strict_deterministic_d2h;
        let prev_gate = self.provider.strict_deterministic_d2h_enabled();
        if gate && !prev_gate {
            // Only reset the violation counter when *this* call is what
            // engages the gate. If a caller has manually enabled the
            // gate to accumulate violations across a broader strict
            // section, we must not clobber their telemetry.
            self.provider.reset_deterministic_d2h_violations();
            self.provider.enable_strict_deterministic_d2h();
        }
        // Cloning the Arc keeps the guard independent of `self`, so the
        // guard can coexist with `&mut self` calls inside the strata loop.
        let _gate_guard = D2hGateGuard {
            provider: Arc::clone(&self.provider),
            engaged: gate,
            previous: prev_gate,
        };

        // Execute strata in order
        for (idx, stratum) in plan.strata.iter().enumerate() {
            // Count rules and check if recursive
            let (num_rules, is_recursive) = stratum
                .sccs
                .iter()
                .map(|&scc_id| {
                    let rules = plan
                        .rules_by_scc
                        .get(scc_id as usize)
                        .map(|r| r.len())
                        .unwrap_or(0);
                    let recursive = plan
                        .sccs
                        .get(scc_id as usize)
                        .map(|s| s.is_recursive)
                        .unwrap_or(false);
                    (rules, recursive)
                })
                .fold((0, false), |(r, rec), (nr, nrec)| (r + nr, rec || nrec));

            self.profiler.begin_stratum(idx, num_rules, is_recursive);
            self.execute_stratum_impl(stratum, plan)?;

            // Record peak memory after stratum
            let mem_bytes = self.provider.memory().allocated_bytes();
            self.profiler.record_peak_memory(mem_bytes);

            self.profiler.end_stratum();
        }

        // Ensure all GPU work completes before returning control to callers.
        self.provider.device().synchronize()?;
        self.adaptive_reoptimization_stats.last_observations =
            self.adaptive_join_observations.clone();

        // If there are no strata, return empty buffer
        self.provider.create_empty_buffer(Schema::new(vec![]))
    }

    /// Execute a stratum (public API)
    ///
    /// This method cannot be called directly because stratum execution requires
    /// access to the full ExecutionPlan (for rules_by_scc mapping). Use
    /// `execute_plan` instead, which processes all strata with proper context.
    ///
    /// # Arguments
    /// * `_stratum` - The stratum (unused - see error)
    ///
    /// # Returns
    /// Evaluate a predicate expression for a single row
    #[cfg(test)]
    fn evaluate_predicate(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<bool> {
        match expr {
            Expr::Column(col_idx) => {
                // Interpret column value as boolean
                let col_type = schema.column_type(*col_idx);
                if let Some(ScalarType::Bool) = col_type {
                    Ok(columns
                        .get(*col_idx)
                        .map(|c| c.get(row_idx).copied().unwrap_or(0) != 0)
                        .unwrap_or(false))
                } else {
                    // Non-bool columns: check if non-zero
                    Ok(true)
                }
            }

            Expr::Const(ConstValue::Bool(b)) => Ok(*b),
            Expr::Const(_) => Ok(true), // Non-bool constants are truthy

            Expr::Compare { left, op, right } => {
                let use_float =
                    Self::expr_may_be_float(left, schema) || Self::expr_may_be_float(right, schema);

                if use_float {
                    let left_val = Self::evaluate_expr_as_f64(left, columns, row_idx, schema)?;
                    let right_val = Self::evaluate_expr_as_f64(right, columns, row_idx, schema)?;

                    Ok(match op {
                        CompareOp::Eq => left_val == right_val,
                        CompareOp::Ne => left_val != right_val,
                        CompareOp::Lt => left_val < right_val,
                        CompareOp::Le => left_val <= right_val,
                        CompareOp::Gt => left_val > right_val,
                        CompareOp::Ge => left_val >= right_val,
                    })
                } else {
                    let left_val = Self::evaluate_expr_as_i64(left, columns, row_idx, schema)?;
                    let right_val = Self::evaluate_expr_as_i64(right, columns, row_idx, schema)?;

                    Ok(match op {
                        CompareOp::Eq => left_val == right_val,
                        CompareOp::Ne => left_val != right_val,
                        CompareOp::Lt => left_val < right_val,
                        CompareOp::Le => left_val <= right_val,
                        CompareOp::Gt => left_val > right_val,
                        CompareOp::Ge => left_val >= right_val,
                    })
                }
            }

            Expr::And(exprs) => {
                for e in exprs {
                    if !Self::evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            Expr::Or(exprs) => {
                for e in exprs {
                    if Self::evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            Expr::Not(inner) => Ok(!Self::evaluate_predicate(inner, columns, row_idx, schema)?),

            // Arithmetic expressions are not used as predicates directly
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Mod(_, _)
            | Expr::Abs(_)
            | Expr::Min(_, _)
            | Expr::Max(_, _)
            | Expr::Pow(_, _)
            | Expr::Cast(_, _)
            | Expr::Conditional { .. } => Err(XlogError::Execution(
                "Arithmetic expression cannot be evaluated as boolean predicate".into(),
            )),
        }
    }

    #[cfg(test)]
    fn evaluate_expr_as_f64(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<f64> {
        match expr {
            Expr::Column(col_idx) => {
                let col_type = schema.column_type(*col_idx).unwrap_or(ScalarType::U32);
                let col_data = columns
                    .get(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;

                let type_size = col_type.size_bytes();
                let start = row_idx * type_size;

                Ok(match col_type {
                    ScalarType::F64 => {
                        let bytes = &col_data[start..start + 8];
                        f64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    ScalarType::F32 => {
                        let bytes = &col_data[start..start + 4];
                        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::U32 | ScalarType::Symbol => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::I32 => {
                        let bytes = &col_data[start..start + 4];
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::U64 => {
                        let bytes = &col_data[start..start + 8];
                        u64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as f64
                    }
                    ScalarType::I64 => {
                        let bytes = &col_data[start..start + 8];
                        i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as f64
                    }
                    ScalarType::Bool => col_data.get(start).copied().unwrap_or(0) as f64,
                })
            }

            Expr::Const(val) => Ok(match val {
                ConstValue::U32(v) => *v as f64,
                ConstValue::I32(v) => *v as f64,
                ConstValue::U64(v) => *v as f64,
                ConstValue::I64(v) => *v as f64,
                ConstValue::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ConstValue::F32(f) => *f as f64,
                ConstValue::F64(f) => *f,
                ConstValue::Symbol(_) => {
                    return Err(XlogError::Execution(
                        "Cannot evaluate Symbol constant as f64".to_string(),
                    ));
                }
            }),

            Expr::Add(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                + Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Sub(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                - Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Mul(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                * Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Div(l, r) => {
                let left_val = Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?;
                if right_val == 0.0 {
                    return Err(XlogError::Execution("Division by zero".to_string()));
                }
                Ok(left_val / right_val)
            }
            Expr::Mod(l, r) => {
                let left_val = Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?;
                if right_val == 0.0 {
                    return Err(XlogError::Execution("Modulo by zero".to_string()));
                }
                Ok(left_val % right_val)
            }
            Expr::Abs(inner) => {
                Ok(Self::evaluate_expr_as_f64(inner, columns, row_idx, schema)?.abs())
            }
            Expr::Min(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                .min(Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?)),
            Expr::Max(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                .max(Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?)),
            Expr::Pow(base, exp) => Ok(Self::evaluate_expr_as_f64(base, columns, row_idx, schema)?
                .powf(Self::evaluate_expr_as_f64(exp, columns, row_idx, schema)?)),
            Expr::Cast(inner, target_type) => match target_type {
                ScalarType::F64 => Self::evaluate_expr_as_f64(inner, columns, row_idx, schema),
                ScalarType::F32 => {
                    Ok(Self::evaluate_expr_as_f64(inner, columns, row_idx, schema)? as f32 as f64)
                }
                _ => Ok(Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)? as f64),
            },

            _ => Err(XlogError::Execution(
                "Cannot evaluate compound expression as f64".to_string(),
            )),
        }
    }

    /// Evaluate an expression as an i64 value
    #[cfg(test)]
    fn evaluate_expr_as_i64(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<i64> {
        match expr {
            Expr::Column(col_idx) => {
                let col_type = schema.column_type(*col_idx).unwrap_or(ScalarType::U32);
                let col_data = columns
                    .get(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;

                let type_size = col_type.size_bytes();
                let start = row_idx * type_size;

                Ok(match col_type {
                    ScalarType::U32 => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::I32 => {
                        let bytes = &col_data[start..start + 4];
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::U64 => {
                        let bytes = &col_data[start..start + 8];
                        u64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as i64
                    }
                    ScalarType::I64 => {
                        let bytes = &col_data[start..start + 8];
                        i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    ScalarType::Bool => col_data.get(start).copied().unwrap_or(0) as i64,
                    ScalarType::Symbol => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::F32 => {
                        let bytes = &col_data[start..start + 4];
                        let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        val as i64
                    }
                    ScalarType::F64 => {
                        let bytes = &col_data[start..start + 8];
                        let val = f64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        val as i64
                    }
                })
            }

            Expr::Const(val) => Ok(match val {
                ConstValue::U32(v) => *v as i64,
                ConstValue::I32(v) => *v as i64,
                ConstValue::U64(v) => *v as i64,
                ConstValue::I64(v) => *v,
                ConstValue::Bool(b) => *b as i64,
                ConstValue::F32(f) => *f as i64,
                ConstValue::F64(f) => *f as i64,
                ConstValue::Symbol(_) => 0,
            }),

            // Arithmetic expressions - evaluate them and return the result
            Expr::Add(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_add(right_val))
            }
            Expr::Sub(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_sub(right_val))
            }
            Expr::Mul(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_mul(right_val))
            }
            Expr::Div(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                if right_val == 0 {
                    return Err(XlogError::Execution("Division by zero".to_string()));
                }
                Ok(left_val / right_val)
            }
            Expr::Mod(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                if right_val == 0 {
                    return Err(XlogError::Execution("Modulo by zero".to_string()));
                }
                Ok(left_val % right_val)
            }
            Expr::Abs(inner) => {
                let val = Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)?;
                Ok(val.abs())
            }
            Expr::Min(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.min(right_val))
            }
            Expr::Max(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.max(right_val))
            }
            Expr::Pow(base, exp) => {
                let base_val = Self::evaluate_expr_as_i64(base, columns, row_idx, schema)?;
                let exp_val = Self::evaluate_expr_as_i64(exp, columns, row_idx, schema)?;
                if exp_val < 0 {
                    Err(XlogError::Execution(
                        "Negative exponent in integer pow".to_string(),
                    ))
                } else if exp_val > u32::MAX as i64 {
                    // Exponent too large - would overflow anyway
                    Ok(i64::MAX)
                } else {
                    Ok(base_val.pow(exp_val as u32))
                }
            }
            Expr::Cast(inner, _target_type) => {
                // For i64 evaluation, cast is a no-op since we evaluate everything as i64
                Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)
            }

            _ => Err(XlogError::Execution(
                "Cannot evaluate compound expression as value".to_string(),
            )),
        }
    }

    /// Get the relation name for a RelId, creating a default name if not registered
    fn get_or_create_rel_name(&mut self, rel_id: RelId, default: &str) -> String {
        if let Some(name) = self.rel_names.get(&rel_id) {
            name.clone()
        } else {
            self.register_relation(rel_id, default);
            default.to_string()
        }
    }

    // ============== Helper methods ==============

    /// Create an empty buffer with the given schema
    fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
        self.provider.create_empty_buffer(schema)
    }

    /// Clone a buffer (device-to-device copy)
    fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        if buffer.is_empty() {
            return self.create_empty_buffer(buffer.schema().clone());
        }

        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col_type_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let bytes = (buffer.num_rows() as usize) * col_type_size;

            if let Some(src_col) = buffer.column(col_idx) {
                let mut dst_col = self.provider.memory().alloc::<u8>(bytes)?;
                if bytes > 0 {
                    self.provider
                        .device()
                        .inner()
                        .dtod_copy(src_col, &mut dst_col)
                        .map_err(|e| {
                            XlogError::Execution(format!("Failed to clone column on device: {}", e))
                        })?;
                }
                result_columns.push(dst_col.into());
            }
        }

        let d_num_rows = self.clone_device_row_count(buffer)?;
        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.num_rows(),
            d_num_rows,
            buffer.schema().clone(),
        ))
    }

    fn clone_device_row_count(&self, buffer: &CudaBuffer) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.provider.memory().alloc::<u32>(1)?;
        self.provider
            .device()
            .inner()
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Execution(format!("Failed to copy row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
        if let Some(n) = buffer.cached_row_count() {
            return Ok(n);
        }
        // Metadata-only read: row counts are control-plane state, not
        // tuple data. Route through `dtoh_scalar_untracked` so the
        // metadata-vs-data-plane contract stays grepable and the
        // deterministic-D2H gate continues to allow it. Re-map the
        // provider-level `XlogError::Kernel` into `XlogError::Execution`
        // with the executor's historical "Failed to read row count"
        // context so callers see a consistent error category.
        let n = self
            .provider
            .dtoh_scalar_untracked::<u32>(buffer.num_rows_device(), 0)
            .map_err(|e| XlogError::Execution(format!("Failed to read row count: {}", e)))?;
        buffer.set_cached_row_count_if_unset(n);
        Ok(n)
    }
}

/// RAII guard that restores the provider's deterministic-D2H gate state on
/// drop. Engaged only when `Executor::execute_plan` opted in via
/// `RuntimeConfig::strict_deterministic_d2h`.
struct D2hGateGuard {
    provider: Arc<CudaKernelProvider>,
    engaged: bool,
    previous: bool,
}

impl Drop for D2hGateGuard {
    fn drop(&mut self) {
        if !self.engaged {
            return;
        }
        if self.previous {
            self.provider.enable_strict_deterministic_d2h();
        } else {
            self.provider.disable_strict_deterministic_d2h();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use xlog_core::MemoryBudget;
    use xlog_cuda::{CudaDevice, GpuMemoryManager};
    use xlog_ir::{CompiledRule, RirMeta, Scc};

    fn has_cuda_device() -> bool {
        // Check if CUDA device is available using CudaDevice wrapper
        CudaDevice::new(0).is_ok()
    }

    fn create_test_executor() -> Option<Executor> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
        Some(Executor::new(provider))
    }

    fn create_test_executor_with_config(config: RuntimeConfig) -> Option<Executor> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
        Some(Executor::new_with_config(provider, config))
    }

    fn device_row_count(executor: &Executor, rows: u64) -> TrackedCudaSlice<u32> {
        let rows_u32 = u32::try_from(rows).expect("row count fits u32");
        let mut d_num_rows = executor.provider.memory().alloc::<u32>(1).expect("alloc");
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
            .expect("htod");
        d_num_rows
    }

    fn create_test_buffer(executor: &Executor, data: &[u32], col_name: &str) -> CudaBuffer {
        let schema = Schema::new(vec![(col_name.to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col = executor
            .provider
            .memory()
            .alloc::<u8>(bytes.len())
            .expect("alloc");
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");

        let rows = data.len() as u64;
        let d_num_rows = device_row_count(executor, rows);
        CudaBuffer::from_columns(vec![col.into()], rows, d_num_rows, schema)
    }

    fn read_buffer_u32(executor: &Executor, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
        executor
            .provider
            .download_column::<u32>(buffer, col)
            .unwrap_or_default()
    }

    fn buffer_row_count(executor: &Executor, buffer: &CudaBuffer) -> u32 {
        if let Some(n) = buffer.cached_row_count() {
            return n;
        }
        let mut host_rows = [0u32];
        executor
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .expect("dtoh row count");
        buffer.set_cached_row_count_if_unset(host_rows[0]);
        host_rows[0]
    }

    fn to_f64_column_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn to_f32_column_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ============== Basic Executor Tests ==============

    #[test]
    fn test_executor_creation() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        assert!(executor.store().is_empty());
    }

    #[test]
    fn test_predicate_f64_comparisons() {
        let schema = Schema::new(vec![("x".to_string(), ScalarType::F64)]);
        let values = [1.0f64, 2.0, 3.0, f64::NAN];
        let columns = vec![to_f64_column_bytes(&values)];

        let gt_two = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::F64(2.0))),
        };

        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&gt_two, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, true, false]);

        let eq_nan = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::F64(f64::NAN))),
        };
        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&eq_nan, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, false, false]);

        let ne_nan = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Ne,
            right: Box::new(Expr::Const(ConstValue::F64(f64::NAN))),
        };
        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&ne_nan, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, true, true]);
    }

    #[test]
    fn test_predicate_f32_comparisons() {
        let schema = Schema::new(vec![("x".to_string(), ScalarType::F32)]);
        let values = [1.0f32, 2.0, 3.0, f32::NAN];
        let columns = vec![to_f32_column_bytes(&values)];

        let le_two = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Le,
            right: Box::new(Expr::Const(ConstValue::F32(2.0))),
        };

        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&le_two, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_predicate_mixed_float_int_comparisons() {
        let schema = Schema::new(vec![
            ("x".to_string(), ScalarType::F64),
            ("y".to_string(), ScalarType::U32),
        ]);

        let x = [1.5f64, 2.0, 2.5];
        let y = [1u32, 2, 3];
        let columns = vec![
            to_f64_column_bytes(&x),
            y.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ];

        let x_gt_2 = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::U32(2))),
        };
        let results: Vec<bool> = (0..x.len())
            .map(|row| Executor::evaluate_predicate(&x_gt_2, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, true]);

        let y_lt_2_5 = Expr::Compare {
            left: Box::new(Expr::Column(1)),
            op: CompareOp::Lt,
            right: Box::new(Expr::Const(ConstValue::F64(2.5))),
        };
        let results: Vec<bool> = (0..y.len())
            .map(|row| Executor::evaluate_predicate(&y_lt_2_5, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, false]);
    }

    #[test]
    fn test_register_and_get_relation() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Register a relation
        executor.register_relation(RelId(1), "test_rel");

        // Verify mapping
        assert_eq!(executor.get_rel_name(RelId(1)), Some("test_rel"));
        assert_eq!(executor.get_rel_name(RelId(2)), None);
    }

    // ============== Scan Node Tests ==============

    #[test]
    fn test_execute_scan_not_found() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        executor.register_relation(RelId(1), "missing_rel");

        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_err());
    }

    #[test]
    fn test_execute_scan_success() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("test_rel", buffer);
        executor.register_relation(RelId(1), "test_rel");

        // Execute scan
        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    // ============== Filter Node Tests ==============

    #[test]
    fn test_execute_filter_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let predicate = Expr::Const(ConstValue::Bool(true));
        let result = executor.execute_filter(&empty, &predicate);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_filter_all_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(true));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);
    }

    #[test]
    fn test_execute_filter_none_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(false));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_filter_comparison() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key > 3
        let predicate = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::U32(3))),
        };

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 2);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![4, 5]);
    }

    #[test]
    fn test_execute_filter_and() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key >= 2 AND key <= 4
        let predicate = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Ge,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Le,
                right: Box::new(Expr::Const(ConstValue::U32(4))),
            },
        ]);

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![2, 3, 4]);
    }

    // ============== Project Node Tests ==============

    #[test]
    fn test_execute_project_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_project(&empty, &[ProjectExpr::Column(0)]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
        assert_eq!(result.arity(), 1);
    }

    #[test]
    fn test_execute_project_reorder() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create a 2-column buffer
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);

        let a_data: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_data: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut col_a = executor
            .provider
            .memory()
            .alloc::<u8>(a_data.len())
            .unwrap();
        let mut col_b = executor
            .provider
            .memory()
            .alloc::<u8>(b_data.len())
            .unwrap();

        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&a_data, &mut col_a)
            .unwrap();
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&b_data, &mut col_b)
            .unwrap();

        let d_num_rows = device_row_count(&executor, 3);
        let buffer =
            CudaBuffer::from_columns(vec![col_a.into(), col_b.into()], 3, d_num_rows, schema);

        // Project: [b, a] (reverse order)
        let result =
            executor.execute_project(&buffer, &[ProjectExpr::Column(1), ProjectExpr::Column(0)]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
        assert_eq!(result.arity(), 2);

        // First column should be b's values
        let col0 = read_buffer_u32(&executor, &result, 0);
        assert_eq!(col0, vec![10, 20, 30]);

        // Second column should be a's values
        let col1 = read_buffer_u32(&executor, &result, 1);
        assert_eq!(col1, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_computed_projection_wiring() {
        // Test that ProjectExpr::Computed is handled correctly
        // Even if arithmetic stubs return errors, verify the flow is correct
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create a 2-column buffer
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);

        let a_data: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col_a = executor
            .provider
            .memory()
            .alloc::<u8>(a_data.len())
            .unwrap();
        let mut col_b = executor
            .provider
            .memory()
            .alloc::<u8>(b_data.len())
            .unwrap();

        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&a_data, &mut col_a)
            .unwrap();
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&b_data, &mut col_b)
            .unwrap();

        let d_num_rows = device_row_count(&executor, 3);
        let buffer =
            CudaBuffer::from_columns(vec![col_a.into(), col_b.into()], 3, d_num_rows, schema);

        // Project with computed expression: a + b
        let add_expr = Expr::Add(Box::new(Expr::Column(0)), Box::new(Expr::Column(1)));
        let projections = vec![
            ProjectExpr::Column(0),                           // Pass through column a
            ProjectExpr::Computed(add_expr, ScalarType::U32), // Compute a + b
        ];

        let result = executor.execute_project(&buffer, &projections);

        // The wiring should be correct - result depends on whether CUDA arithmetic kernels are available
        // If available: result has 2 columns with computed values
        // If not available: may return error from provider stubs
        match result {
            Ok(res) => {
                // Wiring worked and arithmetic kernels are available
                assert_eq!(buffer_row_count(&executor, &res), 3);
                assert_eq!(res.arity(), 2);

                // First column should be a's values (pass-through)
                let col0 = read_buffer_u32(&executor, &res, 0);
                assert_eq!(col0, vec![10, 20, 30]);

                // Second column should be a + b = [11, 22, 33]
                let col1 = read_buffer_u32(&executor, &res, 1);
                assert_eq!(col1, vec![11, 22, 33]);
            }
            Err(e) => {
                // Arithmetic kernels not available - that's OK for this test
                // The important thing is that the wiring reached the provider
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("not implemented")
                        || err_msg.contains("not yet implemented")
                        || err_msg.contains("not supported")
                        || err_msg.contains("stub")
                        || err_msg.contains("Unsupported")
                        || err_msg.contains("arithmetic kernels"),
                    "Unexpected error: {}. Expected arithmetic kernel stub error.",
                    err_msg
                );
            }
        }
    }

    // ============== Union Node Tests ==============

    #[test]
    fn test_execute_union_empty_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let result = executor.execute_union(&[]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_union_single_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");

        let result = executor.execute_union(&[buffer]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_union_multiple_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer1 = create_test_buffer(&executor, &[1, 2], "key");
        let buffer2 = create_test_buffer(&executor, &[3, 4], "key");
        let buffer3 = create_test_buffer(&executor, &[5], "key");

        let result = executor.execute_union(&[buffer1, buffer2, buffer3]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);
    }

    // ============== Distinct Node Tests ==============

    #[test]
    fn test_execute_distinct_empty() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_distinct(&empty, &[0]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    // ============== Diff Node Tests ==============

    #[test]
    fn test_execute_diff() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let right = create_test_buffer(&executor, &[2, 4], "key");

        let result = executor.execute_diff(&left, &right);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 3, 5]);
    }

    // ============== Fixpoint Tests ==============

    #[test]
    fn test_execute_fixpoint_base_only() {
        // Test fixpoint with a base case that reaches fixpoint immediately
        // (recursive step produces nothing new)
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create base relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Create an empty recursive relation (simulating a recursive step that produces nothing)
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Base: scan base_rel
        // Recursive: scan empty_rel (produces nothing new)
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return base case since recursive produces nothing
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_fixpoint_empty_base() {
        // Test fixpoint with empty base case
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create empty base relation
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema.clone()).unwrap();
        executor.store_mut().put("empty_base", empty_buffer);
        executor.register_relation(RelId(1), "empty_base");

        // Create recursive relation (won't be used since base is empty)
        let rec_buffer = create_test_buffer(&executor, &[4, 5, 6], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return empty since base is empty
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_fixpoint_one_iteration() {
        // Test fixpoint that converges after one iteration
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1, 2]
        let base_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Recursive produces [1, 2, 3] - after diff with R, only [3] remains
        let rec_buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        // After first iteration, R = [1, 2, 3], recursive produces [1, 2, 3] again
        // diff([1, 2, 3], [1, 2, 3]) = empty -> fixpoint reached

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be [1, 2, 3]
        assert_eq!(buffer_row_count(&executor, &result), 3);
    }

    #[test]
    fn test_execute_fixpoint_multiple_iterations() {
        // Test fixpoint that requires multiple iterations to converge
        // This simulates transitive closure behavior
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1]
        let base_buffer = create_test_buffer(&executor, &[1], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // For this test, we need a recursive rule that can expand
        // Since we can't easily simulate join-based recursion without complex setup,
        // we'll test a simpler case where recursive produces cumulative data

        // Recursive relation will produce [1, 2] in first iteration,
        // then [1, 2, 3] in second iteration, etc.
        // This requires a more complex setup, so let's test the basic convergence

        // Simplified test: recursive produces union of base with [2]
        // First iteration: R=[1], rec produces [1, 2] -> delta_new = [2]
        // Second iteration: R=[1, 2], rec produces [1, 2] -> delta_new = empty
        let rec_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be union of [1] and [2] = [1, 2]
        assert_eq!(buffer_row_count(&executor, &result), 2);
    }

    #[test]
    fn test_execute_fixpoint_via_node() {
        // Test fixpoint through execute_node to ensure the match arm works
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a base buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Empty recursive means immediate fixpoint
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
    }

    #[test]
    fn test_fixpoint_cleanup() {
        // Test that fixpoint properly cleans up delta and full relations
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Register names for delta and full relations to check cleanup
        executor.register_relation(RelId(2), "__delta_test");
        executor.register_relation(RelId(3), "__full_test");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // After fixpoint, the delta and full relations should be cleaned up
        assert!(!executor.store().contains("__delta_test"));
        assert!(!executor.store().contains("__full_test"));
    }

    // ============== Execute Plan Tests ==============

    #[test]
    fn test_execute_plan_empty() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let plan = ExecutionPlan::new(vec![]);

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_plan_with_stratum() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build a simple plan
        let scc = Scc {
            id: 0,
            predicates: vec!["output".to_string()],
            is_recursive: false,
        };

        let rule = CompiledRule {
            head: "output".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc],
            strata: vec![stratum],
            rules_by_scc: vec![vec![rule]],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        };

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());

        // Verify output relation was created
        assert!(executor.store().contains("output"));
        let output = executor.store().get("output").unwrap();
        assert_eq!(buffer_row_count(&executor, output), 5);
    }

    #[test]
    fn test_apply_deltas_and_recompute_updates_dependents() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let input = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", input);
        executor.register_relation(RelId(1), "input");

        // SCC0: identity rule for input (mirrors how compiled facts appear as scan rules).
        // SCC1: output depends on input.
        let scc0 = Scc {
            id: 0,
            predicates: vec!["input".to_string()],
            is_recursive: false,
        };
        let scc1 = Scc {
            id: 1,
            predicates: vec!["output".to_string()],
            is_recursive: false,
        };

        let input_rule = CompiledRule {
            head: "input".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };

        let output_rule = CompiledRule {
            head: "output".to_string(),
            body: RirNode::Filter {
                input: Box::new(RirNode::Scan { rel: RelId(1) }),
                predicate: Expr::Compare {
                    left: Box::new(Expr::Column(0)),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Const(ConstValue::U32(2))),
                },
            },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0, 1],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc0, scc1],
            strata: vec![stratum],
            rules_by_scc: vec![vec![input_rule], vec![output_rule]],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        };

        executor.execute_plan(&plan).expect("initial execute_plan");
        let initial_out = executor.store().get("output").expect("output missing");
        let initial_vals = read_buffer_u32(&executor, initial_out, 0);
        assert_eq!(initial_vals, vec![3, 4, 5]);

        let delete_buf = create_test_buffer(&executor, &[5], "key");
        let insert_buf = create_test_buffer(&executor, &[10], "key");

        let mut deltas = HashMap::new();
        deltas.insert(
            "input".to_string(),
            RelationDelta::new(Some(insert_buf), Some(delete_buf)),
        );

        executor
            .apply_deltas_and_recompute(&plan, &deltas)
            .expect("apply_deltas_and_recompute");

        let out = executor
            .store()
            .get("output")
            .expect("output missing after recompute");
        let vals = read_buffer_u32(&executor, out, 0);
        assert_eq!(vals, vec![3, 4, 10]);
    }

    #[test]
    fn test_apply_deltas_and_recompute_insert_only_recomputes_anti_join() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let lhs = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("lhs", lhs);
        executor.register_relation(RelId(1), "lhs");

        let blocked = create_test_buffer(&executor, &[], "key");
        executor.store_mut().put("blocked", blocked);
        executor.register_relation(RelId(2), "blocked");

        // SCC0: lhs identity rule
        // SCC1: blocked identity rule
        // SCC2: out = lhs \ blocked (anti-join)
        let scc0 = Scc {
            id: 0,
            predicates: vec!["lhs".to_string()],
            is_recursive: false,
        };
        let scc1 = Scc {
            id: 1,
            predicates: vec!["blocked".to_string()],
            is_recursive: false,
        };
        let scc2 = Scc {
            id: 2,
            predicates: vec!["out".to_string()],
            is_recursive: false,
        };

        let lhs_rule = CompiledRule {
            head: "lhs".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };
        let blocked_rule = CompiledRule {
            head: "blocked".to_string(),
            body: RirNode::Scan { rel: RelId(2) },
            meta: RirMeta::default(),
        };
        let out_rule = CompiledRule {
            head: "out".to_string(),
            body: RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Anti,
            },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0, 1, 2],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc0, scc1, scc2],
            strata: vec![stratum],
            rules_by_scc: vec![vec![lhs_rule], vec![blocked_rule], vec![out_rule]],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        };

        executor.execute_plan(&plan).expect("initial execute_plan");
        let initial = executor.store().get("out").expect("out missing");
        let initial_vals = read_buffer_u32(&executor, initial, 0);
        assert_eq!(initial_vals, vec![1, 2, 3, 4, 5]);

        // Insert into the "blocked" relation: output should shrink.
        let insert_buf = create_test_buffer(&executor, &[2, 4], "key");
        let mut deltas = HashMap::new();
        deltas.insert(
            "blocked".to_string(),
            RelationDelta::new(Some(insert_buf), None),
        );

        executor
            .apply_deltas_and_recompute(&plan, &deltas)
            .expect("apply_deltas_and_recompute");

        let out = executor
            .store()
            .get("out")
            .expect("out missing after update");
        let vals = read_buffer_u32(&executor, out, 0);
        assert_eq!(vals, vec![1, 3, 5]);
    }

    // ============== RIR Node Composition Tests ==============

    #[test]
    fn test_execute_filter_project_chain() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build: Project(Filter(Scan))
        let scan = RirNode::Scan { rel: RelId(1) };
        let filter = RirNode::Filter {
            input: Box::new(scan),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
        };
        let project = RirNode::Project {
            input: Box::new(filter),
            columns: vec![ProjectExpr::Column(0)],
        };

        let result = executor.execute_node(&project);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![3, 4, 5]);
    }

    // ============== v0.8.6 Common Subexpression Elimination Tests ==============

    fn duplicate_join_union_plan() -> RirNode {
        let join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        RirNode::Union {
            inputs: vec![join.clone(), join],
        }
    }

    fn seed_cse_join_fixture(executor: &mut Executor, right: &[u32]) {
        executor.register_relation(RelId(1), "left");
        executor.register_relation(RelId(2), "right");
        let left = create_test_buffer(executor, &[1, 2, 3, 4], "key");
        let right = create_test_buffer(executor, right, "key");
        executor.put_relation("left", left);
        executor.put_relation("right", right);
    }

    #[test]
    fn test_common_subexpression_cache_reuses_duplicate_inner_join_when_enabled() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_cse_join_fixture(&mut executor, &[2, 3, 5]);

        let result = executor
            .execute_node(&duplicate_join_union_plan())
            .expect("duplicate join union executes");

        assert_eq!(buffer_row_count(&executor, &result), 2);
        let stats = executor.common_subexpression_stats();
        assert_eq!(stats.hits, 1);
        assert!(stats.misses >= 1);
        assert_eq!(stats.unsafe_rejections, 0);
    }

    #[test]
    fn test_common_subexpression_off_on_preserves_output_and_records_reuse_only_when_enabled() {
        let mut disabled = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(false)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        let mut enabled = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_cse_join_fixture(&mut disabled, &[2, 3, 5]);
        seed_cse_join_fixture(&mut enabled, &[2, 3, 5]);
        let plan = duplicate_join_union_plan();

        disabled.provider.reset_d2h_transfer_count();
        enabled.provider.reset_d2h_transfer_count();
        let disabled_result = disabled.execute_node(&plan).expect("disabled CSE output");
        let enabled_result = enabled.execute_node(&plan).expect("enabled CSE output");
        let disabled_d2h = disabled.provider.d2h_transfer_count();
        let enabled_d2h = enabled.provider.d2h_transfer_count();

        assert_eq!(
            read_buffer_u32(&disabled, &disabled_result, 0),
            read_buffer_u32(&enabled, &enabled_result, 0)
        );
        assert_eq!(enabled_d2h, disabled_d2h);
        assert_eq!(disabled.common_subexpression_stats().hits, 0);
        assert_eq!(disabled.common_subexpression_stats().misses, 0);
        assert_eq!(enabled.common_subexpression_stats().hits, 1);
    }

    #[test]
    fn test_common_subexpression_cache_invalidates_on_relation_generation_change() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_cse_join_fixture(&mut executor, &[2, 3, 5]);
        let plan = duplicate_join_union_plan();

        executor.execute_node(&plan).expect("first execution");
        assert_eq!(executor.common_subexpression_stats().hits, 1);

        let changed_right = create_test_buffer(&executor, &[4], "key");
        executor.put_relation("right", changed_right);
        let result = executor.execute_node(&plan).expect("second execution");

        assert_eq!(buffer_row_count(&executor, &result), 1);
        let stats = executor.common_subexpression_stats();
        assert_eq!(stats.hits, 2);
        assert!(stats.misses >= 2);
    }

    #[test]
    fn test_common_subexpression_cache_rejects_unsafe_difference_boundary() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_cse_join_fixture(&mut executor, &[2, 3, 5]);
        let diff = RirNode::Diff {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
        };

        executor
            .execute_node(&RirNode::Union {
                inputs: vec![diff.clone(), diff],
            })
            .expect("unsafe duplicate diff still executes without CSE sharing");

        let stats = executor.common_subexpression_stats();
        assert_eq!(stats.hits, 0);
        assert!(stats.unsafe_rejections >= 1);
        assert!(stats
            .rejection_reasons
            .iter()
            .any(|reason| reason == "negation_or_difference_boundary"));
    }

    #[test]
    fn test_common_subexpression_key_rejects_aggregate_and_tensor_boundaries() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_common_subexpression_elimination(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        let aggregate = RirNode::GroupBy {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            key_cols: vec![0],
            aggs: vec![(0, xlog_core::AggOp::Count)],
        };
        let tensor = RirNode::TensorMaskedJoin {
            mask_name: "W".to_string(),
            schema_size: 1,
            left_keys: vec![0],
            right_keys: vec![0],
            rel_index: vec![(RelId(1), "left".to_string())],
            head_rel_name: "head".to_string(),
            head_rel_id: RelId(3),
            max_active_rules: 1,
            head_projection: vec![0],
        };
        let chain = RirNode::ChainJoin {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_key: 0,
            right_key: 0,
            output_columns: vec![ProjectExpr::Column(0)],
            fallback: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
        };

        assert!(executor.common_subexpression_key(&aggregate).is_none());
        assert!(executor.common_subexpression_key(&tensor).is_none());
        assert!(executor.common_subexpression_key(&chain).is_none());

        let reasons = &executor.common_subexpression_stats().rejection_reasons;
        assert!(reasons.iter().any(|reason| reason == "aggregate_boundary"));
        assert!(reasons
            .iter()
            .any(|reason| reason == "provenance_or_tensor_boundary"));
        assert!(reasons
            .iter()
            .any(|reason| reason == "specialized_dispatch_boundary"));
    }

    // ============== v0.8.6 Adaptive Runtime Re-Optimization Tests ==============

    fn adaptive_scc() -> Scc {
        Scc {
            id: 0,
            predicates: vec!["out".to_string()],
            is_recursive: false,
        }
    }

    fn adaptive_stratum() -> Stratum {
        Stratum {
            id: 0,
            sccs: vec![0],
        }
    }

    fn adaptive_rule(body: RirNode) -> CompiledRule {
        CompiledRule {
            head: "out".to_string(),
            body,
            meta: RirMeta::default(),
        }
    }

    fn adaptive_plan(body: RirNode) -> ExecutionPlan {
        ExecutionPlan {
            sccs: vec![adaptive_scc()],
            strata: vec![adaptive_stratum()],
            rules_by_scc: vec![vec![adaptive_rule(body)]],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        }
    }

    fn adaptive_baseline_join_plan() -> ExecutionPlan {
        adaptive_plan(RirNode::Project {
            input: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            columns: vec![ProjectExpr::Column(0)],
        })
    }

    fn adaptive_scan_candidate_plan(rel: RelId) -> ExecutionPlan {
        adaptive_plan(RirNode::Scan { rel })
    }

    fn seed_adaptive_fixture(executor: &mut Executor, right: &[u32]) {
        executor.register_relation(RelId(1), "left");
        executor.register_relation(RelId(2), "right");
        let left = create_test_buffer(executor, &[1, 2, 3, 4, 5, 6, 7, 8], "key");
        let right = create_test_buffer(executor, right, "key");
        executor.put_relation("left", left);
        executor.put_relation("right", right);
    }

    #[test]
    fn test_adaptive_reoptimization_disabled_uses_baseline_and_records_decision() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_adaptive_reoptimization(Some(false)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_adaptive_fixture(&mut executor, &[1, 2, 3, 4, 5, 6, 7, 8]);

        let baseline = adaptive_baseline_join_plan();
        let candidate = adaptive_scan_candidate_plan(RelId(2));
        let result = executor
            .execute_plan_with_adaptive_candidate(&baseline, &candidate)
            .expect("disabled adaptation executes baseline");

        assert_eq!(
            read_buffer_u32(&executor, &result, 0),
            (1..=8).collect::<Vec<_>>()
        );
        let stats = executor.adaptive_reoptimization_stats();
        assert_eq!(stats.disabled, 1);
        assert_eq!(stats.adopted, 0);
        assert_eq!(stats.rolled_back, 0);
        assert_eq!(
            stats.last_decision.as_ref().map(|decision| decision.action),
            Some(AdaptiveReoptimizationAction::Disabled)
        );
    }

    #[test]
    fn test_adaptive_reoptimization_adopts_equivalent_candidate_and_records_telemetry() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_adaptive_reoptimization(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_adaptive_fixture(&mut executor, &[1, 2, 3, 4, 5, 6, 7, 8]);
        executor.provider.reset_host_transfer_stats();

        let baseline = adaptive_baseline_join_plan();
        let candidate = adaptive_scan_candidate_plan(RelId(1));
        let result = executor
            .execute_plan_with_adaptive_candidate(&baseline, &candidate)
            .expect("equivalent candidate is adopted");

        assert_eq!(
            read_buffer_u32(&executor, &result, 0),
            (1..=8).collect::<Vec<_>>()
        );
        let stats = executor.adaptive_reoptimization_stats();
        assert_eq!(stats.adopted, 1);
        assert_eq!(stats.rolled_back, 0);
        assert_eq!(stats.last_observations.len(), 1);
        assert!(stats.last_observations[0].cardinality_delta_abs > 0);
        assert!(stats.last_observations[0].selectivity_delta_abs > 0.0);
        assert_eq!(stats.data_plane_dtoh_calls, 0);
    }

    #[test]
    fn test_adaptive_reoptimization_rolls_back_bad_candidate_with_typed_diagnostic() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_adaptive_reoptimization(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_adaptive_fixture(&mut executor, &[1, 2, 3, 4, 5, 6, 7, 8]);

        let baseline = adaptive_baseline_join_plan();
        let bad_candidate = adaptive_scan_candidate_plan(RelId(2));
        executor.put_relation("right", create_test_buffer(&executor, &[99], "key"));
        let result = executor
            .execute_plan_with_adaptive_candidate(&baseline, &bad_candidate)
            .expect("bad candidate rolls back to baseline output");

        assert_eq!(read_buffer_u32(&executor, &result, 0), Vec::<u32>::new());
        let out = executor.store().get("out").expect("rollback restored out");
        assert_eq!(read_buffer_u32(&executor, out, 0), Vec::<u32>::new());
        let stats = executor.adaptive_reoptimization_stats();
        assert_eq!(stats.adopted, 0);
        assert_eq!(stats.rolled_back, 1);
        assert!(stats.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == AdaptiveReoptimizationDiagnosticKind::CandidateOutputMismatch
        }));
    }

    #[test]
    fn test_adaptive_reoptimization_decisions_are_deterministic_under_replay() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_adaptive_reoptimization(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_adaptive_fixture(&mut executor, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let baseline = adaptive_baseline_join_plan();
        executor
            .execute_plan(&baseline)
            .expect("baseline execution records telemetry");
        let observations = executor
            .adaptive_reoptimization_stats()
            .last_observations
            .clone();

        let first = executor.replay_adaptive_reoptimization_decision(&observations);
        for _ in 0..100 {
            assert_eq!(
                executor.replay_adaptive_reoptimization_decision(&observations),
                first
            );
        }
    }

    // ============== v0.8.6 Persistent Hash Index Manager Tests ==============

    fn persistent_index_join_plan() -> ExecutionPlan {
        adaptive_baseline_join_plan()
    }

    fn persistent_index_heavy_join_plan(repetitions: usize) -> ExecutionPlan {
        let mut inputs = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            inputs.push(RirNode::Project {
                input: Box::new(RirNode::Join {
                    left: Box::new(RirNode::Scan { rel: RelId(1) }),
                    right: Box::new(RirNode::Scan { rel: RelId(2) }),
                    left_keys: vec![0],
                    right_keys: vec![0],
                    join_type: JoinType::Semi,
                }),
                columns: vec![ProjectExpr::Column(0)],
            });
        }
        adaptive_plan(RirNode::Union { inputs })
    }

    fn seed_persistent_index_fixture(executor: &mut Executor, rows: u32) {
        executor.register_relation(RelId(1), "left");
        executor.register_relation(RelId(2), "right");
        let values: Vec<u32> = (0..rows).collect();
        let left = create_test_buffer(executor, &values, "key");
        let right = create_test_buffer(executor, &values, "key");
        executor.put_relation("left", left);
        executor.put_relation("right", right);
    }

    fn seed_persistent_index_performance_fixture(
        executor: &mut Executor,
        left_rows: u32,
        right_rows: u32,
    ) {
        executor.register_relation(RelId(1), "left");
        executor.register_relation(RelId(2), "right");
        let left_values: Vec<u32> = (0..left_rows).collect();
        let right_values: Vec<u32> = (0..right_rows).collect();
        let left = create_test_buffer(executor, &left_values, "key");
        let right = create_test_buffer(executor, &right_values, "key");
        executor.put_relation("left", left);
        executor.put_relation("right", right);
    }

    fn warm_persistent_index(executor: &mut Executor, plan: &ExecutionPlan, times: usize) {
        for _ in 0..times {
            executor.execute_plan(plan).expect("persistent index plan");
        }
    }

    fn median_duration(samples: &mut [Duration]) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn measure_persistent_index_fixture(
        mut executor: Executor,
        plan: &ExecutionPlan,
        warmup: usize,
        iterations: usize,
    ) -> (
        Duration,
        u64,
        JoinIndexCacheStats,
        xlog_cuda::provider::HostTransferStats,
    ) {
        let mut output_rows = None;
        warm_persistent_index(&mut executor, plan, warmup);
        executor.provider.reset_host_transfer_stats();

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let output = executor.execute_plan(plan).expect("persistent index plan");
            executor
                .provider
                .device()
                .synchronize()
                .expect("sync device");
            samples.push(start.elapsed());
            output_rows = Some(if let Some(buffer) = executor.store().get("out") {
                executor
                    .buffer_row_count(buffer)
                    .expect("read output row count")
                    .into()
            } else {
                output.num_rows()
            });
        }

        (
            median_duration(&mut samples),
            output_rows.expect("at least one measured execution"),
            executor.join_index_cache_stats(),
            executor.provider.host_transfer_stats(),
        )
    }

    #[test]
    fn test_persistent_hash_index_reuses_across_repeated_session_evaluations() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_persistent_hash_indexes(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_fixture(&mut executor, 2_500);
        let plan = persistent_index_join_plan();
        executor.provider.reset_host_transfer_stats();

        warm_persistent_index(&mut executor, &plan, 5);

        let stats = executor.join_index_cache_stats();
        let transfers = executor.provider.host_transfer_stats();
        assert_eq!(stats.builds, 1);
        assert!(stats.hits >= 1);
        assert_eq!(stats.stale_rejections, 0);
        assert_eq!(stats.entries, 1);
        assert!(stats.total_bytes > 0);
        assert_eq!(transfers.dtoh_calls, 0);
        assert_eq!(transfers.htod_calls, 0);
    }

    #[test]
    fn test_persistent_hash_index_invalidates_on_relation_generation_change() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default().with_persistent_hash_indexes(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_fixture(&mut executor, 2_500);
        let plan = persistent_index_join_plan();
        warm_persistent_index(&mut executor, &plan, 5);
        assert_eq!(executor.join_index_cache_stats().entries, 1);

        let changed_values: Vec<u32> = (10_000..12_500).collect();
        let changed_right = create_test_buffer(&executor, &changed_values, "key");
        executor.put_relation("right", changed_right);

        let stats = executor.join_index_cache_stats();
        assert_eq!(stats.entries, 0);
        assert!(stats.invalidations >= 1);
    }

    #[test]
    fn test_persistent_hash_index_background_build_records_requests() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default()
                .with_persistent_hash_indexes(Some(true))
                .with_persistent_hash_index_background_build(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_fixture(&mut executor, 2_500);
        let plan = persistent_index_join_plan();

        warm_persistent_index(&mut executor, &plan, 5);

        let stats = executor.join_index_cache_stats();
        assert_eq!(stats.background_build_requests, 1);
        assert_eq!(stats.background_builds_completed, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn test_persistent_hash_index_background_build_defers_current_join_reuse() {
        let mut executor = match create_test_executor_with_config(
            RuntimeConfig::default()
                .with_persistent_hash_indexes(Some(true))
                .with_persistent_hash_index_background_build(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_fixture(&mut executor, 2_500);
        let plan = persistent_index_join_plan();

        let mut before_build = executor.join_index_cache_stats();
        let mut after_build = None;
        for _ in 0..5 {
            executor
                .execute_plan(&plan)
                .expect("background-build warm evaluation");
            let stats = executor.join_index_cache_stats();
            if stats.background_build_requests > before_build.background_build_requests {
                after_build = Some(stats);
                break;
            }
            before_build = stats;
        }

        let after_first = after_build.expect("background build request observed");
        assert_eq!(after_first.background_build_requests, 1);
        assert_eq!(after_first.background_builds_completed, 1);
        assert_eq!(after_first.background_builds_deferred, 1);
        assert_eq!(
            after_first.hits, before_build.hits,
            "background build must not be consumed by the same evaluation that requested it"
        );
        assert_eq!(after_first.entries, 1);

        executor
            .execute_plan(&plan)
            .expect("second evaluation reuses completed background index");
        let after_second = executor.join_index_cache_stats();
        assert_eq!(after_second.background_build_requests, 1);
        assert_eq!(after_second.background_builds_deferred, 1);
        assert!(after_second.hits >= 1);
    }

    #[test]
    fn test_persistent_hash_index_performance_fixture_meets_speedup_target() {
        const LEFT_ROWS: u32 = 8;
        const RIGHT_ROWS: u32 = 8_000_000;
        const JOIN_REPETITIONS: usize = 1;
        const WARMUP: usize = 12;
        const ITERATIONS: usize = 9;

        let mut cached = match create_test_executor_with_config(
            RuntimeConfig::default().with_persistent_hash_indexes(Some(true)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_performance_fixture(&mut cached, LEFT_ROWS, RIGHT_ROWS);

        let mut uncached = match create_test_executor_with_config(
            RuntimeConfig::default().with_persistent_hash_indexes(Some(false)),
        ) {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        seed_persistent_index_performance_fixture(&mut uncached, LEFT_ROWS, RIGHT_ROWS);

        let plan = persistent_index_heavy_join_plan(JOIN_REPETITIONS);
        let (cached_median, cached_rows, cached_stats, cached_transfers) =
            measure_persistent_index_fixture(cached, &plan, WARMUP, ITERATIONS);
        let (uncached_median, uncached_rows, uncached_stats, uncached_transfers) =
            measure_persistent_index_fixture(uncached, &plan, WARMUP, ITERATIONS);

        let speedup_ratio = uncached_median.as_secs_f64() / cached_median.as_secs_f64();
        eprintln!(
            "persistent_hash_index_perf left_rows={} right_rows={} join_repetitions={} warmup={} iterations={} \
             cached_median_sec={:.9} uncached_median_sec={:.9} speedup_ratio={:.3} \
             cached_output_rows={} uncached_output_rows={} cached_builds={} cached_hits={} \
             uncached_builds={} cached_dtoh_calls={} cached_htod_calls={}",
            LEFT_ROWS,
            RIGHT_ROWS,
            JOIN_REPETITIONS,
            WARMUP,
            ITERATIONS,
            cached_median.as_secs_f64(),
            uncached_median.as_secs_f64(),
            speedup_ratio,
            cached_rows,
            uncached_rows,
            cached_stats.builds,
            cached_stats.hits,
            uncached_stats.builds,
            cached_transfers.dtoh_calls,
            cached_transfers.htod_calls
        );

        assert_eq!(cached_rows, uncached_rows);
        assert_eq!(cached_rows, LEFT_ROWS as u64);
        assert_eq!(cached_stats.builds, 1);
        assert!(cached_stats.hits >= ITERATIONS as u64);
        assert_eq!(uncached_stats.builds, 0);
        assert_eq!(cached_transfers.dtoh_calls, 0);
        assert_eq!(cached_transfers.htod_calls, 0);
        assert_eq!(uncached_transfers.dtoh_calls, 0);
        assert_eq!(uncached_transfers.htod_calls, 0);
        assert!(
            speedup_ratio >= 1.5,
            "persistent index speedup {:.3} below 1.5 target",
            speedup_ratio
        );
    }

    // ============== MC Relation Reset Tests ==============

    #[test]
    fn test_reset_for_mc_relations_preserves_static_and_clears_dynamic() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping: no CUDA device");
                return;
            }
        };

        executor.register_relation(RelId(1), "base_rel");
        executor.register_relation(RelId(2), "dyn_rel");

        let schema = Schema::new(vec![("x".to_string(), ScalarType::U32)]);
        let base = create_test_buffer(&executor, &[1u32], "x");
        let dyn_buf = create_test_buffer(&executor, &[9u32], "x");
        executor.put_relation("base_rel", base);
        executor.put_relation("dyn_rel", dyn_buf);

        executor
            .reset_for_mc_relations(&["base_rel"], &[("dyn_rel", schema.clone())])
            .unwrap();

        assert_eq!(
            buffer_row_count(&executor, executor.store().get("base_rel").unwrap()),
            1
        );
        assert_eq!(
            buffer_row_count(&executor, executor.store().get("dyn_rel").unwrap()),
            0
        );
    }
}
