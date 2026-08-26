//! GPU-accelerated evaluation of compiled Datalog programs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use xlog_core::{resolve_bool, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaColumn, CudaKernelProvider};
use xlog_ir::{EpistemicExecutablePlan, ExecutionPlan};
use xlog_logic::ast::{PredColumn, PredDecl, TypeRef};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_split_execution,
    epistemic_extensional_multi_arity_predicates, prepare_epistemic_program,
    reduce_epistemic_program_to_ordinary,
    reduce_epistemic_program_to_ordinary_for_stratified_schema,
    try_plan_stratified_epistemic_program, try_prepare_g91_compatibility_reduction,
    try_reduce_case_a_recursive_epistemic_program, try_reduce_prepared_recursive_epistemic_program,
    EpistemicSplitExecutablePlan, G91CompatibilityReduction,
};
use xlog_logic::ground_term_encoding::append_ground_term_bytes;
use xlog_logic::{
    format_constraint_body, Atom, BodyLiteral, Compiler, Constraint, EpistemicLiteral, EpistemicOp,
    Program, Query, Rule, Term,
};
use xlog_runtime::executor::JoinIndexCacheStats;
use xlog_runtime::resident_graph::{
    ResidentGraphCertifiedPlan, ResidentGraphCoreTransferStats, ResidentGraphDeclineReason,
    ResidentGraphDeferredProfile, ResidentGraphExecutionError, ResidentGraphExecutionStats,
    ResidentGraphFinalObservationStats, ResidentGraphPrepareOptions, ResidentGraphSchemaCatalog,
    ResidentGraphSelectionKind,
};
use xlog_runtime::{
    DeltaRecomputeStats, EpistemicGpuExecutionResult, EpistemicGpuWorkspaceCapacities,
    ExecutionStats, Executor, OpStats, RelationDelta, RelationStore, StratumStats,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentSelectionMode {
    Auto,
    Disabled,
    Require,
}

impl ResidentSelectionMode {
    fn from_env() -> Result<Self> {
        let disabled = resolve_bool(None, "XLOG_DISABLE_RESIDENT_RECURSION", false)?;
        let required = resolve_bool(None, "XLOG_REQUIRE_RESIDENT_RECURSION", false)?;
        if disabled && required {
            return Err(XlogError::Execution(
                "resident execution environment flags are mutually exclusive".to_string(),
            ));
        }
        Ok(if disabled {
            Self::Disabled
        } else if required {
            Self::Require
        } else {
            Self::Auto
        })
    }

    fn enabled(self) -> bool {
        self != Self::Disabled
    }
}

struct ResidentCompletedProfile {
    telemetry: ResidentGraphExecutionStats,
    iterations: u32,
}

const RESIDENT_LATENCY_DIAGNOSTICS_ENV: &str = "XLOG_RESIDENT_LATENCY_DIAGNOSTICS";
static RESIDENT_LATENCY_SAMPLE: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct ResidentLatencyDiagnostic {
    sample: u64,
    certificate_input_ns: u64,
    certificate_cache_was_warm: bool,
    certificate_initialized_here: bool,
    certificate_initialization_ns: u64,
    certificate_cache_access_ns: u64,
    input_setup_ns: u64,
    prepare_capture_allocation_ns: u64,
    launch_submission_ns: u64,
    sync_wall_ns: u64,
    device_event_ns: u64,
    receipt_d2h_ns: u64,
    receipt_decode_schema_staging_ns: u64,
    owner_teardown_residual_ns: u64,
    commit_ns: u64,
    result_stats_construction_ns: u64,
    executor_store_teardown_ns: u64,
    staged_outputs: u64,
    relation_registrations: usize,
    remaining_store_relations_before_drop: usize,
    runtime_bytes: [usize; 8],
    manager_bytes: [u64; 8],
}

#[cfg(test)]
mod resident_latency_diagnostic_tests {
    use super::{
        finalized_resident_latency_diagnostic_lines, resident_latency_diagnostic_line,
        ResidentLatencyDiagnostic,
    };

    #[test]
    fn certificate_latency_distinguishes_cold_initialization_from_warm_access() {
        let mut cold = ResidentLatencyDiagnostic::new();
        cold.sample = 3;
        cold.certificate_cache_was_warm = false;
        cold.certificate_initialized_here = true;
        cold.certificate_initialization_ns = 41;
        let cold_line =
            resident_latency_diagnostic_line(Some(&cold), 101).expect("cold diagnostic line");
        assert!(cold_line.contains("sample=3"));
        assert!(cold_line.contains("certificate_cache_was_warm=false"));
        assert!(cold_line.contains("certificate_initialized_here=true"));
        assert!(cold_line.contains("certificate_initialization_ns=41"));
        assert!(cold_line.contains("certificate_cache_access_ns=0"));

        let mut warm = ResidentLatencyDiagnostic::new();
        warm.sample = 4;
        warm.certificate_cache_was_warm = true;
        warm.certificate_cache_access_ns = 7;
        let warm_line =
            resident_latency_diagnostic_line(Some(&warm), 102).expect("warm diagnostic line");
        assert!(warm_line.contains("sample=4"));
        assert!(warm_line.contains("certificate_cache_was_warm=true"));
        assert!(warm_line.contains("certificate_initialized_here=false"));
        assert!(warm_line.contains("certificate_initialization_ns=0"));
        assert!(warm_line.contains("certificate_cache_access_ns=7"));
        assert_eq!(resident_latency_diagnostic_line(None, 0), None);
    }

    #[test]
    fn finalized_latency_diagnostics_derive_after_total_and_preserve_sample_order() {
        let mut outer = ResidentLatencyDiagnostic::new();
        outer.sample = 17;
        let prepare_derived = std::cell::Cell::new(false);
        let outer_derived = std::cell::Cell::new(false);

        let lines = finalized_resident_latency_diagnostic_lines(
            101,
            Some(&outer),
            Some(|| {
                prepare_derived.set(true);
                "resident prepare phases: sample=17 total_ns=53".to_string()
            }),
            |diagnostic, total_ns| {
                outer_derived.set(true);
                diagnostic.format_line(total_ns)
            },
        )
        .expect("enabled diagnostics finalize one fixed pair");

        assert!(prepare_derived.get());
        assert!(outer_derived.get());
        assert_eq!(lines.len(), 2);
        assert!(lines[0]
            .as_deref()
            .is_some_and(|line| line.starts_with("resident prepare phases: sample=17 ")));
        assert!(lines[1].as_deref().is_some_and(|line| {
            line.starts_with("resident latency phases: sample=17 total_ns=101 ")
        }));

        let disabled_prepare_called = std::cell::Cell::new(false);
        let disabled_outer_called = std::cell::Cell::new(false);
        let disabled = finalized_resident_latency_diagnostic_lines(
            0,
            None,
            Some(|| {
                disabled_prepare_called.set(true);
                String::new()
            }),
            |diagnostic, total_ns| {
                disabled_outer_called.set(true);
                diagnostic.format_line(total_ns)
            },
        );
        assert!(disabled.is_none());
        assert!(!disabled_prepare_called.get());
        assert!(!disabled_outer_called.get());
    }
}

impl ResidentLatencyDiagnostic {
    fn new() -> Self {
        Self {
            sample: RESIDENT_LATENCY_SAMPLE.fetch_add(1, Ordering::Relaxed),
            ..Self::default()
        }
    }

    fn format_line(&self, total_ns: u64) -> String {
        let additive_phases = [
            self.certificate_input_ns,
            self.prepare_capture_allocation_ns,
            self.launch_submission_ns,
            self.sync_wall_ns,
            self.receipt_d2h_ns,
            self.receipt_decode_schema_staging_ns,
            self.owner_teardown_residual_ns,
            self.commit_ns,
            self.result_stats_construction_ns,
            self.executor_store_teardown_ns,
        ];
        let unattributed_host_ns = resident_latency_unattributed_ns(total_ns, &additive_phases);
        let owner_runtime_bytes_released =
            self.runtime_bytes[4].saturating_sub(self.runtime_bytes[5]);
        let owner_manager_bytes_released =
            self.manager_bytes[4].saturating_sub(self.manager_bytes[5]);
        let executor_runtime_bytes_released =
            self.runtime_bytes[6].saturating_sub(self.runtime_bytes[7]);
        let executor_manager_bytes_released =
            self.manager_bytes[6].saturating_sub(self.manager_bytes[7]);
        format!(
            "resident latency phases: sample={} total_ns={} certificate_input_ns={} certificate_cache_was_warm={} certificate_initialized_here={} certificate_initialization_ns={} certificate_cache_access_ns={} input_setup_ns={} certificate_input_unattributed_ns={} prepare_capture_allocation_ns={} launch_submission_ns={} sync_wall_ns={} device_event_ns_nonadditive={} receipt_d2h_ns={} receipt_decode_schema_staging_ns={} owner_teardown_residual_ns={} commit_ns={} result_stats_construction_ns={} executor_store_teardown_ns={} unattributed_host_ns={} staged_outputs={} relation_registrations={} remaining_store_relations_before_drop={} allocation_snapshot_order=runtime_ready|after_setup|after_prepare|after_launch|after_sync|after_observe|after_commit|after_executor_drop runtime_bytes={:?} manager_bytes={:?} owner_runtime_bytes_released={} owner_manager_bytes_released={} executor_runtime_bytes_released={} executor_manager_bytes_released={} deallocation_calls=unavailable",
            self.sample,
            total_ns,
            self.certificate_input_ns,
            self.certificate_cache_was_warm,
            self.certificate_initialized_here,
            self.certificate_initialization_ns,
            self.certificate_cache_access_ns,
            self.input_setup_ns,
            self.certificate_input_ns
                .saturating_sub(self.certificate_initialization_ns)
                .saturating_sub(self.certificate_cache_access_ns)
                .saturating_sub(self.input_setup_ns),
            self.prepare_capture_allocation_ns,
            self.launch_submission_ns,
            self.sync_wall_ns,
            self.device_event_ns,
            self.receipt_d2h_ns,
            self.receipt_decode_schema_staging_ns,
            self.owner_teardown_residual_ns,
            self.commit_ns,
            self.result_stats_construction_ns,
            self.executor_store_teardown_ns,
            unattributed_host_ns,
            self.staged_outputs,
            self.relation_registrations,
            self.remaining_store_relations_before_drop,
            self.runtime_bytes,
            self.manager_bytes,
            owner_runtime_bytes_released,
            owner_manager_bytes_released,
            executor_runtime_bytes_released,
            executor_manager_bytes_released,
        )
    }
}

#[cfg(test)]
fn resident_latency_diagnostic_line(
    diagnostic: Option<&ResidentLatencyDiagnostic>,
    total_ns: u64,
) -> Option<String> {
    diagnostic.map(|diagnostic| diagnostic.format_line(total_ns))
}

fn finalized_resident_latency_diagnostic_lines<F, G>(
    total_ns: u64,
    diagnostic: Option<&ResidentLatencyDiagnostic>,
    prepare_line: Option<F>,
    format_outer: G,
) -> Option<[Option<String>; 2]>
where
    F: FnOnce() -> String,
    G: FnOnce(&ResidentLatencyDiagnostic, u64) -> String,
{
    let diagnostic = diagnostic?;
    Some([
        prepare_line.map(|prepare_line| prepare_line()),
        Some(format_outer(diagnostic, total_ns)),
    ])
}

fn resident_latency_diagnostics_enabled() -> bool {
    std::env::var(RESIDENT_LATENCY_DIAGNOSTICS_ENV).as_deref() == Ok("1")
}

fn resident_latency_elapsed_ns(started: Option<std::time::Instant>) -> u64 {
    started
        .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn resident_latency_unattributed_ns(total_ns: u64, phases: &[u64]) -> u64 {
    phases.iter().copied().fold(total_ns, u64::saturating_sub)
}

/// Result of evaluating a single query in a Datalog program.
pub struct LogicQueryResult {
    /// Display relation name. Ordinary query projections use an internal name such as
    /// `__xlog_query_0`; epistemic materializations use the source predicate name.
    pub relation_name: String,
    /// Output variable names in column order.
    pub columns: Vec<String>,
    /// Per-output-column sort labels in column order.
    pub sort_labels: Vec<String>,
    /// GPU-resident column buffer with the result tuples.
    pub buffer: CudaBuffer,
}

/// Result of evaluating an entire Datalog program.
pub struct LogicEvalResult {
    /// One result per `?-` query in the source program.
    pub queries: Vec<LogicQueryResult>,
    /// Execution statistics (populated when profiling is enabled).
    pub stats: Option<ExecutionStats>,
}

/// Runtime state retained by a persistent logic session.
pub struct LogicSessionRuntime {
    reusable_state_identity: Arc<LogicProgramIdentity>,
    executor: Executor,
    profiling: bool,
}

#[derive(Debug)]
struct LogicProgramIdentity {
    resident_certification:
        OnceLock<std::result::Result<Arc<ResidentGraphCertifiedPlan>, Arc<str>>>,
    #[cfg(test)]
    resident_certification_initializations: AtomicU64,
}

impl LogicProgramIdentity {
    fn new() -> Self {
        Self {
            resident_certification: OnceLock::new(),
            #[cfg(test)]
            resident_certification_initializations: AtomicU64::new(0),
        }
    }

    fn get_or_init_resident_certification(
        &self,
        initialize: impl FnOnce() -> Result<ResidentGraphCertifiedPlan>,
    ) -> Result<Arc<ResidentGraphCertifiedPlan>> {
        let cached = self.resident_certification.get_or_init(|| {
            #[cfg(test)]
            self.resident_certification_initializations
                .fetch_add(1, Ordering::Relaxed);
            initialize()
                .map(Arc::new)
                .map_err(|error| Arc::<str>::from(error.to_string()))
        });
        cached.as_ref().map(Arc::clone).map_err(|message| {
            XlogError::Execution(format!("resident route certification failed: {message}"))
        })
    }

    fn get_or_init_resident_certification_with_outcome(
        &self,
        initialize: impl FnOnce() -> Result<ResidentGraphCertifiedPlan>,
    ) -> Result<(Arc<ResidentGraphCertifiedPlan>, bool, bool)> {
        let cache_was_warm = self.resident_certification.get().is_some();
        let mut initialized_here = false;
        let cached = self.resident_certification.get_or_init(|| {
            initialized_here = true;
            #[cfg(test)]
            self.resident_certification_initializations
                .fetch_add(1, Ordering::Relaxed);
            initialize()
                .map(Arc::new)
                .map_err(|error| Arc::<str>::from(error.to_string()))
        });
        cached
            .as_ref()
            .map(|certified| (Arc::clone(certified), cache_was_warm, initialized_here))
            .map_err(|message| {
                XlogError::Execution(format!("resident route certification failed: {message}"))
            })
    }

    #[cfg(test)]
    fn resident_certification_initializations(&self) -> u64 {
        self.resident_certification_initializations
            .load(Ordering::Relaxed)
    }
}

/// A materialized derived store produced by one compiled logic program.
///
/// The store's program identity is intentionally opaque. It can be inspected
/// read-only, but only the originating [`LogicProgram`] (or one of its clones)
/// can accept it as reusable execution state.
pub struct LogicMaterializedStore {
    reusable_state_identity: Arc<LogicProgramIdentity>,
    store: RelationStore,
}

impl LogicMaterializedStore {
    /// Borrow the materialized relations for read-only result inspection.
    pub fn as_relation_store(&self) -> &RelationStore {
        &self.store
    }
}

impl LogicSessionRuntime {
    /// Return persistent hash-index cache telemetry for the retained executor.
    pub fn join_index_cache_stats(&self) -> JoinIndexCacheStats {
        self.executor.join_index_cache_stats()
    }

    /// Return multiway/Free-Join dispatch telemetry for the retained executor.
    pub fn wcoj_dispatch_stats(&self) -> WcojDispatchStats {
        WcojDispatchStats {
            free_join_dispatch_count: self.executor.free_join_dispatch_count(),
            factorized_delta_dispatch_count: self.executor.factorized_delta_dispatch_count(),
            wcoj_groupby_fusion_dispatch_count: self.executor.wcoj_groupby_fusion_dispatch_count(),
            wcoj_error_decline_count: self.executor.wcoj_error_decline_count(),
        }
    }
}

/// Multiway/Free-Join dispatch telemetry counters for a retained session
/// executor. Counts accumulate across evaluates within the session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WcojDispatchStats {
    /// Free Join dispatches taken through the multiway plan.
    pub free_join_dispatch_count: u64,
    /// Factorized recursive-delta dispatches taken in the semi-naive
    /// fixpoint (dense bitvector or sparse hash-set route).
    pub factorized_delta_dispatch_count: u64,
    /// Aggregate-fused group-by-root dispatches (no materialized join rows).
    pub wcoj_groupby_fusion_dispatch_count: u64,
    /// WCOJ pipeline errors that declined to the binary-join fallback.
    pub wcoj_error_decline_count: u64,
}

/// Planner-grade telemetry for a persistent-session relation delta update.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeltaPlannerTelemetry {
    /// True when the relation-delta path reused an existing session/cache.
    pub cache_reused: bool,
    /// Planner decision used for this delta update.
    pub fallback_decision: String,
    /// Number of SCCs affected by the delta dependency closure.
    pub affected_sccs: usize,
    /// Number of SCCs recomputed from scratch.
    pub recomputed_sccs: usize,
    /// Number of SCCs updated incrementally.
    pub incremental_sccs: usize,
    /// Estimated speedup of delta evaluation over full recompute when available.
    pub estimated_delta_speedup: Option<f64>,
    /// Measured speedup of delta evaluation over full recompute when both timings are available.
    pub measured_delta_speedup: Option<f64>,
    /// Human-readable planner guidance for downstream diagnostics.
    pub planner_advice: Vec<String>,
}

impl DeltaPlannerTelemetry {
    /// Build planner telemetry from a delta report and optional timing evidence.
    pub fn from_delta_report(
        report: &LogicDeltaReport,
        cache_reused: bool,
        measured_micros: Option<(u64, u64)>,
    ) -> Self {
        let fallback_decision = if report.affected_sccs == 0 {
            "no_op"
        } else if report.has_deletes || report.recomputed_sccs > 0 {
            "full_recompute_fallback"
        } else {
            "incremental"
        }
        .to_string();
        let estimated_delta_speedup = if report.affected_sccs > 0 {
            Some((report.affected_sccs.max(1) as f64) / (report.incremental_sccs.max(1) as f64))
        } else {
            None
        };
        let measured_delta_speedup = measured_micros.and_then(|(delta_us, full_us)| {
            if delta_us == 0 {
                None
            } else {
                Some(full_us as f64 / delta_us as f64)
            }
        });

        let mut planner_advice = Vec::new();
        if fallback_decision == "full_recompute_fallback" {
            planner_advice.push(
                "full recompute fallback selected; inspect deletes or affected SCC fanout"
                    .to_string(),
            );
        } else if let Some(speedup) = measured_delta_speedup {
            if speedup >= 1.0 {
                planner_advice.push(format!("delta path is faster by {speedup:.2}x"));
            } else {
                planner_advice.push(format!(
                    "full recompute may be faster; delta measured {speedup:.2}x"
                ));
            }
        } else if fallback_decision == "incremental" {
            planner_advice.push(
                "incremental delta path selected; run equivalence timing to measure speedup"
                    .to_string(),
            );
        }

        Self {
            cache_reused,
            fallback_decision,
            affected_sccs: report.affected_sccs,
            recomputed_sccs: report.recomputed_sccs,
            incremental_sccs: report.incremental_sccs,
            estimated_delta_speedup,
            measured_delta_speedup,
            planner_advice,
        }
    }
}

/// Summary for a persistent-session relation delta update.
pub struct LogicDeltaReport {
    /// Number of relation delta entries supplied by the caller before coalescing.
    pub input_delta_count: usize,
    /// Number of changed relation names in the delta batch.
    pub changed_relations: usize,
    /// Changed relation names after coalescing.
    pub changed_relation_names: Vec<String>,
    /// Total inserted rows across all changed relations.
    pub insert_rows: u64,
    /// Total deleted rows across all changed relations.
    pub delete_rows: u64,
    /// True when at least one relation supplied delete rows.
    pub has_deletes: bool,
    /// Number of SCCs whose dependency closure was affected.
    pub affected_sccs: usize,
    /// Number of affected SCCs that were cleared and fully recomputed.
    pub recomputed_sccs: usize,
    /// Number of affected SCCs updated without clearing prior output.
    pub incremental_sccs: usize,
    /// Net insert rows after batch coalescing and insert/delete cancellation.
    pub coalesced_insert_rows: u64,
    /// Net delete rows after batch coalescing and insert/delete cancellation.
    pub coalesced_delete_rows: u64,
    /// Rows canceled because an insert and delete for the same relation matched in the batch.
    pub canceled_rows: u64,
    /// Planner-grade cache, fallback, and speedup telemetry.
    pub planner_telemetry: DeltaPlannerTelemetry,
    /// Metadata-only debug trace for the delta recompute.
    pub debug_trace: Vec<String>,
}

/// Direction of an incoming relation-delta occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationDeltaDirection {
    /// Tuples supplied on the insertion side of an update.
    Insert,
    /// Tuples supplied on the deletion side of an update.
    Delete,
}

/// Tuples canceled at one ordered merge step of a relation-delta batch.
pub struct RelationDeltaCancellation {
    update_index: usize,
    incoming_direction: RelationDeltaDirection,
    tuples: CudaBuffer,
}

impl RelationDeltaCancellation {
    /// Return the zero-based position of the incoming update in the original batch.
    pub fn update_index(&self) -> usize {
        self.update_index
    }

    /// Return the direction of the incoming occurrence that caused the cancellation.
    pub fn incoming_direction(&self) -> RelationDeltaDirection {
        self.incoming_direction
    }

    /// Borrow the GPU-resident tuples canceled at this merge step.
    pub fn tuples(&self) -> &CudaBuffer {
        &self.tuples
    }
}

#[derive(Clone, Copy)]
struct PreparedRelationDeltaReportSeed {
    input_delta_count: usize,
    changed_relations: usize,
    coalesced_insert_rows: u64,
    coalesced_delete_rows: u64,
    canceled_rows: u64,
}

/// Device-coalesced relation updates prepared for validation and later application.
///
/// Dropping this value releases its GPU-resident net-delta and cancellation
/// buffers without changing any relation store.
#[must_use = "prepared relation deltas have no effect until they are committed"]
pub struct PreparedRelationDeltaBatch {
    deltas: HashMap<String, RelationDelta>,
    cancellations: HashMap<String, Vec<RelationDeltaCancellation>>,
    report_seed: PreparedRelationDeltaReportSeed,
}

impl PreparedRelationDeltaBatch {
    /// Borrow the final net relation deltas produced by the device coalescer.
    pub fn net_deltas(&self) -> &HashMap<String, RelationDelta> {
        &self.deltas
    }

    /// Borrow per-relation cancellation traces in global update order.
    pub fn cancellations(&self) -> &HashMap<String, Vec<RelationDeltaCancellation>> {
        &self.cancellations
    }

    fn into_application_parts(
        self,
    ) -> (
        HashMap<String, RelationDelta>,
        PreparedRelationDeltaReportSeed,
    ) {
        (self.deltas, self.report_seed)
    }
}

/// A fully staged relation update bound to its authoritative and derived state.
///
/// The exclusive borrows prevent callers from mutating or substituting the
/// authoritative store, cache, or runtime between preparation and commit.
/// Dropping this value discards every staged update and prospective derived
/// state. The authoritative store remains unchanged, while the borrowed cache
/// and runtime slots remain empty because their prior values were consumed by
/// preparation.
///
/// A prepared commit has no API for selecting a different destination:
///
/// ```compile_fail
/// use xlog_gpu::logic::PreparedRelationDeltaCommit;
/// use xlog_runtime::RelationStore;
///
/// fn commit_into_another_store(
///     prepared: PreparedRelationDeltaCommit<'_>,
///     other: &mut RelationStore,
/// ) {
///     prepared.commit(other);
/// }
/// ```
///
/// The source store also remains exclusively borrowed until commit:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use xlog_core::Result;
/// use xlog_cuda::{CudaBuffer, CudaKernelProvider};
/// use xlog_gpu::logic::{
///     LogicMaterializedStore, LogicProgram, LogicSessionRuntime, PreparedRelationDeltaBatch,
/// };
/// use xlog_runtime::RelationStore;
///
/// fn mutate_after_prepare(
///     program: &LogicProgram,
///     provider: Arc<CudaKernelProvider>,
///     store: &mut RelationStore,
///     cache: &mut Option<LogicMaterializedStore>,
///     runtime: &mut Option<LogicSessionRuntime>,
///     batch: PreparedRelationDeltaBatch,
///     replacement: CudaBuffer,
/// ) -> Result<()> {
///     let prepared = program.prepare_relation_delta_commit_with_session_runtime(
///         provider, store, cache, runtime, batch,
///     )?;
///     store.put("fact", replacement);
///     prepared.commit();
///     Ok(())
/// }
/// ```
#[must_use = "dropping a prepared commit discards its staged relation updates"]
pub struct PreparedRelationDeltaCommit<'a> {
    provider: Arc<CudaKernelProvider>,
    authoritative_relation_store: &'a mut RelationStore,
    cached_store_slot: &'a mut Option<LogicMaterializedStore>,
    session_runtime_slot: &'a mut Option<LogicSessionRuntime>,
    staged_base_updates: Vec<(String, CudaBuffer)>,
    prospective_cached_store: Option<LogicMaterializedStore>,
    prospective_session_runtime: Option<LogicSessionRuntime>,
    report: LogicDeltaReport,
}

impl PreparedRelationDeltaCommit<'_> {
    /// Borrow the prospective materialized derived/cache store.
    ///
    /// This store is suitable for direct query-result comparison. It must not
    /// seed an independent full recompute because it contains intensional heads
    /// that an executor may union with newly derived rows. For a no-op batch it
    /// returns retained derived state when available, or the unchanged
    /// authoritative store otherwise.
    pub fn prospective_derived_store(&self) -> &RelationStore {
        if let Some(store) = self.prospective_cached_store.as_ref() {
            return &store.store;
        }
        if let Some(runtime) = self.prospective_session_runtime.as_ref() {
            return runtime.executor.store();
        }
        &*self.authoritative_relation_store
    }

    /// Clone the authoritative base snapshot with every staged base update overlaid.
    ///
    /// This fallible, on-demand snapshot is the correct seed for an independent
    /// full recompute. It includes staged relations that were absent from the
    /// authoritative store and does not mutate authoritative contents or
    /// versions. Ordinary prepare and commit paths do not pay for these clones.
    pub fn clone_prospective_base_store(&self) -> Result<RelationStore> {
        let mut authoritative_names = self
            .authoritative_relation_store
            .names()
            .filter(|name| {
                !self
                    .staged_base_updates
                    .iter()
                    .any(|(staged_name, _)| staged_name == name)
            })
            .collect::<Vec<_>>();
        authoritative_names.sort_unstable();
        let mut cloned = RelationStore::new(self.provider.clone());
        cloned.try_reserve_relations(authoritative_names.len() + self.staged_base_updates.len())?;

        for name in authoritative_names {
            let buffer = self.authoritative_relation_store.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Authoritative relation {name} disappeared while cloning prospective base state"
                ))
            })?;
            let context = format!("cloning prospective base relation '{name}'");
            let cloned_buffer = self
                .provider
                .clone_buffer(buffer)
                .map_err(|error| relation_clone_error(context, error))?;
            cloned.put(name, cloned_buffer);
        }

        for (name, buffer) in &self.staged_base_updates {
            let context = format!("cloning staged prospective base relation '{name}'");
            let cloned_buffer = self
                .provider
                .clone_buffer(buffer)
                .map_err(|error| relation_clone_error(context, error))?;
            cloned.put(name, cloned_buffer);
        }
        Ok(cloned)
    }

    /// Install every staged base update and derived-state replacement together.
    ///
    /// All allocation, destination-capacity reservation, recomputation,
    /// constraint validation, and buffer cloning has completed before this
    /// method is available, so committing only moves owned values and is
    /// infallible.
    pub fn commit(self) -> LogicDeltaReport {
        for (name, buffer) in self.staged_base_updates {
            self.authoritative_relation_store.put_owned(name, buffer);
        }
        *self.cached_store_slot = self.prospective_cached_store;
        *self.session_runtime_slot = self.prospective_session_runtime;
        self.report
    }
}

#[derive(Default)]
struct PendingRelationDelta {
    insert: Option<CudaBuffer>,
    delete: Option<CudaBuffer>,
}

/// One stratum of a stratified epistemic plan: the epistemic head(s) it
/// materializes plus the GPU executable plan that computes them.
///
/// Lower strata are executed first; their GATED head outputs are written into the
/// relation store as base relations BEFORE higher strata run, so a higher
/// stratum's `know`/`possible` over a lower head gates against the materialized
/// (now-base) relation through the existing tuple-key membership filter.
#[derive(Clone)]
struct StratumExecutable {
    /// The stratum's GPU plan: single-head or joint multi-head split. The gated
    /// head relation name(s) are recovered from the plan's reductions at runtime.
    plan: StratumPlanKind,
}

#[derive(Clone)]
struct StratifiedExecutable {
    strata: Vec<StratumExecutable>,
    /// Single ordinary closure and authored-constraint stage executed only after
    /// every modal stratum has materialized its gated heads.
    ordinary_post: GpuOrdinaryPass,
}

#[derive(Clone)]
enum StratumPlanKind {
    Single(Box<EpistemicExecutablePlan>),
    Split(Box<EpistemicSplitExecutablePlan>),
    /// A higher stratum that RECURSES over a lower stratum's materialized
    /// (now-base) determined head. Once the determined head is a base relation in
    /// the store, its `know`/`possible` modal is over an invariant relation, so the
    /// stratum is admissible Case-A: the modal resolves to an ordinary join (no
    /// second gate) and the recursive semi-naive engine iterates the fixpoint. The
    /// reduced ordinary program drives an ordinary RIR plan whose head IS this
    /// stratum's user-visible output relation.
    Ordinary {
        plan: Box<ExecutionPlan>,
        /// User-visible output head predicate(s) this stratum computes.
        head_predicates: Vec<String>,
    },
}

#[derive(Clone)]
enum LogicExecutionPlan {
    Ordinary(Box<ExecutionPlan>),
    EpistemicG91Compatibility(Box<EpistemicG91CompatibilityGpuPlan>),
    EpistemicWfsGpu(Box<EpistemicWfsGpuPlan>),
    EpistemicSingle(Box<EpistemicExecutablePlan>),
    EpistemicSplit(Box<EpistemicSplitExecutablePlan>),
    /// Stratified epistemic execution: ordered strata, each materializing its
    /// gated head(s) into the store before the next stratum runs.
    EpistemicStratified(Box<StratifiedExecutable>),
}

#[derive(Clone)]
struct EpistemicG91CompatibilityGpuPlan {
    upper_bound: GpuEvaluationPass,
    refinement: GpuEvaluationPass,
    snapshot_relations: BTreeMap<String, String>,
    convergence_predicates: Vec<String>,
    max_iterations: usize,
}

#[derive(Clone)]
enum GpuEvaluationPass {
    Ordinary(Box<GpuOrdinaryPass>),
    Wfs(Box<EpistemicWfsGpuPlan>),
}

#[derive(Clone)]
struct EpistemicWfsGpuPlan {
    overapprox: GpuOrdinaryPass,
    lower: GpuOrdinaryPass,
    upper: GpuOrdinaryPass,
    intensional_predicates: Vec<String>,
    upper_fixed_names: HashMap<String, String>,
    lower_fixed_names: HashMap<String, String>,
    max_iterations: usize,
}

#[derive(Clone)]
struct GpuOrdinaryPass {
    plan: ExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
}

/// Compile-time epistemic provenance, retained even when the executable plan is
/// `Ordinary` (e.g. a Case-A recursive epistemic fixpoint whose modal literals were
/// resolved into invariant joins). This carries the source's epistemic literals so
/// the epistemic plan dump can emit a stable id for a recursive epistemic fixpoint that no
/// longer carries an epistemic GPU plan.
#[derive(Clone)]
struct EpistemicProvenance {
    /// How the epistemic source was reduced for execution.
    reduction: &'static str,
    /// Epistemic `know`/`possible` literals (with negation) seen in the source EIR.
    literals: Vec<xlog_ir::EirEpistemicLiteral>,
    /// Whether ordinary-plan query results retain the source epistemic relation name
    /// and logical zero-column shape instead of exposing compiler projection details.
    surface_source_queries: bool,
}

/// A compiled Datalog program ready for GPU evaluation.
#[derive(Clone)]
pub struct LogicProgram {
    reusable_state_identity: Arc<LogicProgramIdentity>,
    /// Merged authored program retained for public diagnostics and result labels.
    source_program: Program,
    /// Normalized or reduced program used by compilation and execution.
    program: Program,
    /// Integrity constraints as authored before normalization, retained only when
    /// their order matches the source-order constraint provenance used by the plan.
    authored_constraints: Option<Vec<Constraint>>,
    plan: LogicExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
    /// `Some` iff the source program contained epistemic literals (regardless of
    /// whether the executable plan ended up epistemic or ordinary).
    epistemic_provenance: Option<EpistemicProvenance>,
}

/// Read-only metadata for one argument of a compiled relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicArgumentSchema {
    name: String,
    source_named: bool,
    sort: Option<String>,
    scalar_type: ScalarType,
}

impl LogicArgumentSchema {
    /// Return the compiled column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return whether the column name was written in the source declaration.
    pub fn source_named(&self) -> bool {
        self.source_named
    }

    /// Return the source domain alias, when the argument used one.
    pub fn sort(&self) -> Option<&str> {
        self.sort.as_deref()
    }

    /// Return the resolved scalar type from the compiled relation schema.
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
}

impl LogicProgram {
    /// Compile a Datalog source string into a GPU-executable program.
    pub fn compile(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;
        Self::compile_program(program)
    }

    /// Compile an already parsed program into a GPU-executable program.
    ///
    /// This method does not resolve imports; import-aware callers merge them
    /// first. The program enters the canonical execution normalizer once here.
    pub fn compile_program(mut program: Program) -> Result<Self> {
        if program.authored_constraint_source_bound.is_some() {
            program.validate_prepared_authored_constraint_identity()?;
        } else {
            program.prepare_authored_constraint_identity_at_root()?;
        }
        let source_program = program.clone();
        let normalized = normalize_program_for_execution(program)?;
        Self::compile_normalized_program(normalized, source_program)
    }

    fn compile_normalized_program(normalized: Program, source_program: Program) -> Result<Self> {
        for query_index in 0..normalized.queries.len() {
            let generated_head = format!("__xlog_query_{query_index}");
            let authored_collision = source_program
                .predicates
                .iter()
                .any(|declaration| declaration.name == generated_head)
                || source_program
                    .rules
                    .iter()
                    .any(|rule| rule.head.predicate == generated_head);
            if authored_collision {
                return Err(XlogError::Compilation(format!(
                    "authored relation {generated_head} collides with generated query head"
                )));
            }
        }
        // Function, meta-term, list, and shared-variable normalization preserve
        // constraint count and source order. Keep the authored snapshot only
        // while that one-to-one invariant remains observable.
        let authored_constraints = (source_program.constraints.len()
            == normalized.constraints.len())
        .then(|| source_program.constraints.clone());
        let reusable_state_identity = Arc::new(LogicProgramIdentity::new());
        let compiled = if program_has_epistemic_literals(&normalized) {
            Self::compile_epistemic_program(
                normalized,
                source_program,
                authored_constraints,
                reusable_state_identity,
            )?
        } else {
            let mut compiler = Compiler::new();
            let plan = match qualify_same_name_multi_arity_program(&normalized) {
                Some(qualified) => compiler.compile_prepared_program(&qualified)?,
                None => compiler.compile_prepared_program(&normalized)?,
            };
            let mut schemas = compiler.schemas().clone();
            augment_same_name_multi_arity_schemas(&normalized, &mut schemas)?;
            Self {
                reusable_state_identity,
                source_program,
                program: normalized,
                authored_constraints,
                plan: LogicExecutionPlan::Ordinary(Box::new(plan)),
                schemas,
                rel_ids: compiler.rel_ids().clone(),
                epistemic_provenance: None,
            }
        };
        Ok(compiled.finalize_compilation())
    }

    fn finalize_compilation(self) -> Self {
        if !self.program.queries.is_empty() {
            if let LogicExecutionPlan::Ordinary(plan) = &self.plan {
                let _ = self.resident_certified_plan_for_plan(plan);
            }
        }
        self
    }

    fn validate_reusable_state_identity(
        &self,
        state_identity: &Arc<LogicProgramIdentity>,
        state_name: &str,
    ) -> Result<()> {
        if Arc::ptr_eq(&self.reusable_state_identity, state_identity) {
            return Ok(());
        }
        Err(XlogError::Execution(format!(
            "{state_name} belongs to a different compiled logic program"
        )))
    }

    fn validate_reusable_state_slots(
        &self,
        cached_store: Option<&LogicMaterializedStore>,
        session_runtime: Option<&LogicSessionRuntime>,
    ) -> Result<()> {
        if let Some(cached_store) = cached_store {
            self.validate_reusable_state_identity(
                &cached_store.reusable_state_identity,
                "materialized cache",
            )?;
        }
        if let Some(session_runtime) = session_runtime {
            self.validate_reusable_state_identity(
                &session_runtime.reusable_state_identity,
                "session runtime",
            )?;
        }
        Ok(())
    }

    fn bind_materialized_store(&self, store: RelationStore) -> LogicMaterializedStore {
        LogicMaterializedStore {
            reusable_state_identity: self.reusable_state_identity.clone(),
            store,
        }
    }

    fn compile_epistemic_program(
        normalized: Program,
        source_program: Program,
        authored_constraints: Option<Vec<Constraint>>,
        reusable_state_identity: Arc<LogicProgramIdentity>,
    ) -> Result<Self> {
        // Capture epistemic provenance up front: the source-EIR modal literals are
        // retained even when a Case-A recursive reduction lowers the program to an
        // Ordinary executable plan, so the epistemic plan dump can still emit a stable id
        // for a recursive epistemic fixpoint.
        let provenance_literals = collect_eir_epistemic_literals(&normalized);
        let prepared = prepare_epistemic_program(&normalized)?;
        let active_program = prepared.active_program();

        // Positive Gelfond-1991 `possible` cycles require tuple-level dynamic
        // compatibility. Predicate SCC membership alone cannot prove that the same
        // concrete tuple survives every edge's ordinary body filters. Route the
        // complete active program through a descending GPU fixpoint before any
        // stratified or ordinary least-fixpoint reduction can erase those gates.
        if let Some(reduction) = try_prepare_g91_compatibility_reduction(&prepared)? {
            let plan = compile_g91_compatibility_gpu_plan(&reduction)?;
            let schemas = g91_plan_combined_schemas(&plan);
            let rel_ids = g91_plan_combined_rel_ids(&plan);
            return Ok(Self {
                reusable_state_identity,
                source_program,
                program: reduction.refinement_program().clone(),
                authored_constraints,
                plan: LogicExecutionPlan::EpistemicG91Compatibility(Box::new(plan)),
                schemas,
                rel_ids,
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "g91_tuple_compatibility",
                    literals: provenance_literals,
                    surface_source_queries: true,
                }),
            });
        }

        // Stratified epistemic execution FIRST: a modal literal ranges over an
        // epistemically-DETERMINED derived head (`b :- know a` where `a :- know p`,
        // `p` invariant — possibly with the higher stratum RECURSING over the
        // determined head, e.g. `reach :- reach, know a`). Partition into strata;
        // each is compiled through the existing epistemic OR Case-A ordinary path,
        // and at runtime each lower stratum's GATED head is materialized into the
        // store as a base relation before the higher stratum gates against it (via
        // the existing tuple-key membership filter or — once the head is a materialized
        // base relation — Case-A resolve-into-body; either way NO double-gating
        // against a still-modal relation). A shared BASE modal `q` (EDB, not a
        // determined derived head) returns `None` here and falls through to
        // the joint path UNCHANGED; plain Case-A recursion over an EDB modal
        // (`know edge`) also returns `None` and falls through to Case-A below.
        if let Some(stratified) = try_plan_stratified_epistemic_program(active_program)? {
            // SCHEMA-ONLY reduction: resolve augmenting positive modals over INVARIANT
            // *or* epistemically-DETERMINED targets into positive ordinary atoms, so an
            // augmented head whose extra output column is bound by a modal over a
            // multi-column determined head (`out(X) :- node(X), know r(X, Y)`, `r`
            // determined) types its appended `Y` column from `r`'s declaration instead
            // of failing closed as `UnsafeVariable`. This drives ONLY plan schema
            // inference; per-stratum EXECUTION compiles below over sub-programs where
            // the determined head is already a materialized base relation (strict
            // invariant resolve), so no modal is ever resolved over an un-gated
            // candidate at runtime.
            let reduced =
                reduce_epistemic_program_to_ordinary_for_stratified_schema(active_program)?;
            let mut schema_compiler = Compiler::new();
            schema_compiler.compile_prepared_program(&reduced)?;
            let mut schemas = schema_compiler.schemas().clone();
            augment_same_name_multi_arity_schemas(active_program, &mut schemas)?;

            let mut strata = Vec::with_capacity(stratified.strata.len());
            for stratum in &stratified.strata {
                strata.push(StratumExecutable {
                    plan: Self::compile_stratum_plan(&stratum.program)?,
                });
            }
            let ordinary_post = compile_gpu_ordinary_pass(&stratified.ordinary_post_program)?;
            for (name, schema) in &ordinary_post.schemas {
                schemas
                    .entry(name.clone())
                    .or_insert_with(|| schema.clone());
            }
            let plan = LogicExecutionPlan::EpistemicStratified(Box::new(StratifiedExecutable {
                strata,
                ordinary_post,
            }));
            let rel_ids = epistemic_relation_ids(&plan)?;
            return Ok(Self {
                reusable_state_identity,
                source_program,
                program: normalized,
                authored_constraints,
                plan,
                schemas,
                rel_ids,
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "stratified",
                    literals: provenance_literals,
                    surface_source_queries: true,
                }),
            });
        }

        // Case A/B: reduce admissible recursive epistemic programs to ordinary
        // recursion. Stratified reduced programs route through the existing ordinary
        // semi-naive engine; non-monotone reduced SCCs route through the GPU-native
        // WFS alternating-fixpoint plan below. Recursive shapes outside the admissible
        // fragment still fail closed in `try_reduce_case_a_recursive_epistemic_program`.
        if let Some(recursive_reduced) = try_reduce_prepared_recursive_epistemic_program(&prepared)?
        {
            let strat = xlog_logic::stratify::analyze_stratification(&recursive_reduced);
            if !strat.non_monotone_sccs.is_empty() {
                let wfs_plan = compile_epistemic_wfs_gpu_plan(&recursive_reduced)?;
                let schemas = wfs_plan_combined_schemas(&wfs_plan);
                let rel_ids = wfs_plan_combined_rel_ids(&wfs_plan);
                return Ok(Self {
                    reusable_state_identity,
                    source_program,
                    program: recursive_reduced,
                    authored_constraints,
                    plan: LogicExecutionPlan::EpistemicWfsGpu(Box::new(wfs_plan)),
                    schemas,
                    rel_ids,
                    epistemic_provenance: Some(EpistemicProvenance {
                        reduction: "wfs_gpu_recursive",
                        literals: provenance_literals,
                        surface_source_queries: true,
                    }),
                });
            }
            let mut compiler = Compiler::new();
            let plan = compiler.compile_prepared_program(&recursive_reduced)?;
            return Ok(Self {
                reusable_state_identity,
                source_program,
                program: recursive_reduced,
                authored_constraints,
                plan: LogicExecutionPlan::Ordinary(Box::new(plan)),
                schemas: compiler.schemas().clone(),
                rel_ids: compiler.rel_ids().clone(),
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "ordinary_recursive_modal_reduction",
                    literals: provenance_literals,
                    surface_source_queries: true,
                }),
            });
        }

        let reduced = reduce_epistemic_program_to_ordinary(active_program)?;
        let mut schema_compiler = Compiler::new();
        schema_compiler.compile_prepared_program(&reduced)?;
        let mut schemas = schema_compiler.schemas().clone();
        augment_same_name_multi_arity_schemas(active_program, &mut schemas)?;

        let plan = if epistemic_output_head_predicate_count(active_program) > 1 {
            LogicExecutionPlan::EpistemicSplit(Box::new(compile_epistemic_gpu_split_execution(
                active_program,
            )?))
        } else {
            match compile_epistemic_gpu_execution(active_program) {
                Ok(executable) => LogicExecutionPlan::EpistemicSingle(Box::new(executable)),
                Err(XlogError::UnsupportedEpistemicConstruct { construct, .. })
                    if construct == "epistemic GPU final output relation" =>
                {
                    LogicExecutionPlan::EpistemicSplit(Box::new(
                        compile_epistemic_gpu_split_execution(active_program)?,
                    ))
                }
                Err(err) => return Err(err),
            }
        };
        let rel_ids = epistemic_relation_ids(&plan)?;
        Ok(Self {
            reusable_state_identity,
            source_program,
            program: normalized,
            authored_constraints,
            plan,
            schemas,
            rel_ids,
            epistemic_provenance: Some(EpistemicProvenance {
                reduction: "epistemic_executable",
                literals: provenance_literals,
                surface_source_queries: false,
            }),
        })
    }

    /// Compile one stratum sub-program into its plan kind.
    ///
    /// A stratum whose epistemic heads gate only over invariant or
    /// already-materialized lower-stratum relations is either an admissible Case-A
    /// recursion (the modal resolves to an ordinary join over the now-base relation)
    /// or a plain single/joint epistemic plan. Case-A is tried first so a recursive
    /// higher stratum (`reach :- reach, know a`, `a` materialized base) routes
    /// through the ordinary semi-naive engine.
    fn compile_stratum_plan(stratum_program: &Program) -> Result<StratumPlanKind> {
        if let Some(case_a_reduced) =
            try_reduce_case_a_recursive_epistemic_program(stratum_program)?
        {
            let mut compiler = Compiler::new();
            let plan = compiler.compile_prepared_program(&case_a_reduced)?;
            let head_predicates = epistemic_stratum_output_heads(stratum_program);
            return Ok(StratumPlanKind::Ordinary {
                plan: Box::new(plan),
                head_predicates,
            });
        }
        if epistemic_output_head_predicate_count(stratum_program) > 1 {
            Ok(StratumPlanKind::Split(Box::new(
                compile_epistemic_gpu_split_execution(stratum_program)?,
            )))
        } else {
            Ok(StratumPlanKind::Single(Box::new(
                compile_epistemic_gpu_execution(stratum_program)?,
            )))
        }
    }

    /// Compile a program with module resolution.
    ///
    /// This method resolves all imports using the provided resolver and merges
    /// imported predicates, functions, and rules into the main program.
    ///
    /// Pragmas are entry-file-scoped: directives declared in imported modules
    /// are dropped at merge time. This library entry point does not surface
    /// them — embedders that want the CLI's `warning[W0510]` behavior should
    /// call `resolver.ignored_import_pragmas()` before compiling and report
    /// the returned records on their own diagnostics channel.
    ///
    /// # Arguments
    /// * `source` - The source code of the main program
    /// * `resolver` - A pre-loaded ModuleResolver with all dependencies resolved
    ///
    /// # Returns
    /// The compiled LogicProgram with all imports merged
    pub fn compile_with_resolver(
        source: &str,
        resolver: &xlog_logic::resolver::ModuleResolver,
    ) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;

        // Merge imports from the resolver
        let merged = resolver
            .merge_imports(program)
            .map_err(|e| XlogError::Compilation(format!("Module resolution failed: {}", e)))?;

        Self::compile_program(merged)
    }

    /// Serialize the compiled epistemic execution plan to a JSON summary.
    ///
    /// Returns `None` for ordinary (non-epistemic) programs. For epistemic
    /// programs this dumps the EIR-derived GPU plan(s): selected mode, the
    /// epistemic `know`/`possible` literals (with negation), required GPU hot-path
    /// phases/kernels, world-view integrity constraints, reduced-program head
    /// summaries, the fail-closed GPU execution policy, and a deterministic plan
    /// id (a stable hash of the canonical summary). Runtime evidence separately
    /// records observed dispatch, kernel, device-buffer, candidate-accounting,
    /// solver/probability event, and scoped-transfer behavior.
    pub fn epistemic_plan_json(&self) -> Option<String> {
        let mut has_ordinary_post = false;
        let gpu_plans: Vec<(String, &xlog_ir::EpistemicGpuPlan)> = match &self.plan {
            // A program whose source was epistemic but whose executable plan is
            // ordinary either resolved admissible recursive modal literals into joins
            // or removed every unfounded FAEEL modal rule. It carries no epistemic GPU
            // plan and executes through the ordinary GPU engine under the same
            // reject-unsupported policy. Emit a provenance summary with a stable id so the reduction is
            // auditable.
            LogicExecutionPlan::Ordinary(_) => {
                let prov = self.epistemic_provenance.as_ref()?;
                return Some(epistemic_provenance_summary_json(
                    "epistemic_reduced_ordinary",
                    prov,
                    None,
                    None,
                ));
            }
            LogicExecutionPlan::EpistemicWfsGpu(wfs) => {
                let prov = self.epistemic_provenance.as_ref()?;
                return Some(epistemic_provenance_summary_json(
                    self.plan_kind_label(),
                    prov,
                    Some(wfs.max_iterations),
                    Some(wfs),
                ));
            }
            LogicExecutionPlan::EpistemicG91Compatibility(g91) => {
                let prov = self.epistemic_provenance.as_ref()?;
                return Some(g91_compatibility_summary_json(
                    self.plan_kind_label(),
                    prov,
                    g91,
                ));
            }
            LogicExecutionPlan::EpistemicSingle(plan) => {
                vec![("single".to_string(), &plan.gpu_plan)]
            }
            LogicExecutionPlan::EpistemicSplit(split) => split
                .components
                .iter()
                .enumerate()
                .map(|(i, c)| (format!("split[{i}]"), &c.executable.gpu_plan))
                .collect(),
            LogicExecutionPlan::EpistemicStratified(stratified) => {
                let mut plans = Vec::new();
                for (i, stratum) in stratified.strata.iter().enumerate() {
                    match &stratum.plan {
                        StratumPlanKind::Single(plan) => {
                            plans.push((format!("stratum[{i}]"), &plan.gpu_plan));
                        }
                        StratumPlanKind::Split(split) => {
                            for (j, c) in split.components.iter().enumerate() {
                                plans.push((
                                    format!("stratum[{i}].split[{j}]"),
                                    &c.executable.gpu_plan,
                                ));
                            }
                        }
                        // Recursive/ordinary higher strata carry no epistemic GPU
                        // plan (the modal already resolved to an ordinary join over
                        // a materialized base); the enclosing summary carries the
                        // same fail-closed GPU execution policy.
                        StratumPlanKind::Ordinary { .. } => {}
                    }
                }
                has_ordinary_post = true;
                plans
            }
        };
        Some(epistemic_plan_summary_json(
            self.plan_kind_label(),
            &gpu_plans,
            has_ordinary_post,
        ))
    }

    fn plan_kind_label(&self) -> &'static str {
        match &self.plan {
            LogicExecutionPlan::Ordinary(_) => "ordinary",
            LogicExecutionPlan::EpistemicG91Compatibility(_) => "epistemic_g91_compatibility_gpu",
            LogicExecutionPlan::EpistemicWfsGpu(_) => "epistemic_wfs_gpu",
            LogicExecutionPlan::EpistemicSingle(_) => "epistemic_single",
            LogicExecutionPlan::EpistemicSplit(_) => "epistemic_split",
            LogicExecutionPlan::EpistemicStratified(_) => "epistemic_stratified",
        }
    }

    /// Look up the schema for a named relation.
    pub fn schema(&self, relation: &str) -> Option<&Schema> {
        self.schemas.get(relation)
    }

    /// Return the full schema map (relation name to schema).
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

    /// Return ordered argument metadata for a compiled relation.
    ///
    /// Column names and scalar types come from the compiled schema. Source
    /// declarations additionally preserve whether each name was explicit and
    /// which domain alias, if any, supplied its scalar type.
    pub fn argument_schema(&self, relation: &str) -> Option<Vec<LogicArgumentSchema>> {
        let schema = self.schemas.get(relation)?;
        let presentation_program = self.presentation_program();
        let source_declaration = presentation_program
            .predicates
            .iter()
            .rev()
            .find(|decl| arity_qualified_name(&decl.name, decl.arity()) == relation)
            .or_else(|| {
                presentation_program
                    .predicates
                    .iter()
                    .rev()
                    .find(|decl| decl.name == relation)
            });
        let source_columns = source_declaration.map(|declaration| declaration.schema_columns());

        Some(
            schema
                .columns
                .iter()
                .enumerate()
                .map(|(index, (name, scalar_type))| {
                    let source_column = source_columns
                        .as_ref()
                        .and_then(|columns| columns.get(index));
                    LogicArgumentSchema {
                        name: name.clone(),
                        source_named: source_column
                            .and_then(|column| column.name.as_ref())
                            .is_some(),
                        sort: source_column.and_then(|column| match &column.typ {
                            TypeRef::Domain(name) => Some(name.clone()),
                            _ => None,
                        }),
                        scalar_type: *scalar_type,
                    }
                })
                .collect(),
        )
    }

    /// Return stable rule provenance for source-visible rules.
    pub fn rule_provenance(&self) -> Vec<xlog_logic::RuleProvenance> {
        xlog_logic::source_diagnostics(&self.source_program, &self.program, None).0
    }

    /// Return direct proof traces for source queries.
    pub fn proof_traces(&self) -> Vec<xlog_logic::QueryProofTrace> {
        xlog_logic::source_diagnostics(&self.source_program, &self.program, None).1
    }

    fn presentation_program(&self) -> &Program {
        &self.source_program
    }

    /// Create a persistent user-visible relation store initialized with inline facts.
    pub fn create_relation_store(
        &self,
        provider: Arc<CudaKernelProvider>,
    ) -> Result<RelationStore> {
        let mut store = RelationStore::new(provider.clone());
        for (name, schema) in &self.schemas {
            if is_user_visible_relation(name) || is_list_helper_relation(name) {
                store.put(name, provider.create_empty_buffer(schema.clone())?);
            }
        }
        self.load_facts_into_store(provider.as_ref(), &mut store)?;
        Ok(store)
    }

    /// Evaluate using a persistent base relation store.
    ///
    /// The provided store is treated as immutable seed state. Buffers are cloned
    /// into a fresh executor for each evaluation so repeated evaluations reuse
    /// stored relations without mutating the persistent store itself.
    pub fn evaluate_with_relation_store(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        let (result, _) =
            self.evaluate_with_relation_store_and_cache(provider, relation_store, profiling)?;
        Ok(result)
    }

    /// Evaluate using a persistent relation store and return the complete runtime store.
    pub fn evaluate_with_relation_store_and_cache(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
        profiling: bool,
    ) -> Result<(LogicEvalResult, LogicMaterializedStore)> {
        self.reject_compiler_generated_query_relation_names(
            relation_store.names(),
            "persistent caller",
        )?;
        let resident_mode = ResidentSelectionMode::from_env()?;
        let mut executor =
            self.executor_from_materialized_store(provider.clone(), relation_store, profiling)?;
        executor.execute_plan(self.ordinary_plan("relation-store evaluation")?)?;
        self.enforce_constraints(&provider, &executor)?;

        let total_output_rows = self.total_query_rows(executor.store())?;
        let mut stats = if profiling {
            Some(executor.execution_stats(total_output_rows))
        } else {
            None
        };
        if resident_mode.enabled() {
            if resident_mode == ResidentSelectionMode::Require {
                return Err(XlogError::Execution(
                    "resident conditional-graph execution was required, but complete-store evaluation requires the existing GPU path"
                        .to_string(),
                ));
            }
            if let Some(stats) = stats.as_mut() {
                stats.resident_graph = Some(ResidentGraphExecutionStats::declined(
                    ResidentGraphDeclineReason::FullStoreRequested,
                ));
            }
        }

        let cached_store = self.clone_relation_store(&provider, executor.store())?;
        let result = self.logic_result_from_store(provider.as_ref(), &cached_store, stats)?;
        Ok((result, self.bind_materialized_store(cached_store)))
    }

    /// Create retained runtime state for a persistent relation session.
    pub fn create_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
        profiling: bool,
    ) -> Result<LogicSessionRuntime> {
        self.reject_compiler_generated_query_relation_names(
            relation_store.names(),
            "persistent caller",
        )?;
        self.ordinary_plan("persistent relation session")?;
        let executor =
            self.executor_from_materialized_store(provider, relation_store, profiling)?;
        Ok(LogicSessionRuntime {
            reusable_state_identity: self.reusable_state_identity.clone(),
            executor,
            profiling,
        })
    }

    fn create_session_runtime_from_materialized_store(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &LogicMaterializedStore,
        profiling: bool,
    ) -> Result<LogicSessionRuntime> {
        self.validate_reusable_state_identity(
            &relation_store.reusable_state_identity,
            "materialized cache",
        )?;
        self.ordinary_plan("materialized relation session")?;
        Ok(LogicSessionRuntime {
            reusable_state_identity: self.reusable_state_identity.clone(),
            executor: self.executor_from_materialized_store(
                provider,
                &relation_store.store,
                profiling,
            )?,
            profiling,
        })
    }

    /// Evaluate with retained session runtime state and return a materialized store snapshot.
    pub fn evaluate_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        runtime: &mut LogicSessionRuntime,
    ) -> Result<(LogicEvalResult, LogicMaterializedStore)> {
        self.validate_reusable_state_identity(&runtime.reusable_state_identity, "session runtime")?;
        let resident_mode = ResidentSelectionMode::from_env()?;
        runtime.executor.set_profiling(runtime.profiling);
        runtime
            .executor
            .execute_plan(self.ordinary_plan("session runtime evaluation")?)?;
        self.enforce_constraints(&provider, &runtime.executor)?;

        let total_output_rows = self.total_query_rows(runtime.executor.store())?;
        let mut stats = if runtime.profiling {
            Some(runtime.executor.execution_stats(total_output_rows))
        } else {
            None
        };
        if resident_mode.enabled() {
            if resident_mode == ResidentSelectionMode::Require {
                return Err(XlogError::Execution(
                    "resident conditional-graph execution was required, but persistent session evaluation requires the existing GPU path"
                        .to_string(),
                ));
            }
            if let Some(stats) = stats.as_mut() {
                stats.resident_graph = Some(ResidentGraphExecutionStats::declined(
                    ResidentGraphDeclineReason::FullStoreRequested,
                ));
            }
        }

        let cached_store = self.clone_relation_store(&provider, runtime.executor.store())?;
        let result = self.logic_result_from_store(provider.as_ref(), &cached_store, stats)?;
        Ok((result, self.bind_materialized_store(cached_store)))
    }

    /// Build query results from an already materialized runtime store.
    ///
    /// A raw relation store cannot attest that it was materialized by this
    /// compiled program:
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use xlog_core::Result;
    /// use xlog_cuda::CudaKernelProvider;
    /// use xlog_gpu::logic::LogicProgram;
    /// use xlog_runtime::RelationStore;
    ///
    /// fn evaluate_untrusted_store(
    ///     program: &LogicProgram,
    ///     provider: Arc<CudaKernelProvider>,
    ///     raw_store: &RelationStore,
    /// ) -> Result<()> {
    ///     program.evaluate_cached_relation_store(provider, raw_store)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn evaluate_cached_relation_store(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &LogicMaterializedStore,
    ) -> Result<LogicEvalResult> {
        self.validate_reusable_state_identity(
            &relation_store.reusable_state_identity,
            "materialized cache",
        )?;
        self.logic_result_from_store(provider.as_ref(), &relation_store.store, None)
    }

    /// Apply relation deltas to a persistent session store through the runtime delta path.
    ///
    /// A cache from another compiled program is rejected without consuming it.
    /// After identity validation, an operational preparation failure leaves the
    /// authoritative relation store unchanged but discards the consumed cache.
    pub fn apply_relation_deltas(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<LogicMaterializedStore>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
        self.validate_reusable_state_slots(cached_store.as_ref(), None)?;
        let mut session_runtime = None;
        let prepared = self.prepare_relation_delta_commit(
            provider,
            relation_store,
            cached_store,
            &mut session_runtime,
            deltas,
            None,
        )?;
        Ok(prepared.commit())
    }

    /// Apply relation deltas while preserving retained session runtime state.
    ///
    /// State from another compiled program is rejected without consuming it.
    /// After identity validation, an operational preparation failure leaves the
    /// authoritative store unchanged and the derived-state slots empty.
    pub fn apply_relation_deltas_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<LogicMaterializedStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
        self.validate_reusable_state_slots(cached_store.as_ref(), session_runtime.as_ref())?;
        let prepared = self.prepare_relation_delta_commit(
            provider,
            relation_store,
            cached_store,
            session_runtime,
            deltas,
            None,
        )?;
        Ok(prepared.commit())
    }

    /// Prepare raw relation deltas without ordered-batch coalescing.
    ///
    /// This preserves the runtime's delete-then-insert semantics when one
    /// relation delta contains both directions. Callers that accept an ordered
    /// batch must use [`LogicProgram::prepare_relation_delta_batch`] exactly
    /// once and pass its result to
    /// [`LogicProgram::prepare_relation_delta_commit_with_session_runtime`].
    /// Foreign reusable state is rejected before it is consumed. Once identity
    /// validation succeeds, the current cache and runtime are consumed during
    /// preparation; on operational error their caller slots remain empty while
    /// the authoritative store remains unchanged.
    pub fn prepare_relation_deltas_commit_with_session_runtime<'a>(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &'a mut RelationStore,
        cached_store: &'a mut Option<LogicMaterializedStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
        self.validate_reusable_state_slots(cached_store.as_ref(), session_runtime.as_ref())?;
        self.prepare_relation_delta_commit(
            provider,
            relation_store,
            cached_store,
            session_runtime,
            deltas,
            None,
        )
    }

    /// Coalesce an ordered batch on the device and retain cancellation tuples
    /// for selected relations.
    pub fn prepare_relation_delta_batch(
        &self,
        provider: &CudaKernelProvider,
        delta_batch: Vec<(String, RelationDelta)>,
        cancellation_capture_relations: &BTreeSet<String>,
    ) -> Result<PreparedRelationDeltaBatch> {
        self.reject_compiler_generated_query_relation_names(
            delta_batch.iter().map(|(name, _)| name.as_str()),
            "caller delta",
        )?;
        coalesce_relation_delta_batch_with_cancellation_capture(
            provider,
            delta_batch,
            cancellation_capture_relations,
        )
    }

    /// Build the prospective authoritative base store for an independent
    /// full-recompute diagnostic without consuming retained session state.
    ///
    /// This must run before preparing the retained-runtime commit. If cloning
    /// or applying a base delta fails, the caller's cache and session runtime
    /// remain available for subsequent evaluations.
    pub fn clone_prospective_base_for_prepared_delta_batch(
        &self,
        provider: &Arc<CudaKernelProvider>,
        authoritative_relation_store: &RelationStore,
        prepared_batch: &PreparedRelationDeltaBatch,
    ) -> Result<RelationStore> {
        self.reject_compiler_generated_query_relation_names(
            authoritative_relation_store.names(),
            "persistent caller",
        )?;

        let deltas = prepared_batch.net_deltas();
        let mut unchanged_names = authoritative_relation_store
            .names()
            .filter(|name| !deltas.contains_key(*name))
            .collect::<Vec<_>>();
        unchanged_names.sort_unstable();

        let mut changed_names = deltas.keys().map(String::as_str).collect::<Vec<_>>();
        changed_names.sort_unstable();

        let mut prospective = RelationStore::new(provider.clone());
        prospective.try_reserve_relations(unchanged_names.len() + changed_names.len())?;

        for name in unchanged_names {
            let buffer = authoritative_relation_store.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Authoritative relation {name} disappeared while cloning prospective base state"
                ))
            })?;
            let context = format!("cloning prospective base relation '{name}'");
            let cloned = provider
                .clone_buffer(buffer)
                .map_err(|error| relation_clone_error(context, error))?;
            prospective.put(name, cloned);
        }

        for name in changed_names {
            let delta = deltas.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Prepared relation delta for {name} disappeared while cloning prospective base state"
                ))
            })?;
            let existing = authoritative_relation_store.get(name);
            let schema = existing
                .map(|buffer| buffer.schema().clone())
                .or_else(|| delta.insert.as_ref().map(|buffer| buffer.schema().clone()))
                .or_else(|| delta.delete.as_ref().map(|buffer| buffer.schema().clone()))
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "Delta update for {name} has no existing relation and no schema"
                    ))
                })?;
            let context = format!("cloning prospective base relation '{name}'");
            let mut updated = match existing {
                Some(buffer) => provider
                    .clone_buffer(buffer)
                    .map_err(|error| relation_clone_error(context, error))?,
                None => provider.create_empty_buffer(schema)?,
            };
            if let Some(delete) = &delta.delete {
                updated = provider.diff_gpu(&updated, delete)?;
            }
            if let Some(insert) = &delta.insert {
                updated = provider.union_gpu(&updated, insert)?;
            }
            prospective.put(name, updated);
        }

        Ok(prospective)
    }

    /// Prepare a fully staged retained-runtime commit from a coalesced batch.
    ///
    /// Foreign reusable state is rejected before the coalesced batch is consumed.
    /// Once identity validation succeeds, the current runtime and cache move into
    /// the transaction. An operational preparation failure discards those values
    /// and leaves the caller slots empty; the authoritative store is unchanged.
    pub fn prepare_relation_delta_commit_with_session_runtime<'a>(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &'a mut RelationStore,
        cached_store: &'a mut Option<LogicMaterializedStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        prepared_batch: PreparedRelationDeltaBatch,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
        self.validate_reusable_state_slots(cached_store.as_ref(), session_runtime.as_ref())?;
        let (deltas, report_seed) = prepared_batch.into_application_parts();
        self.prepare_relation_delta_commit(
            provider,
            relation_store,
            cached_store,
            session_runtime,
            deltas,
            Some(report_seed),
        )
    }

    fn prepare_relation_delta_commit<'a>(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &'a mut RelationStore,
        cached_store: &'a mut Option<LogicMaterializedStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
        report_seed: Option<PreparedRelationDeltaReportSeed>,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
        self.validate_reusable_state_slots(cached_store.as_ref(), session_runtime.as_ref())?;
        self.reject_compiler_generated_query_relation_names(
            relation_store.names(),
            "persistent caller",
        )?;
        self.reject_compiler_generated_query_relation_names(
            deltas.keys().map(String::as_str),
            "caller delta",
        )?;
        let insert_rows = deltas
            .values()
            .filter_map(|delta| delta.insert.as_ref())
            .map(CudaBuffer::num_rows)
            .sum();
        let delete_rows = deltas
            .values()
            .filter_map(|delta| delta.delete.as_ref())
            .map(CudaBuffer::num_rows)
            .sum();
        let cache_reused = session_runtime.is_some() || cached_store.is_some();
        let mut changed_relation_names = deltas.keys().cloned().collect::<Vec<_>>();
        changed_relation_names.sort();

        let prior_cached_store = cached_store.take();
        let prior_session_runtime = session_runtime.take();

        let missing_relation_count = changed_relation_names
            .iter()
            .filter(|name| !relation_store.contains(name))
            .count();
        relation_store.try_reserve_relations(missing_relation_count)?;

        if deltas.is_empty() {
            if let Some(seed) = report_seed {
                return Ok(PreparedRelationDeltaCommit {
                    provider,
                    authoritative_relation_store: relation_store,
                    cached_store_slot: cached_store,
                    session_runtime_slot: session_runtime,
                    staged_base_updates: Vec::new(),
                    prospective_cached_store: prior_cached_store,
                    prospective_session_runtime: prior_session_runtime,
                    report: no_op_delta_report(seed),
                });
            }
        }

        let mut working_runtime = match prior_session_runtime {
            Some(runtime) => runtime,
            None => {
                if let Some(materialized_store) = prior_cached_store.as_ref() {
                    self.create_session_runtime_from_materialized_store(
                        provider.clone(),
                        materialized_store,
                        false,
                    )?
                } else {
                    self.create_session_runtime(provider.clone(), relation_store, false)?
                }
            }
        };

        if prior_cached_store.is_none() {
            self.evaluate_with_session_runtime(provider.clone(), &mut working_runtime)?;
        }

        let delta_stats = working_runtime.executor.apply_deltas_and_recompute(
            self.ordinary_plan("session relation-delta recompute")?,
            &deltas,
        )?;
        self.enforce_constraints(&provider, &working_runtime.executor)?;

        let mut staged_base_updates = Vec::with_capacity(changed_relation_names.len());
        for name in &changed_relation_names {
            let updated = working_runtime.executor.store().get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Delta relation {} missing after runtime recompute",
                    name
                ))
            })?;
            let context = format!("cloning staged base relation '{name}'");
            staged_base_updates.push((
                name.clone(),
                provider
                    .clone_buffer(updated)
                    .map_err(|error| relation_clone_error(context, error))?,
            ));
        }
        let prospective_cached_store = Some(self.bind_materialized_store(
            self.clone_prepared_relation_snapshot(&provider, working_runtime.executor.store())?,
        ));

        let mut report = logic_delta_report(delta_stats, insert_rows, delete_rows);
        report.changed_relation_names = changed_relation_names;
        report.planner_telemetry =
            DeltaPlannerTelemetry::from_delta_report(&report, cache_reused, None);
        report.debug_trace = delta_debug_trace(&report);
        if let Some(seed) = report_seed {
            report.input_delta_count = seed.input_delta_count;
            report.changed_relations = seed.changed_relations;
            report.coalesced_insert_rows = seed.coalesced_insert_rows;
            report.coalesced_delete_rows = seed.coalesced_delete_rows;
            report.canceled_rows = seed.canceled_rows;
            report.planner_telemetry =
                DeltaPlannerTelemetry::from_delta_report(&report, true, None);
            report.debug_trace = delta_debug_trace(&report);
        }

        Ok(PreparedRelationDeltaCommit {
            provider,
            authoritative_relation_store: relation_store,
            cached_store_slot: cached_store,
            session_runtime_slot: session_runtime,
            staged_base_updates,
            prospective_cached_store,
            prospective_session_runtime: Some(working_runtime),
            report,
        })
    }

    /// Apply an ordered batch of relation deltas after device-side coalescing.
    ///
    /// A fully canceled batch returns a no-op report without changing the
    /// authoritative store or advancing derived runtime state. If preparation
    /// fails, the authoritative store remains unchanged. A foreign cache is
    /// rejected before device coalescing and is not consumed; after successful
    /// identity validation an operational failure may discard the cache.
    pub fn apply_relation_delta_batch(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<LogicMaterializedStore>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
        self.validate_reusable_state_slots(cached_store.as_ref(), None)?;
        self.reject_compiler_generated_query_relation_names(
            relation_store.names(),
            "persistent caller",
        )?;
        let prepared_batch =
            self.prepare_relation_delta_batch(provider.as_ref(), delta_batch, &BTreeSet::new())?;
        let mut session_runtime = None;
        let prepared = self.prepare_relation_delta_commit_with_session_runtime(
            provider,
            relation_store,
            cached_store,
            &mut session_runtime,
            prepared_batch,
        )?;
        Ok(prepared.commit())
    }

    /// Apply an ordered batch of relation deltas while preserving session runtime state.
    ///
    /// A fully canceled batch returns a no-op report without changing the
    /// authoritative store or advancing derived runtime state. If preparation
    /// fails, the authoritative store remains unchanged while the derived cache
    /// and retained runtime slots are left empty. Foreign state is rejected
    /// before device coalescing and remains in the caller slots.
    pub fn apply_relation_delta_batch_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<LogicMaterializedStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
        self.validate_reusable_state_slots(cached_store.as_ref(), session_runtime.as_ref())?;
        self.reject_compiler_generated_query_relation_names(
            relation_store.names(),
            "persistent caller",
        )?;
        let prepared_batch =
            self.prepare_relation_delta_batch(provider.as_ref(), delta_batch, &BTreeSet::new())?;
        let prepared = self.prepare_relation_delta_commit_with_session_runtime(
            provider,
            relation_store,
            cached_store,
            session_runtime,
            prepared_batch,
        )?;
        Ok(prepared.commit())
    }

    /// Evaluate the program with the given input relations (no profiling).
    pub fn evaluate(
        &self,
        provider: Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
    ) -> Result<LogicEvalResult> {
        self.evaluate_with_options(provider, inputs, false)
    }

    fn finish_nonordinary_resident_selection(
        &self,
        mut result: LogicEvalResult,
        mode: ResidentSelectionMode,
    ) -> Result<LogicEvalResult> {
        match mode {
            ResidentSelectionMode::Disabled => Ok(result),
            ResidentSelectionMode::Auto => {
                if let Some(stats) = result.stats.as_mut() {
                    stats.resident_graph = Some(ResidentGraphExecutionStats::declined(
                        ResidentGraphDeclineReason::NonOrdinaryPlan,
                    ));
                }
                Ok(result)
            }
            ResidentSelectionMode::Require => Err(XlogError::Execution(
                "resident conditional-graph execution was required for a non-ordinary program"
                    .to_string(),
            )),
        }
    }

    fn compiler_generated_query_heads(&self) -> Result<BTreeSet<String>> {
        match &self.plan {
            LogicExecutionPlan::Ordinary(plan) => {
                if plan.generated_query_rules.len() != self.program.queries.len() {
                    return Err(XlogError::Execution(format!(
                            "compiler-generated query provenance count {} does not match authored query count {}",
                            plan.generated_query_rules.len(),
                            self.program.queries.len()
                        )));
                }
                let mut heads = BTreeSet::new();
                let mut rule_positions = BTreeSet::new();
                for (position, provenance) in plan.generated_query_rules.iter().enumerate() {
                    if provenance.query_index != position {
                        return Err(XlogError::Execution(format!(
                                "compiler-generated query provenance position {position} carries query index {}",
                                provenance.query_index
                            )));
                    }
                    if !rule_positions.insert((provenance.scc_index, provenance.rule_index)) {
                        return Err(XlogError::Execution(format!(
                                "compiler-generated query provenance {} reuses compiled rule scc={} rule={}",
                                provenance.query_index, provenance.scc_index, provenance.rule_index
                            )));
                    }
                    let expected_head = format!("__xlog_query_{}", provenance.query_index);
                    let rule = plan
                            .rules_by_scc
                            .get(provenance.scc_index)
                            .and_then(|rules| rules.get(provenance.rule_index))
                            .ok_or_else(|| {
                                XlogError::Execution(format!(
                                    "compiler-generated query provenance {} references missing compiled rule scc={} rule={}",
                                    provenance.query_index,
                                    provenance.scc_index,
                                    provenance.rule_index
                                ))
                            })?;
                    if rule.head != expected_head {
                        return Err(XlogError::Execution(format!(
                                "compiler-generated query provenance {} expects head {expected_head} but references authored head {}",
                                provenance.query_index, rule.head
                            )));
                    }
                    let occurrence_count = plan
                        .rules_by_scc
                        .iter()
                        .flatten()
                        .filter(|candidate| candidate.head == expected_head)
                        .count();
                    if occurrence_count != 1 {
                        return Err(XlogError::Execution(format!(
                                "compiler-generated query head {expected_head} must have exactly one compiled rule, found {occurrence_count}"
                            )));
                    }
                    heads.insert(expected_head);
                }
                Ok(heads)
            }
            _ => Ok((0..self.program.queries.len())
                .map(|index| format!("__xlog_query_{index}"))
                .collect()),
        }
    }

    fn reject_compiler_generated_query_relation_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
        relation_source: &str,
    ) -> Result<()> {
        let generated_query_heads = self.compiler_generated_query_heads()?;
        if let Some(name) = names
            .into_iter()
            .find(|name| generated_query_heads.contains(*name))
        {
            return Err(XlogError::Execution(format!(
                "{relation_source} relation {name} collides with generated query head"
            )));
        }
        Ok(())
    }

    fn evaluate_ordinary_with_resident_mode(
        &self,
        provider: Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
        mode: ResidentSelectionMode,
    ) -> Result<LogicEvalResult> {
        let mut latency_diagnostic =
            resident_latency_diagnostics_enabled().then(ResidentLatencyDiagnostic::new);
        let total_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let certificate_input_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let ordinary_plan = self.ordinary_plan("resident route certification")?;
        if self.program.queries.is_empty() {
            return self.evaluate_existing_gpu_after_resident_decline(
                provider,
                inputs,
                profiling,
                ordinary_plan,
                mode,
                ResidentGraphDeclineReason::FullStoreRequested,
            );
        }
        let certificate_initialization_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let certification = if latency_diagnostic.is_some() {
            self.resident_certified_plan_with_outcome_for_plan(ordinary_plan)
        } else {
            self.resident_certified_plan_for_plan(ordinary_plan)
                .map(|certified| (certified, false, false))
        };
        let (certified_plan, certificate_cache_was_warm, certificate_initialized_here) =
            match certification {
                Ok(outcome) => outcome,
                Err(error) => {
                    return self.evaluate_existing_gpu_after_resident_decline(
                        provider,
                        inputs,
                        profiling,
                        ordinary_plan,
                        mode,
                        ResidentGraphDeclineReason::WorkspaceUnbounded {
                            detail: error.to_string(),
                        },
                    )
                }
            };
        let certificate = certified_plan.certificate();
        if !certificate.is_supported() {
            let reason = certificate.declines().first().cloned().unwrap_or_else(|| {
                ResidentGraphDeclineReason::WorkspaceUnbounded {
                    detail: "route inspection did not produce a resident certificate".into(),
                }
            });
            return self.evaluate_existing_gpu_after_resident_decline(
                provider,
                inputs,
                profiling,
                ordinary_plan,
                mode,
                reason,
            );
        }
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            let elapsed_ns = resident_latency_elapsed_ns(certificate_initialization_started);
            diagnostic.certificate_cache_was_warm = certificate_cache_was_warm;
            diagnostic.certificate_initialized_here = certificate_initialized_here;
            if certificate_initialized_here {
                diagnostic.certificate_initialization_ns = elapsed_ns;
            } else {
                diagnostic.certificate_cache_access_ns = elapsed_ns;
            }
        }
        let input_setup_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());

        for (name, buffer) in &inputs {
            let expected_schema = self.schemas.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Input relation {name} not declared in program schemas"
                ))
            })?;
            ensure_schema_type_compatible(expected_schema, buffer.schema()).map_err(|error| {
                XlogError::Execution(format!("Input relation {name} schema mismatch: {error}"))
            })?;
        }

        if let Some(relation) = inputs.iter().find_map(|(name, buffer)| {
            (!Self::resident_input_is_local(&provider, buffer)).then(|| name.clone())
        }) {
            return self.evaluate_existing_gpu_after_resident_decline(
                provider,
                inputs,
                profiling,
                ordinary_plan,
                mode,
                ResidentGraphDeclineReason::ImportedInputUnsupported { relation },
            );
        }

        for (name, buffer) in &inputs {
            if !buffer.canonical_full_row_set_certified() {
                provider
                    .validated_logical_row_count(buffer)
                    .map_err(|error| {
                        XlogError::Execution(format!(
                            "Input relation {name} has invalid logical row metadata: {error}"
                        ))
                    })?;
            }
        }

        let runtime = Arc::clone(provider.memory().runtime().ok_or_else(|| {
            XlogError::Execution(
                "internal invariant violated: CUDA provider has no owned runtime".into(),
            )
        })?);
        if !Arc::ptr_eq(provider.device(), provider.memory().device())
            || !Arc::ptr_eq(provider.device(), runtime.device())
            || !runtime.supports_block_use_tracking()
        {
            return Err(XlogError::Execution(
                "internal invariant violated: CUDA provider runtime ownership graph is inconsistent"
                    .into(),
            ));
        }
        let resident_provider = provider;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.runtime_bytes[0] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[0] = resident_provider.memory().allocated_bytes();
        }

        let resident_inputs = inputs;

        let mut canonical_replacements = HashMap::new();
        let canonicalization = resident_inputs.iter().try_for_each(|(name, buffer)| {
            let expected_schema = self.schemas.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Input relation {name} not declared in program schemas"
                ))
            })?;
            ensure_schema_type_compatible(expected_schema, buffer.schema()).map_err(|error| {
                XlogError::Execution(format!("Input relation {name} schema mismatch: {error}"))
            })?;
            if buffer.schema() == expected_schema && buffer.canonical_full_row_set_certified() {
                return Ok(());
            }
            let mut normalized = None;
            let canonical_source = if buffer.schema() == expected_schema {
                buffer
            } else {
                let mut clone = resident_provider.clone_buffer(buffer)?;
                clone.set_schema(expected_schema.clone());
                normalized.insert(clone)
            };
            let canonical = resident_provider.union_many_gpu(&[canonical_source])?;
            if !canonical.canonical_full_row_set_certified() {
                return Err(XlogError::Execution(format!(
                    "resident input {name} did not acquire a full-row set proof"
                )));
            }
            canonical_replacements.insert(name.clone(), canonical);
            Ok(())
        });
        if let Err(error) = canonicalization {
            drop(canonical_replacements);
            resident_provider.device().synchronize().map_err(|cleanup| {
                    XlogError::Kernel(format!(
                        "resident input canonicalization failed ({error}); cleanup synchronization failed: {cleanup}"
                    ))
                })?;
            runtime.reap_pending().map_err(|cleanup| {
                    XlogError::Kernel(format!(
                        "resident input canonicalization failed ({error}); cleanup reap failed: {cleanup}"
                    ))
                })?;
            return self.evaluate_existing_gpu_after_resident_decline(
                resident_provider,
                resident_inputs,
                profiling,
                ordinary_plan,
                mode,
                ResidentGraphDeclineReason::WorkspaceUnbounded {
                    detail: format!("input full-row canonicalization failed: {error}"),
                },
            );
        }
        let resident_inputs = resident_inputs
            .into_iter()
            .map(|(name, buffer)| {
                let canonical = canonical_replacements.remove(&name).unwrap_or(buffer);
                (name, canonical)
            })
            .collect();

        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.input_setup_ns = resident_latency_elapsed_ns(input_setup_started);
            diagnostic.certificate_input_ns =
                resident_latency_elapsed_ns(certificate_input_started);
            diagnostic.runtime_bytes[1] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[1] = resident_provider.memory().allocated_bytes();
        }
        let prepare_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let mut executor = self.prepare_resident_executor(
            &resident_provider,
            resident_inputs,
            profiling,
            ordinary_plan,
        )?;
        let prepare_options = latency_diagnostic
            .as_ref()
            .map(|diagnostic| {
                ResidentGraphPrepareOptions::default()
                    .with_latency_diagnostic_sample(diagnostic.sample)
            })
            .unwrap_or_default();
        let mut prepared = match executor
            .prepare_certified_resident_graph(certified_plan.as_ref(), prepare_options)
        {
            Ok(prepared) => prepared,
            Err(ResidentGraphExecutionError::Declined(reason)) => {
                runtime
                    .reap_pending()
                    .map_err(|error| XlogError::Kernel(error.to_string()))?;
                return match mode {
                        ResidentSelectionMode::Auto => {
                            executor.execute_plan(ordinary_plan)?;
                            let mut result = self.finish_ordinary_evaluation(
                                &resident_provider,
                                executor,
                                profiling,
                                None,
                                None,
                            )?;
                            if let Some(stats) = result.stats.as_mut() {
                                stats.resident_graph =
                                    Some(ResidentGraphExecutionStats::declined(reason));
                            }
                            Ok(result)
                        }
                        ResidentSelectionMode::Require => Err(XlogError::Execution(format!(
                            "resident conditional-graph execution was required but declined: {reason:?}"
                        ))),
                        ResidentSelectionMode::Disabled => unreachable!(
                            "disabled resident selection does not call the resident evaluator"
                        ),
                    };
            }
            Err(error) => return Err(Self::resident_execution_error(error)),
        };
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.prepare_capture_allocation_ns = resident_latency_elapsed_ns(prepare_started);
            diagnostic.runtime_bytes[2] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[2] = resident_provider.memory().allocated_bytes();
        }
        let prepare_diagnostic = prepared.take_prepare_diagnostic();

        let transfer_before = resident_provider.host_transfer_stats();
        let provider_dtoh_before = resident_provider.d2h_transfer_count();
        let untracked_dtoh_before = resident_provider.untracked_metadata_dtoh_count();
        let deterministic_d2h_before = resident_provider.deterministic_d2h_violation_count();
        let final_before = resident_provider.final_observation_transfer_stats();
        let graph_before = runtime.conditional_graph_stats();

        let launch_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let in_flight = prepared.launch().map_err(Self::resident_execution_error)?;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.launch_submission_ns = resident_latency_elapsed_ns(launch_started);
            diagnostic.runtime_bytes[3] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[3] = resident_provider.memory().allocated_bytes();
        }
        let sync_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let synchronized = in_flight
            .synchronize_core()
            .map_err(Self::resident_execution_error)?;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.sync_wall_ns = resident_latency_elapsed_ns(sync_started);
            diagnostic.runtime_bytes[4] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[4] = resident_provider.memory().allocated_bytes();
        }

        let transfer_after = resident_provider.host_transfer_stats();
        let provider_dtoh_after = resident_provider.d2h_transfer_count();
        let untracked_dtoh_after = resident_provider.untracked_metadata_dtoh_count();
        let deterministic_d2h_after = resident_provider.deterministic_d2h_violation_count();
        let final_before_observation = resident_provider.final_observation_transfer_stats();
        let graph_after = runtime.conditional_graph_stats();
        let core_transfers = ResidentGraphCoreTransferStats {
            tracked_htod_calls: transfer_after
                .htod_calls
                .saturating_sub(transfer_before.htod_calls),
            tracked_htod_bytes: transfer_after
                .htod_bytes
                .saturating_sub(transfer_before.htod_bytes),
            tracked_dtoh_calls: transfer_after
                .dtoh_calls
                .saturating_sub(transfer_before.dtoh_calls),
            tracked_dtoh_bytes: transfer_after
                .dtoh_bytes
                .saturating_sub(transfer_before.dtoh_bytes),
            provider_dtoh_calls: provider_dtoh_after.saturating_sub(provider_dtoh_before),
            untracked_metadata_dtoh_calls: untracked_dtoh_after
                .saturating_sub(untracked_dtoh_before),
        };
        if core_transfers.tracked_htod_calls != 0
            || core_transfers.tracked_htod_bytes != 0
            || core_transfers.tracked_dtoh_calls != 0
            || core_transfers.tracked_dtoh_bytes != 0
            || core_transfers.provider_dtoh_calls != 0
            || core_transfers.untracked_metadata_dtoh_calls != 0
            || final_before_observation.dtoh_calls != final_before.dtoh_calls
            || final_before_observation.dtoh_bytes != final_before.dtoh_bytes
            || final_before_observation.pinned_receipts != final_before.pinned_receipts
        {
            return Err(XlogError::Execution(
                "resident conditional-graph core performed a host transfer".into(),
            ));
        }
        let graph_launches = graph_after.launches.saturating_sub(graph_before.launches);
        let terminal_synchronizations = graph_after
            .terminal_synchronizations
            .saturating_sub(graph_before.terminal_synchronizations);
        let host_iterations = graph_after
            .host_iterations
            .saturating_sub(graph_before.host_iterations);
        let host_allocations = graph_after
            .host_allocations
            .saturating_sub(graph_before.host_allocations);
        let host_status_injections = graph_after
            .host_status_injections
            .saturating_sub(graph_before.host_status_injections);
        let deterministic_d2h_violations =
            deterministic_d2h_after.saturating_sub(deterministic_d2h_before);
        if graph_launches != 1
            || terminal_synchronizations != 1
            || host_iterations != 0
            || host_allocations != 0
            || host_status_injections != 0
            || deterministic_d2h_violations != 0
        {
            return Err(XlogError::Execution(format!(
                    "resident conditional-graph runtime invariant failed: launches={graph_launches}, terminal_synchronizations={terminal_synchronizations}, host_iterations={host_iterations}, host_allocations={host_allocations}, host_status_injections={host_status_injections}, deterministic_d2h_violations={deterministic_d2h_violations}"
                )));
        }

        let observation_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let observed = synchronized
            .observe_final_receipt()
            .map_err(Self::resident_execution_error)?;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            let observation_ns = resident_latency_elapsed_ns(observation_started);
            let phase = observed.phase_timings().ok_or_else(|| {
                XlogError::Execution(
                    "resident latency diagnostics missing final-observation timings".into(),
                )
            })?;
            diagnostic.receipt_d2h_ns = phase.receipt_d2h_ns;
            diagnostic.receipt_decode_schema_staging_ns = phase.decode_schema_staging_ns;
            diagnostic.owner_teardown_residual_ns = observation_ns
                .saturating_sub(phase.receipt_d2h_ns)
                .saturating_sub(phase.decode_schema_staging_ns);
            diagnostic.staged_outputs = observed.staged_output_count();
            diagnostic.relation_registrations = observed.relation_registration_count();
            diagnostic.runtime_bytes[5] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[5] = resident_provider.memory().allocated_bytes();
        }
        let encoded_len = u64::try_from(observed.encoded_len())
            .map_err(|_| XlogError::Execution("resident receipt byte length exceeds u64".into()))?;
        let device_elapsed_ns = observed.device_elapsed_ns();
        let device_scan_invocations = observed.device_scan_invocations();
        let device_filter_invocations = observed.device_filter_invocations();
        let semantic_scan_invocations = observed.semantic_scan_invocations();
        let semantic_filter_invocations = observed.semantic_filter_invocations();
        let staged_store_mutations = observed.staged_output_count();
        let iterations = observed.iterations();
        let final_after = resident_provider.final_observation_transfer_stats();
        let final_observation = ResidentGraphFinalObservationStats {
            dtoh_calls: final_after
                .dtoh_calls
                .saturating_sub(final_before_observation.dtoh_calls),
            dtoh_bytes: final_after
                .dtoh_bytes
                .saturating_sub(final_before_observation.dtoh_bytes),
            pinned_receipts: final_after
                .pinned_receipts
                .saturating_sub(final_before_observation.pinned_receipts),
        };
        if final_observation.dtoh_calls != 1
            || final_observation.dtoh_bytes != encoded_len
            || final_observation.pinned_receipts != 1
        {
            return Err(XlogError::Execution(format!(
                    "resident final observation invariant failed: calls={}, bytes={}, pinned={} expected_bytes={encoded_len}",
                    final_observation.dtoh_calls,
                    final_observation.dtoh_bytes,
                    final_observation.pinned_receipts,
                )));
        }
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.device_event_ns = device_elapsed_ns;
        }
        let commit_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        observed
            .commit(&mut executor)
            .map_err(Self::resident_execution_error)?;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.commit_ns = resident_latency_elapsed_ns(commit_started);
            diagnostic.runtime_bytes[6] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[6] = resident_provider.memory().allocated_bytes();
        }

        let telemetry_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        let timed_scan_filter_invocations = device_scan_invocations
            .checked_add(device_filter_invocations)
            .ok_or_else(|| {
                XlogError::Execution("resident device invocation count overflow".into())
            })?;
        let telemetry = ResidentGraphExecutionStats {
            selection: ResidentGraphSelectionKind::ResidentConditionalGraph,
            decline: None,
            conditional_graph_launches: graph_launches,
            terminal_synchronizations,
            host_iterations,
            host_allocations,
            host_status_injections,
            deterministic_d2h_violations,
            host_dispatched_scan_ops: 0,
            host_dispatched_filter_ops: 0,
            device_scan_invocations,
            device_filter_invocations,
            semantic_scan_invocations,
            semantic_filter_invocations,
            staged_store_mutations,
            deferred_profile: ResidentGraphDeferredProfile {
                timed_scan_filter_invocations,
                device_elapsed_ns,
                final_sync_misattributed_ns: 0,
            },
            core_transfers,
            final_observation,
        };
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.result_stats_construction_ns =
                resident_latency_elapsed_ns(telemetry_started);
        }
        let result = self.finish_ordinary_evaluation(
            &resident_provider,
            executor,
            profiling,
            Some(ResidentCompletedProfile {
                telemetry,
                iterations,
            }),
            latency_diagnostic.as_mut(),
        )?;
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.runtime_bytes[7] = runtime.bytes_outstanding();
            diagnostic.manager_bytes[7] = resident_provider.memory().allocated_bytes();
        }
        let diagnostic_lines = if latency_diagnostic.is_some() {
            let total_ns = resident_latency_elapsed_ns(total_started);
            finalized_resident_latency_diagnostic_lines(
                total_ns,
                latency_diagnostic.as_ref(),
                prepare_diagnostic
                    .map(|diagnostic| move || diagnostic.into_snapshot().format_line()),
                ResidentLatencyDiagnostic::format_line,
            )
        } else {
            None
        };
        if let Some(diagnostic_lines) = diagnostic_lines {
            for line in diagnostic_lines.into_iter().flatten() {
                eprintln!("{line}");
            }
        }
        Ok(result)
    }

    fn evaluate_existing_gpu_after_resident_decline(
        &self,
        provider: Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
        plan: &ExecutionPlan,
        mode: ResidentSelectionMode,
        reason: ResidentGraphDeclineReason,
    ) -> Result<LogicEvalResult> {
        match mode {
            ResidentSelectionMode::Require => Err(XlogError::Execution(format!(
                "resident conditional-graph execution was required but declined: {reason:?}"
            ))),
            ResidentSelectionMode::Auto | ResidentSelectionMode::Disabled => {
                let mut executor = self.prepare_executor(&provider, inputs, profiling)?;
                executor.execute_plan(plan)?;
                let mut result =
                    self.finish_ordinary_evaluation(&provider, executor, profiling, None, None)?;
                if let Some(stats) = result.stats.as_mut() {
                    stats.resident_graph = Some(ResidentGraphExecutionStats::declined(reason));
                }
                Ok(result)
            }
        }
    }

    fn resident_input_is_local(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> bool {
        let expected_manager = Arc::as_ptr(provider.memory()) as usize;
        buffer.num_rows_device().memory_manager_ptr_value() == expected_manager
            && buffer.columns().iter().all(|column| {
                matches!(
                    column,
                    CudaColumn::Owned(slice)
                        if slice.memory_manager_ptr_value() == expected_manager
                )
            })
    }

    fn resident_execution_error(error: ResidentGraphExecutionError) -> XlogError {
        XlogError::Execution(error.to_string())
    }

    fn finish_ordinary_evaluation(
        &self,
        provider: &Arc<CudaKernelProvider>,
        mut executor: Executor,
        profiling: bool,
        resident_profile: Option<ResidentCompletedProfile>,
        mut latency_diagnostic: Option<&mut ResidentLatencyDiagnostic>,
    ) -> Result<LogicEvalResult> {
        let result_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        self.enforce_constraints(provider, &executor)?;

        let mut queries = Vec::with_capacity(self.program.queries.len());
        for (index, query) in self.program.queries.iter().enumerate() {
            let internal_relation_name = format!("__xlog_query_{index}");
            let buffer = executor
                .store_mut()
                .remove(&internal_relation_name)
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "Missing query result relation {internal_relation_name} (compiler bug?)"
                    ))
                })?;
            queries.push(self.logic_query_result(
                provider.as_ref(),
                index,
                query,
                internal_relation_name,
                buffer,
            )?);
        }

        let total_output_rows = queries
            .iter()
            .map(|query| {
                query
                    .buffer
                    .cached_row_count()
                    .map(u64::from)
                    .unwrap_or_else(|| query.buffer.num_rows())
            })
            .sum();
        let mut stats = profiling.then(|| executor.execution_stats(total_output_rows));
        if let (Some(stats), Some(profile)) = (stats.as_mut(), resident_profile) {
            let scan_count =
                usize::try_from(profile.telemetry.device_scan_invocations).map_err(|_| {
                    XlogError::Execution("resident scan profile count exceeds usize".into())
                })?;
            let filter_count = usize::try_from(profile.telemetry.device_filter_invocations)
                .map_err(|_| {
                    XlogError::Execution("resident filter profile count exceeds usize".into())
                })?;
            let (num_rules, is_recursive) = match &self.plan {
                LogicExecutionPlan::Ordinary(plan) => (
                    plan.rules_by_scc.iter().map(Vec::len).sum(),
                    plan.sccs.iter().any(|scc| scc.is_recursive),
                ),
                _ => (0, false),
            };
            let mut stratum = StratumStats::new(0, num_rules, is_recursive);
            stratum.iterations = profile.iterations as usize;
            stratum.duration_us = profile.telemetry.deferred_profile.device_elapsed_ns / 1_000;
            stratum.ops.reserve(scan_count.saturating_add(filter_count));
            stratum.ops.extend((0..scan_count).map(|_| OpStats {
                op_name: "scan".to_string(),
                ..OpStats::default()
            }));
            stratum.ops.extend((0..filter_count).map(|_| OpStats {
                op_name: "filter".to_string(),
                ..OpStats::default()
            }));
            stats.total_duration_us = stratum.duration_us;
            stats.strata = vec![stratum];
            stats.resident_graph = Some(profile.telemetry);
        }

        let result = LogicEvalResult { queries, stats };
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.result_stats_construction_ns = diagnostic
                .result_stats_construction_ns
                .saturating_add(resident_latency_elapsed_ns(result_started));
            diagnostic.remaining_store_relations_before_drop = executor.store().len();
        }
        let executor_drop_started = latency_diagnostic
            .as_ref()
            .map(|_| std::time::Instant::now());
        drop(executor);
        if let Some(diagnostic) = latency_diagnostic.as_mut() {
            diagnostic.executor_store_teardown_ns =
                resident_latency_elapsed_ns(executor_drop_started);
        }
        Ok(result)
    }

    fn prepare_resident_executor(
        &self,
        provider: &Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
        plan: &ExecutionPlan,
    ) -> Result<Executor> {
        let derived_relations = plan
            .rules_by_scc
            .iter()
            .flatten()
            .map(|rule| rule.head.clone())
            .collect::<BTreeSet<_>>();
        self.prepare_executor_excluding_derived_placeholders(
            provider,
            inputs,
            profiling,
            Some(&derived_relations),
        )
    }

    fn prepare_executor_excluding_derived_placeholders(
        &self,
        provider: &Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
        derived_relations: Option<&BTreeSet<String>>,
    ) -> Result<Executor> {
        let mut executor = Executor::new(provider.clone());
        executor.set_profiling(profiling);
        for (name, rel_id) in &self.rel_ids {
            executor.register_relation(*rel_id, name);
        }

        let arity_qualified_predicates = if self.epistemic_provenance.is_some() {
            epistemic_extensional_multi_arity_predicates(&self.program)
        } else {
            predicate_arities(&self.program)
                .into_iter()
                .filter_map(|(predicate, arities)| (arities.len() > 1).then_some(predicate))
                .collect()
        };
        let inline_fact_relations = self
            .program
            .facts()
            .map(|fact| {
                let predicate = fact.head.predicate.as_str();
                if arity_qualified_predicates.contains(predicate) {
                    arity_qualified_name(predicate, fact.head.terms.len())
                } else {
                    predicate.to_string()
                }
            })
            .collect::<BTreeSet<_>>();

        for (name, schema) in &self.schemas {
            let is_derived_placeholder = derived_relations.is_some_and(|set| set.contains(name))
                && !inline_fact_relations.contains(name);
            if is_derived_placeholder {
                continue;
            }
            executor
                .store_mut()
                .put(name, provider.create_empty_buffer(schema.clone())?);
        }

        for (name, buffer) in inputs {
            let schema = self.schemas.get(&name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Input relation {} not declared in program schemas",
                    name
                ))
            })?;
            ensure_schema_type_compatible(schema, buffer.schema()).map_err(|e| {
                XlogError::Execution(format!("Input relation {} schema mismatch: {}", name, e))
            })?;
            executor.store_mut().put(&name, buffer);
        }

        self.load_facts(provider, &mut executor)?;
        Ok(executor)
    }

    fn executor_from_materialized_store(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
        profiling: bool,
    ) -> Result<Executor> {
        let mut executor = Executor::new(provider.clone());
        executor.set_profiling(profiling);
        for (name, rel_id) in &self.rel_ids {
            executor.register_relation(*rel_id, name);
        }

        for (name, schema) in &self.schemas {
            executor
                .store_mut()
                .put(name, provider.create_empty_buffer(schema.clone())?);
        }

        for name in relation_store.names() {
            let buffer = relation_store.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Persistent relation {} disappeared during evaluation",
                    name
                ))
            })?;
            let schema = self.schemas.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Persistent relation {} not declared in program schemas",
                    name
                ))
            })?;
            ensure_schema_type_compatible(schema, buffer.schema()).map_err(|e| {
                XlogError::Execution(format!(
                    "Persistent relation {} schema mismatch: {}",
                    name, e
                ))
            })?;
            executor
                .store_mut()
                .put(name, provider.clone_buffer(buffer)?);
        }

        Ok(executor)
    }

    #[cfg(test)]
    fn resident_certified_plan(&self) -> Result<Arc<ResidentGraphCertifiedPlan>> {
        let plan = self.ordinary_plan("resident route certification")?;
        self.resident_certified_plan_for_plan(plan)
    }

    #[cfg(test)]
    fn resident_certified_plan_with_outcome(
        &self,
    ) -> Result<(Arc<ResidentGraphCertifiedPlan>, bool, bool)> {
        let plan = self.ordinary_plan("resident route certification")?;
        self.resident_certified_plan_with_outcome_for_plan(plan)
    }

    fn resident_certified_plan_with_outcome_for_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<(Arc<ResidentGraphCertifiedPlan>, bool, bool)> {
        self.reusable_state_identity
            .get_or_init_resident_certification_with_outcome(|| self.inspect_resident_plan(plan))
    }

    fn resident_certified_plan_for_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<Arc<ResidentGraphCertifiedPlan>> {
        self.reusable_state_identity
            .get_or_init_resident_certification(|| self.inspect_resident_plan(plan))
    }

    fn inspect_resident_plan(&self, plan: &ExecutionPlan) -> Result<ResidentGraphCertifiedPlan> {
        let resident_plan = self.resident_dependency_closed_plan(plan);
        let catalog = ResidentGraphSchemaCatalog::from_named_schemas(
            self.rel_ids.iter().filter_map(|(name, relation)| {
                self.schemas
                    .get(name)
                    .cloned()
                    .map(|schema| (name.clone(), *relation, schema))
            }),
        );
        ResidentGraphCertifiedPlan::inspect(Arc::new(resident_plan), &catalog)
    }

    fn resident_dependency_closed_plan(&self, plan: &ExecutionPlan) -> ExecutionPlan {
        self.try_resident_dependency_closed_plan(plan)
            .unwrap_or_else(|| plan.clone())
    }

    fn try_resident_dependency_closed_plan(&self, plan: &ExecutionPlan) -> Option<ExecutionPlan> {
        if self.program.queries.is_empty()
            || plan.generated_query_rules.len() != self.program.queries.len()
        {
            return None;
        }

        let mut roots =
            Vec::with_capacity(plan.generated_query_rules.len() + self.program.constraints.len());
        let mut root_heads = std::collections::HashSet::with_capacity(roots.capacity());
        let mut seen_queries = vec![false; self.program.queries.len()];
        for query in &plan.generated_query_rules {
            let expected_head = format!("__xlog_query_{}", query.query_index);
            let rule = plan
                .rules_by_scc
                .get(query.scc_index)?
                .get(query.rule_index)?;
            let seen = seen_queries.get_mut(query.query_index)?;
            if *seen || rule.head != expected_head {
                return None;
            }
            let occurrences = plan
                .rules_by_scc
                .iter()
                .flatten()
                .filter(|candidate| candidate.head == expected_head)
                .count();
            if occurrences != 1 {
                return None;
            }
            *seen = true;
            roots.push(query.scc_index);
            if !root_heads.insert(expected_head) {
                return None;
            }
        }
        if seen_queries.iter().any(|seen| !seen) {
            return None;
        }

        for constraint_index in 0..self.program.constraints.len() {
            let expected_head = format!("__xlog_constraint_{constraint_index}");
            let positions = plan
                .rules_by_scc
                .iter()
                .enumerate()
                .flat_map(|(scc_index, rules)| {
                    rules
                        .iter()
                        .filter(|rule| rule.head == expected_head)
                        .map(move |_| scc_index)
                })
                .collect::<Vec<_>>();
            let [scc_index] = positions.as_slice() else {
                return None;
            };
            roots.push(*scc_index);
            if !root_heads.insert(expected_head) {
                return None;
            }
        }

        let mut defining_sccs = HashMap::new();
        for (scc_index, rules) in plan.rules_by_scc.iter().enumerate() {
            for rule in rules {
                let Some(relation) = self.rel_ids.get(&rule.head).copied() else {
                    if root_heads.contains(&rule.head) {
                        continue;
                    }
                    return None;
                };
                match defining_sccs.insert(relation, scc_index) {
                    Some(previous) if previous != scc_index => return None,
                    _ => {}
                }
            }
        }

        plan.dependency_closed_subplan(&roots, &defining_sccs)
    }

    #[cfg(test)]
    fn resident_certification_initializations(&self) -> u64 {
        self.reusable_state_identity
            .resident_certification_initializations()
    }

    /// Evaluate the program with optional profiling
    ///
    /// # Arguments
    /// * `provider` - The CUDA kernel provider
    /// * `inputs` - Input relations
    /// * `profiling` - Whether to collect execution statistics
    pub fn evaluate_with_options(
        &self,
        provider: Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        self.reject_compiler_generated_query_relation_names(
            inputs.keys().map(String::as_str),
            "caller input",
        )?;
        let resident_mode = ResidentSelectionMode::from_env()?;
        if matches!(&self.plan, LogicExecutionPlan::Ordinary(_)) && resident_mode.enabled() {
            return self.evaluate_ordinary_with_resident_mode(
                provider,
                inputs,
                profiling,
                resident_mode,
            );
        }
        let mut executor = self.prepare_executor(&provider, inputs, profiling)?;

        if let LogicExecutionPlan::EpistemicG91Compatibility(g91_plan) = &self.plan {
            let result = self
                .evaluate_g91_compatibility_gpu_program(provider, executor, g91_plan, profiling)?;
            return self.finish_nonordinary_resident_selection(result, resident_mode);
        }

        if let LogicExecutionPlan::EpistemicWfsGpu(wfs_plan) = &self.plan {
            let result = self.evaluate_wfs_gpu_program(provider, executor, wfs_plan, profiling)?;
            return self.finish_nonordinary_resident_selection(result, resident_mode);
        }

        let LogicExecutionPlan::Ordinary(plan) = &self.plan else {
            let result = self.evaluate_epistemic_with_executor(&provider, executor, profiling)?;
            return self.finish_nonordinary_resident_selection(result, resident_mode);
        };

        executor.execute_plan(plan)?;

        self.enforce_constraints(&provider, &executor)?;

        let mut queries: Vec<LogicQueryResult> = Vec::with_capacity(self.program.queries.len());
        for (i, query) in self.program.queries.iter().enumerate() {
            let internal_relation_name = format!("__xlog_query_{}", i);
            let buffer = executor
                .store_mut()
                .remove(&internal_relation_name)
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "Missing query result relation {} (compiler bug?)",
                        internal_relation_name
                    ))
                })?;

            queries.push(self.logic_query_result(
                provider.as_ref(),
                i,
                query,
                internal_relation_name,
                buffer,
            )?);
        }

        // Collect execution stats if profiling was enabled
        let total_output_rows: u64 = queries.iter().map(|q| q.buffer.num_rows()).sum();
        let stats = if profiling {
            Some(executor.execution_stats(total_output_rows))
        } else {
            None
        };

        Ok(LogicEvalResult { queries, stats })
    }

    /// Build an executor seeded with declared schemas, caller inputs and program facts.
    ///
    /// Shared by ordinary evaluation and by the epistemic evidence handoff so both
    /// paths seed relations identically.
    fn prepare_executor(
        &self,
        provider: &Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
        profiling: bool,
    ) -> Result<Executor> {
        self.prepare_executor_excluding_derived_placeholders(provider, inputs, profiling, None)
    }

    /// Execute an epistemic program and return its accepted GPU execution evidence.
    ///
    /// `evaluate` reduces the epistemic result to query rows and drops the accepted
    /// world-view evidence with it. The probabilistic production adapter needs that
    /// evidence itself, so this entry point keeps the raw execution result.
    ///
    /// Only single-component epistemic plans are supported: split, stratified and WFS
    /// plans produce several world views whose probabilistic conditioning contract is
    /// not settled yet, so they are rejected loudly rather than silently reduced.
    ///
    /// "Ordinary" here names a plan kind, not the source. An admissible recursive modal
    /// program (`reach(X, Z) :- reach(X, Y), know link(Y, Z).`) is lowered by the Case-A
    /// reduction to ordinary recursion: the world-view machinery is erased, so no
    /// accepted world view survives to hand over and the program is rejected here
    /// despite being full of `know`. The rejection names the `epistemic_provenance`
    /// reduction class so the message explains the lowering instead of reading as a
    /// misclassification.
    pub fn execute_epistemic_evidence(
        &self,
        provider: Arc<CudaKernelProvider>,
        inputs: HashMap<String, CudaBuffer>,
    ) -> Result<EpistemicGpuExecutionResult> {
        let LogicExecutionPlan::EpistemicSingle(executable) = &self.plan else {
            let reduction = self
                .epistemic_provenance
                .as_ref()
                .map(|provenance| provenance.reduction)
                .unwrap_or("none");
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic accepted-evidence handoff".to_string(),
                context: format!(
                    "execute_epistemic_evidence requires a single-component epistemic plan; \
                     ordinary, split, stratified, recursive G91-compatibility and WFS plans \
                     are not supported (epistemic_provenance reduction: {reduction}). A \
                     recursive modal program reduced to ordinary recursion \
                     (ordinary_recursive_modal_reduction) is rejected here by design: the \
                     reduction erases the world-view machinery, so no accepted world view \
                     survives to condition on"
                ),
            });
        };

        let mut executor = self.prepare_executor(&provider, inputs, false)?;
        let result = executor
            .execute_epistemic_gpu_execution(
                executable,
                capacities_for_epistemic_executable(executable)?,
            )
            .map_err(|error| self.present_epistemic_constraint_violation(error))?;
        result.require_runtime_dispatch_certification()?;
        Ok(result)
    }

    /// Compare query result relations between two stores using GPU set difference.
    pub fn relation_stores_query_equivalent(
        &self,
        provider: &CudaKernelProvider,
        left: &RelationStore,
        right: &RelationStore,
    ) -> Result<bool> {
        for idx in 0..self.program.queries.len() {
            let name = format!("__xlog_query_{}", idx);
            let Some(left_buffer) = left.get(&name) else {
                return Ok(false);
            };
            let Some(right_buffer) = right.get(&name) else {
                return Ok(false);
            };
            if !buffers_gpu_set_equivalent(provider, left_buffer, right_buffer)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn clone_relation_store(
        &self,
        provider: &Arc<CudaKernelProvider>,
        source: &RelationStore,
    ) -> Result<RelationStore> {
        let mut cloned = RelationStore::new(provider.clone());
        for name in source.names() {
            let buffer = source.get(name).ok_or_else(|| {
                XlogError::Execution(format!("Relation {} disappeared during clone", name))
            })?;
            cloned.put(name, provider.clone_buffer(buffer)?);
        }
        Ok(cloned)
    }

    fn clone_prepared_relation_snapshot(
        &self,
        provider: &Arc<CudaKernelProvider>,
        source: &RelationStore,
    ) -> Result<RelationStore> {
        let mut relation_names = source.names().collect::<Vec<_>>();
        relation_names.sort_unstable();
        let mut cloned = RelationStore::new(provider.clone());
        cloned.try_reserve_relations(relation_names.len())?;
        for name in relation_names {
            let buffer = source.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Relation {name} disappeared while cloning prepared snapshot"
                ))
            })?;
            let context = format!("cloning prospective relation snapshot '{name}'");
            let cloned_buffer = provider
                .clone_buffer(buffer)
                .map_err(|error| relation_clone_error(context, error))?;
            cloned.put(name, cloned_buffer);
        }
        Ok(cloned)
    }

    fn total_query_rows(&self, store: &RelationStore) -> Result<u64> {
        let mut total = 0;
        for i in 0..self.program.queries.len() {
            let relation_name = format!("__xlog_query_{}", i);
            let buffer = store.get(&relation_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing query result relation {} (compiler bug?)",
                    relation_name
                ))
            })?;
            total += buffer.num_rows();
        }
        Ok(total)
    }

    fn logic_result_from_store(
        &self,
        provider: &CudaKernelProvider,
        store: &RelationStore,
        stats: Option<ExecutionStats>,
    ) -> Result<LogicEvalResult> {
        let mut queries: Vec<LogicQueryResult> = Vec::with_capacity(self.program.queries.len());
        for (i, query) in self.program.queries.iter().enumerate() {
            let relation_name = format!("__xlog_query_{}", i);
            let buffer = store.get(&relation_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing query result relation {} (compiler bug?)",
                    relation_name
                ))
            })?;

            queries.push(self.logic_query_result(
                provider,
                i,
                query,
                relation_name,
                provider.clone_buffer(buffer)?,
            )?);
        }

        Ok(LogicEvalResult { queries, stats })
    }

    fn logic_query_result(
        &self,
        provider: &CudaKernelProvider,
        query_index: usize,
        query: &Query,
        internal_relation_name: String,
        buffer: CudaBuffer,
    ) -> Result<LogicQueryResult> {
        let provenance = self.epistemic_provenance.as_ref();
        let surface_source_query = provenance.is_some_and(|value| value.surface_source_queries);
        let presentation_query = if surface_source_query {
            self.source_program
                .queries
                .get(query_index)
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "missing authored metadata for query {query_index}"
                    ))
                })?
        } else {
            query
        };
        let columns = query_output_vars(presentation_query);
        let buffer = if columns.is_empty() {
            let row_count = provider.device_row_count(&buffer)?;
            let row_count = u32::try_from(row_count).map_err(|_| {
                XlogError::Execution(format!(
                    "query result row count {row_count} exceeds the GPU row-count range"
                ))
            })?;
            provider.create_zero_arity_buffer(Schema::new(Vec::new()), row_count)?
        } else {
            buffer
        };
        let relation_name = if surface_source_query {
            presentation_query.atom.predicate.clone()
        } else {
            internal_relation_name
        };

        Ok(LogicQueryResult {
            relation_name,
            sort_labels: columns.clone(),
            columns,
            buffer,
        })
    }

    fn load_facts(&self, provider: &CudaKernelProvider, executor: &mut Executor) -> Result<()> {
        self.load_facts_into_store(provider, executor.store_mut())
    }

    fn load_facts_into_store(
        &self,
        provider: &CudaKernelProvider,
        store: &mut RelationStore,
    ) -> Result<()> {
        let arity_qualified_predicates = if self.epistemic_provenance.is_some() {
            epistemic_extensional_multi_arity_predicates(&self.program)
        } else {
            predicate_arities(&self.program)
                .into_iter()
                .filter_map(|(predicate, arities)| (arities.len() > 1).then_some(predicate))
                .collect()
        };
        let mut rows_by_pred: HashMap<String, Vec<&[Term]>> = HashMap::new();
        for fact in self.program.facts() {
            let pred = fact.head.predicate.as_str();
            let arity = fact.head.terms.len();
            let key = if arity_qualified_predicates.contains(pred) {
                arity_qualified_name(pred, arity)
            } else {
                pred.to_string()
            };
            rows_by_pred.entry(key).or_default().push(&fact.head.terms);
        }

        for (pred, rows) in rows_by_pred {
            let schema = self.schemas.get(pred.as_str()).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing inferred schema for fact predicate {}",
                    pred
                ))
            })?;

            if rows.iter().any(|r| r.len() != schema.arity()) {
                return Err(XlogError::Execution(format!(
                    "Fact arity mismatch for {} (expected {} columns)",
                    pred,
                    schema.arity()
                )));
            }

            let mut columns: Vec<Vec<u8>> = vec![Vec::new(); schema.arity()];
            for row in rows {
                for (col_idx, term) in row.iter().enumerate() {
                    let typ = schema.column_type(col_idx).ok_or_else(|| {
                        XlogError::Execution(format!("Missing type for column {}", col_idx))
                    })?;
                    append_ground_term_bytes(&mut columns[col_idx], term, typ).map_err(|error| {
                        XlogError::Execution(format!(
                            "Failed to encode fact for predicate {pred} at column {col_idx}: {error}"
                        ))
                    })?;
                }
            }

            let fact_buf = if schema.arity() == 0 {
                // Nullary predicate: every `pred().` assertion denotes the same unit
                // tuple `()`, so presence is a single row. `create_buffer_from_slices`
                // with no column slices yields a 0-row (absent) relation, which would
                // make an asserted nullary fact read as false everywhere downstream
                // (ordinary joins and epistemic modal membership alike).
                provider.create_zero_arity_buffer(schema.clone(), 1)?
            } else {
                let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
                provider.create_buffer_from_slices(&slices, schema.clone())?
            };

            let existing = store.get(&pred).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing base relation {} while loading facts",
                    pred
                ))
            })?;

            let merged = provider.union(existing, &fact_buf)?;
            store.put(pred.as_str(), merged);
        }

        Ok(())
    }

    fn evaluate_wfs_gpu_program(
        &self,
        provider: Arc<CudaKernelProvider>,
        base_executor: Executor,
        wfs: &EpistemicWfsGpuPlan,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        let base_store = self.clone_relation_store(&provider, base_executor.store())?;
        let mut stats = profiling.then(ExecutionStats::default);
        let lower_store =
            self.run_wfs_gpu_fixpoint(&provider, &base_store, wfs, profiling, &mut stats)?;
        self.enforce_constraints_in_store(provider.as_ref(), &lower_store)?;
        let total_output_rows = self.total_query_rows(&lower_store)?;
        finalize_iterative_execution_stats(&mut stats, total_output_rows);
        self.logic_result_from_store(provider.as_ref(), &lower_store, stats)
    }

    fn run_wfs_gpu_fixpoint(
        &self,
        provider: &Arc<CudaKernelProvider>,
        base_store: &RelationStore,
        wfs: &EpistemicWfsGpuPlan,
        profiling: bool,
        stats: &mut Option<ExecutionStats>,
    ) -> Result<RelationStore> {
        let upper_executor =
            self.run_gpu_ordinary_pass(provider, &wfs.overapprox, base_store, &[], profiling)?;
        collect_iterative_execution_stats(stats, &upper_executor);
        let mut upper_store = self.clone_relation_store(provider, upper_executor.store())?;
        let mut lower_store = self.clone_relation_store(provider, base_store)?;

        for _ in 0..wfs.max_iterations {
            let upper_fixed: Vec<_> = wfs
                .upper_fixed_names
                .iter()
                .map(|(source, fixed)| (source.as_str(), fixed.as_str(), &upper_store))
                .collect();
            let lower_executor = self.run_gpu_ordinary_pass(
                provider,
                &wfs.lower,
                base_store,
                &upper_fixed,
                profiling,
            )?;
            collect_iterative_execution_stats(stats, &lower_executor);
            let next_lower = self.clone_relation_store(provider, lower_executor.store())?;

            let lower_fixed: Vec<_> = wfs
                .lower_fixed_names
                .iter()
                .map(|(source, fixed)| (source.as_str(), fixed.as_str(), &next_lower))
                .collect();
            let next_upper_executor = self.run_gpu_ordinary_pass(
                provider,
                &wfs.upper,
                base_store,
                &lower_fixed,
                profiling,
            )?;
            collect_iterative_execution_stats(stats, &next_upper_executor);
            let next_upper = self.clone_relation_store(provider, next_upper_executor.store())?;

            let lower_converged =
                self.wfs_gpu_stores_equivalent(provider, wfs, &lower_store, &next_lower)?;
            let upper_converged =
                self.wfs_gpu_stores_equivalent(provider, wfs, &upper_store, &next_upper)?;
            lower_store = next_lower;
            upper_store = next_upper;
            if lower_converged && upper_converged {
                return Ok(lower_store);
            }
        }

        Err(XlogError::Execution(format!(
            "GPU-backed WFS did not converge within {} alternating-fixpoint iterations; raise \
             #pragma max_recursion_depth only when the finite relation domain requires it",
            wfs.max_iterations
        )))
    }

    fn evaluate_g91_compatibility_gpu_program(
        &self,
        provider: Arc<CudaKernelProvider>,
        base_executor: Executor,
        g91: &EpistemicG91CompatibilityGpuPlan,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        let base_store = self.clone_relation_store(&provider, base_executor.store())?;
        let mut stats = profiling.then(ExecutionStats::default);
        let mut current_store = self.run_gpu_evaluation_pass(
            &provider,
            &g91.upper_bound,
            &base_store,
            &[],
            profiling,
            &mut stats,
        )?;
        let refinement_schemas = gpu_evaluation_pass_schemas(&g91.refinement);

        for _ in 0..g91.max_iterations {
            let snapshots = g91
                .snapshot_relations
                .iter()
                .map(|(source, snapshot)| (source.as_str(), snapshot.as_str(), &current_store))
                .collect::<Vec<_>>();
            let next_store = self.run_gpu_evaluation_pass(
                &provider,
                &g91.refinement,
                &base_store,
                &snapshots,
                profiling,
                &mut stats,
            )?;
            let converged = self.gpu_stores_equivalent(
                &provider,
                &refinement_schemas,
                &g91.convergence_predicates,
                &current_store,
                &next_store,
            )?;
            if converged {
                self.enforce_constraints_in_store(&provider, &next_store)?;
                let total_output_rows = self.total_query_rows(&next_store)?;
                finalize_iterative_execution_stats(&mut stats, total_output_rows);
                return self.logic_result_from_store(provider.as_ref(), &next_store, stats);
            }
            current_store = next_store;
        }

        Err(XlogError::Execution(format!(
            "Gelfond-1991 tuple compatibility did not converge within {} refinement iterations; \
             raise #pragma max_recursion_depth only when the finite relation domain requires it",
            g91.max_iterations
        )))
    }

    fn run_gpu_ordinary_pass(
        &self,
        provider: &Arc<CudaKernelProvider>,
        pass: &GpuOrdinaryPass,
        base_store: &RelationStore,
        fixed_relations: &[(&str, &str, &RelationStore)],
        profiling: bool,
    ) -> Result<Executor> {
        let mut executor = Executor::new(provider.clone());
        executor.set_profiling(profiling);
        for (name, rel_id) in &pass.rel_ids {
            executor.register_relation(*rel_id, name);
        }
        for (name, schema) in &pass.schemas {
            executor
                .store_mut()
                .put(name, provider.create_empty_buffer(schema.clone())?);
        }
        for name in base_store.names() {
            // Query and constraint relations are outputs of this compiled pass.
            // Seeding them from an earlier epistemic stratum would union stale,
            // ungated candidates into the authoritative recomputation.
            if pass.schemas.contains_key(name)
                && !name.starts_with("__xlog_query_")
                && !name.starts_with("__xlog_constraint_")
            {
                let buffer = base_store.get(name).ok_or_else(|| {
                    XlogError::Execution(format!("WFS base relation {name} disappeared"))
                })?;
                executor
                    .store_mut()
                    .put(name, provider.clone_buffer(buffer)?);
            }
        }
        for &(source, fixed, source_store) in fixed_relations {
            let buffer =
                self.gpu_clone_or_empty(provider, &pass.schemas, source, fixed, source_store)?;
            executor.store_mut().put(fixed, buffer);
        }
        executor.execute_plan(&pass.plan)?;
        Ok(executor)
    }

    fn run_gpu_evaluation_pass(
        &self,
        provider: &Arc<CudaKernelProvider>,
        pass: &GpuEvaluationPass,
        base_store: &RelationStore,
        fixed_relations: &[(&str, &str, &RelationStore)],
        profiling: bool,
        stats: &mut Option<ExecutionStats>,
    ) -> Result<RelationStore> {
        match pass {
            GpuEvaluationPass::Ordinary(ordinary) => {
                let executor = self.run_gpu_ordinary_pass(
                    provider,
                    ordinary,
                    base_store,
                    fixed_relations,
                    profiling,
                )?;
                collect_iterative_execution_stats(stats, &executor);
                self.clone_relation_store(provider, executor.store())
            }
            GpuEvaluationPass::Wfs(wfs) => {
                let schemas = wfs_plan_combined_schemas(wfs);
                let mut pass_base = self.clone_relation_store(provider, base_store)?;
                for &(source, fixed, source_store) in fixed_relations {
                    let buffer =
                        self.gpu_clone_or_empty(provider, &schemas, source, fixed, source_store)?;
                    pass_base.put(fixed, buffer);
                }
                self.run_wfs_gpu_fixpoint(provider, &pass_base, wfs, profiling, stats)
            }
        }
    }

    fn gpu_clone_or_empty(
        &self,
        provider: &Arc<CudaKernelProvider>,
        schemas: &HashMap<String, Schema>,
        source_name: &str,
        target_name: &str,
        store: &RelationStore,
    ) -> Result<CudaBuffer> {
        let target_schema = schemas
            .get(target_name)
            .or_else(|| self.schemas.get(target_name))
            .ok_or_else(|| {
                XlogError::Execution(format!(
                    "missing iterative GPU relation schema for {target_name}"
                ))
            })?;
        if let Some(buffer) = store.get(source_name) {
            // Nullary fixed relations use a unary unit marker inside WFS passes.
            // CUDA anti-join kernels then distinguish the absent and present unit
            // tuples using their ordinary nonzero-arity path.
            if buffer.schema().arity() == 0 && target_schema.arity() == 1 {
                if provider.device_row_count(buffer)? == 0 {
                    return provider.create_empty_buffer(target_schema.clone());
                }
                let marker = 1u32.to_le_bytes();
                return provider
                    .create_buffer_from_slices(&[marker.as_slice()], target_schema.clone());
            }
            return provider.clone_buffer(buffer);
        }
        provider.create_empty_buffer(target_schema.clone())
    }

    fn wfs_gpu_stores_equivalent(
        &self,
        provider: &Arc<CudaKernelProvider>,
        wfs: &EpistemicWfsGpuPlan,
        left: &RelationStore,
        right: &RelationStore,
    ) -> Result<bool> {
        self.gpu_stores_equivalent(
            provider,
            &wfs.lower.schemas,
            &wfs.intensional_predicates,
            left,
            right,
        )
    }

    fn gpu_stores_equivalent(
        &self,
        provider: &Arc<CudaKernelProvider>,
        schemas: &HashMap<String, Schema>,
        predicates: &[String],
        left: &RelationStore,
        right: &RelationStore,
    ) -> Result<bool> {
        for pred in predicates {
            let left_buf = self.gpu_clone_or_empty(provider, schemas, pred, pred, left)?;
            let right_buf = self.gpu_clone_or_empty(provider, schemas, pred, pred, right)?;
            if !buffers_gpu_set_equivalent(provider.as_ref(), &left_buf, &right_buf)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn ordinary_plan(&self, context: &str) -> Result<&ExecutionPlan> {
        match &self.plan {
            LogicExecutionPlan::Ordinary(plan) => Ok(plan),
            LogicExecutionPlan::EpistemicG91Compatibility(_)
            | LogicExecutionPlan::EpistemicWfsGpu(_)
            | LogicExecutionPlan::EpistemicSingle(_)
            | LogicExecutionPlan::EpistemicSplit(_)
            | LogicExecutionPlan::EpistemicStratified(_) => {
                Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic high-level persistent execution".to_string(),
                    context: format!(
                        "{context} requires an ordinary RIR plan; use evaluate/evaluate_with_options \
                         for production epistemic GPU dispatch"
                    ),
                })
            }
        }
    }

    fn evaluate_epistemic_with_executor(
        &self,
        provider: &Arc<CudaKernelProvider>,
        mut executor: Executor,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        let mut queries = Vec::new();
        let mut accumulated_stats = None;
        match &self.plan {
            LogicExecutionPlan::EpistemicSingle(executable) => {
                let result = executor
                    .execute_epistemic_gpu_execution(
                        executable,
                        capacities_for_epistemic_executable(executable)?,
                    )
                    .map_err(|error| self.present_epistemic_constraint_violation(error))?;
                result.require_runtime_dispatch_certification()?;
                queries.extend(epistemic_result_to_query_results(
                    epistemic_output_relation_name(executable)?,
                    result,
                ));
            }
            LogicExecutionPlan::EpistemicSplit(split) => {
                let executables: Vec<_> = split
                    .components
                    .iter()
                    .map(|component| &component.executable)
                    .collect();
                let batch = executor
                    .execute_epistemic_gpu_execution_batch_with_trace(
                        &executables,
                        capacities_for_epistemic_split(split)?,
                    )
                    .map_err(|error| self.present_epistemic_constraint_violation(error))?;
                batch
                    .require_trace_matches_components("xlog high-level epistemic GPU execution")?;
                for result in &batch.results {
                    result.require_runtime_dispatch_certification()?;
                }
                for (component, result) in split.components.iter().zip(batch.results) {
                    // A JOINT-SOLVED coalesced multi-head component yields one query
                    // per coupled head: the primary head from `final_output` plus
                    // each additional head materialized against the SAME accepted
                    // world view. Single-head components yield exactly one query.
                    queries.extend(epistemic_result_to_query_results(
                        epistemic_output_relation_name(&component.executable)?,
                        result,
                    ));
                }
            }
            LogicExecutionPlan::EpistemicStratified(stratified) => {
                // Execute strata in topological order on the SAME executor. After
                // each stratum, write its GATED head output(s) into the store as
                // base relations so the NEXT stratum's `know`/`possible` over a
                // lower head reads the gated extension through the existing tuple-key
                // membership filter (or, once the head is a materialized base
                // relation, Case-A resolve-into-body) — never double-gating against
                // a still-modal relation.
                //
                // Every authored query is evaluated by its compiled query rule only
                // after the modal heads and ordinary closure are final. This preserves
                // source order, constants, repeated-variable filters, projections and
                // logical zero-column truth without exposing whole head relations.
                let has_authored_queries = !self.program.queries.is_empty();
                let stratum_count = stratified.strata.len();
                for (stratum_index, stratum) in stratified.strata.iter().enumerate() {
                    let is_last = stratum_index + 1 == stratum_count;
                    match &stratum.plan {
                        StratumPlanKind::Single(executable) => {
                            let result = executor
                                .execute_epistemic_gpu_execution(
                                    executable,
                                    capacities_for_epistemic_executable(executable)?,
                                )
                                .map_err(|error| {
                                    self.present_epistemic_constraint_violation(error)
                                })?;
                            result.require_runtime_dispatch_certification()?;
                            let primary_head = epistemic_output_relation_name(executable)?;
                            Self::materialize_epistemic_stratum_result(
                                &mut executor,
                                primary_head,
                                result,
                                is_last && !has_authored_queries,
                                &mut queries,
                            )?;
                        }
                        StratumPlanKind::Split(split) => {
                            let executables: Vec<_> = split
                                .components
                                .iter()
                                .map(|component| &component.executable)
                                .collect();
                            let batch = executor
                                .execute_epistemic_gpu_execution_batch_with_trace(
                                    &executables,
                                    capacities_for_epistemic_split(split)?,
                                )
                                .map_err(|error| {
                                    self.present_epistemic_constraint_violation(error)
                                })?;
                            batch.require_trace_matches_components(
                                "xlog high-level stratified epistemic GPU execution",
                            )?;
                            for result in &batch.results {
                                result.require_runtime_dispatch_certification()?;
                            }
                            let primaries: Vec<String> = split
                                .components
                                .iter()
                                .map(|component| {
                                    epistemic_output_relation_name(&component.executable)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            for (primary_head, result) in primaries.into_iter().zip(batch.results) {
                                Self::materialize_epistemic_stratum_result(
                                    &mut executor,
                                    primary_head,
                                    result,
                                    is_last && !has_authored_queries,
                                    &mut queries,
                                )?;
                            }
                        }
                        StratumPlanKind::Ordinary {
                            plan,
                            head_predicates,
                        } => {
                            // Case-A recursive stratum over the materialized base
                            // determined head: the ordinary semi-naive engine writes
                            // the (correctly gated) head relation into the store.
                            executor.execute_plan(plan)?;
                            for head in head_predicates {
                                if is_last && !has_authored_queries {
                                    let buffer =
                                        executor.store().get(head.as_str()).ok_or_else(|| {
                                            XlogError::Execution(format!(
                                            "missing stratified ordinary stratum output relation \
                                             {head}"
                                        ))
                                        })?;
                                    let cloned = executor.clone_store_relation(buffer)?;
                                    queries.push(epistemic_buffer_to_query_result(
                                        head.clone(),
                                        cloned,
                                    ));
                                }
                            }
                        }
                    }
                }

                // Compiler-local relation IDs cannot safely be registered into the
                // modal executor. Run the ordinary closure, authored query rules and
                // generated constraint relations in an isolated executor seeded from
                // the final gated store, then make that store authoritative.
                if profiling {
                    accumulated_stats = Some(executor.execution_stats(0));
                }
                executor = self.run_gpu_ordinary_pass(
                    provider,
                    &stratified.ordinary_post,
                    executor.store(),
                    &[],
                    profiling,
                )?;
                if profiling {
                    collect_iterative_execution_stats(&mut accumulated_stats, &executor);
                }
                for (query_index, query) in self.program.queries.iter().enumerate() {
                    let internal_relation_name = format!("__xlog_query_{query_index}");
                    let buffer = executor
                        .store_mut()
                        .remove(&internal_relation_name)
                        .ok_or_else(|| {
                            XlogError::Execution(format!(
                                "missing stratified post-stage query relation \
                                 {internal_relation_name}"
                            ))
                        })?;
                    queries.push(self.logic_query_result(
                        provider,
                        query_index,
                        query,
                        internal_relation_name,
                        buffer,
                    )?);
                }
            }
            LogicExecutionPlan::EpistemicG91Compatibility(_)
            | LogicExecutionPlan::EpistemicWfsGpu(_) => {
                unreachable!("iterative GPU epistemic plans are handled earlier")
            }
            LogicExecutionPlan::Ordinary(_) => {
                unreachable!("ordinary plans are handled earlier")
            }
        }

        self.enforce_constraints_in_store(provider, executor.store())?;
        let total_output_rows: u64 = queries.iter().map(|q| q.buffer.num_rows()).sum();
        let stats = if profiling {
            if let Some(mut stats) = accumulated_stats {
                stats.total_output_rows = total_output_rows;
                Some(stats)
            } else {
                Some(executor.execution_stats(total_output_rows))
            }
        } else {
            None
        };
        Ok(LogicEvalResult { queries, stats })
    }

    /// Materialize one epistemic stratum result's GATED head(s) into the store.
    ///
    /// Every gated head (primary `final_output` plus joint additional heads) is
    /// written to the store so higher strata can gate against it. Explicit authored
    /// queries are projected from this store after all strata and the ordinary post
    /// stage complete.
    fn materialize_epistemic_stratum_result(
        executor: &mut Executor,
        primary_head: String,
        result: EpistemicGpuExecutionResult,
        surface_default_results: bool,
        queries: &mut Vec<LogicQueryResult>,
    ) -> Result<()> {
        executor.materialize_epistemic_head_relation(&primary_head, &result.final_output)?;
        for (head, buffer) in &result.additional_head_outputs {
            executor.materialize_epistemic_head_relation(head, buffer)?;
        }
        if surface_default_results {
            queries.extend(epistemic_result_to_query_results(primary_head, result));
        }
        Ok(())
    }

    fn enforce_constraints(
        &self,
        provider: &CudaKernelProvider,
        executor: &Executor,
    ) -> Result<()> {
        self.enforce_constraints_in_store(provider, executor.store())
    }

    fn constraint_violation_error(&self, constraint_index: usize) -> XlogError {
        let presentation_constraint = self
            .authored_constraints
            .as_ref()
            .and_then(|constraints| {
                constraints
                    .iter()
                    .find(|constraint| constraint.authored_index == Some(constraint_index))
            })
            .or_else(|| {
                self.source_program
                    .constraints
                    .iter()
                    .find(|constraint| constraint.authored_index == Some(constraint_index))
            })
            .or_else(|| {
                self.program
                    .constraints
                    .iter()
                    .find(|constraint| constraint.authored_index == Some(constraint_index))
            });
        let Some(presentation_constraint) = presentation_constraint else {
            return XlogError::Execution(format!("Constraint {constraint_index} violated"));
        };
        XlogError::Execution(format!(
            "Constraint {} violated: {}",
            constraint_index,
            format_constraint_body(&presentation_constraint.body)
        ))
    }

    fn present_epistemic_constraint_violation(&self, error: XlogError) -> XlogError {
        match error {
            XlogError::ConstraintViolation {
                constraint_index, ..
            } => self.constraint_violation_error(constraint_index),
            other => other,
        }
    }

    fn enforce_constraints_in_store(
        &self,
        provider: &CudaKernelProvider,
        store: &RelationStore,
    ) -> Result<()> {
        for constraint in &self.program.constraints {
            if constraint
                .body
                .iter()
                .any(|literal| matches!(literal, BodyLiteral::Epistemic(_)))
            {
                continue;
            }
            let i = constraint.authored_index.ok_or_else(|| {
                XlogError::Execution(
                    "ordinary constraint reached execution without an authored identity"
                        .to_string(),
                )
            })?;
            let name = format!("__xlog_constraint_{i}");
            let buf = store.get(&name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing constraint result relation {} (compiler bug?)",
                    name
                ))
            })?;

            if buf.num_rows() == 0 {
                continue;
            }

            let rows = provider.download_column::<u32>(buf, 0).unwrap_or_default();
            if rows.is_empty() {
                continue;
            }

            return Err(self.constraint_violation_error(i));
        }

        Ok(())
    }
}

fn collect_iterative_execution_stats(stats: &mut Option<ExecutionStats>, executor: &Executor) {
    let Some(combined) = stats.as_mut() else {
        return;
    };
    let mut pass = executor.execution_stats(0);
    let stratum_offset = combined.strata.len();
    for (index, stratum) in pass.strata.iter_mut().enumerate() {
        stratum.stratum_id = stratum_offset + index;
    }
    combined.total_duration_us = combined
        .total_duration_us
        .saturating_add(pass.total_duration_us);
    combined.peak_memory_bytes = combined.peak_memory_bytes.max(pass.peak_memory_bytes);
    combined.memory_budget_bytes = combined.memory_budget_bytes.max(pass.memory_budget_bytes);
    combined.wcoj_triangle_dispatch_count = combined
        .wcoj_triangle_dispatch_count
        .saturating_add(pass.wcoj_triangle_dispatch_count);
    combined.wcoj_4cycle_dispatch_count = combined
        .wcoj_4cycle_dispatch_count
        .saturating_add(pass.wcoj_4cycle_dispatch_count);
    combined.wcoj_groupby_fusion_dispatch_count = combined
        .wcoj_groupby_fusion_dispatch_count
        .saturating_add(pass.wcoj_groupby_fusion_dispatch_count);
    combined.free_join_dispatch_count = combined
        .free_join_dispatch_count
        .saturating_add(pass.free_join_dispatch_count);
    combined.factorized_delta_dispatch_count = combined
        .factorized_delta_dispatch_count
        .saturating_add(pass.factorized_delta_dispatch_count);
    combined.wcoj_error_decline_count = combined
        .wcoj_error_decline_count
        .saturating_add(pass.wcoj_error_decline_count);
    combined.strata.append(&mut pass.strata);
}

fn finalize_iterative_execution_stats(stats: &mut Option<ExecutionStats>, total_output_rows: u64) {
    if let Some(stats) = stats {
        stats.total_output_rows = total_output_rows;
    }
}

const DEFAULT_EPISTEMIC_MAX_MODELS_PER_REDUCTION: usize = 1024;

/// Normalize a parsed program through the pre-compilation passes used by execution.
///
/// This helper does not resolve imports; import-aware callers merge them first. It
/// expands user-defined functions with the entry program's recursion limit, normalizes
/// meta and list builtins, and desugars shared-variable epistemic constraints.
pub fn normalize_program_for_execution(mut program: Program) -> Result<Program> {
    if program.authored_constraint_source_bound.is_some() {
        program.validate_prepared_authored_constraint_identity()?;
    } else {
        program.prepare_authored_constraint_identity_at_root()?;
    }
    let max_recursion = program.directives.max_recursion_depth_or_default();
    let expanded = xlog_logic::expand_program_functions_owned(program, max_recursion)
        .map_err(|e| XlogError::Compilation(e.to_string()))?;
    let normalized = xlog_logic::normalize_meta_builtins_owned(expanded)?;
    let listed = xlog_logic::normalize_list_builtins_owned(normalized)?;
    Ok(desugar_shared_variable_epistemic_constraints(listed))
}

enum WfsNegationTransform<'a> {
    Drop,
    Rename {
        names: &'a HashMap<String, String>,
        source_schemas: &'a HashMap<String, Schema>,
    },
}

fn compile_g91_compatibility_gpu_plan(
    reduction: &G91CompatibilityReduction,
) -> Result<EpistemicG91CompatibilityGpuPlan> {
    let upper_bound = compile_gpu_evaluation_pass(reduction.upper_bound_program())?;
    let upper_schemas = gpu_evaluation_pass_schemas(&upper_bound);
    let mut refinement_program = reduction.refinement_program().clone();
    add_inferred_g91_snapshot_declarations(
        &mut refinement_program,
        reduction.snapshot_relations(),
        &upper_schemas,
    )?;
    let max_iterations = (refinement_program
        .directives
        .max_recursion_depth_or_default() as usize)
        .max(1);
    let refinement = compile_gpu_evaluation_pass(&refinement_program)?;
    Ok(EpistemicG91CompatibilityGpuPlan {
        upper_bound,
        refinement,
        snapshot_relations: reduction.snapshot_relations().clone(),
        convergence_predicates: reduction.convergence_predicates().to_vec(),
        max_iterations,
    })
}

fn compile_gpu_evaluation_pass(program: &Program) -> Result<GpuEvaluationPass> {
    let stratification = xlog_logic::stratify::analyze_stratification(program);
    if stratification.non_monotone_sccs.is_empty() {
        Ok(GpuEvaluationPass::Ordinary(Box::new(
            compile_gpu_ordinary_pass(program)?,
        )))
    } else {
        Ok(GpuEvaluationPass::Wfs(Box::new(
            compile_epistemic_wfs_gpu_plan(program)?,
        )))
    }
}

fn add_inferred_g91_snapshot_declarations(
    refinement: &mut Program,
    snapshots: &BTreeMap<String, String>,
    upper_schemas: &HashMap<String, Schema>,
) -> Result<()> {
    let existing = refinement
        .predicates
        .iter()
        .map(|declaration| declaration.name.clone())
        .collect::<BTreeSet<_>>();
    let mut inferred = Vec::new();
    for (source, snapshot) in snapshots {
        if existing.contains(snapshot) {
            continue;
        }
        let schema =
            upper_schemas
                .get(source)
                .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
                    construct: "Gelfond-1991 compatibility snapshot schema".to_string(),
                    context: format!(
                        "upper-bound compilation produced no schema for compatibility relation \
                     `{source}`"
                    ),
                })?;
        let columns = schema
            .columns
            .iter()
            .map(|(name, scalar_type)| PredColumn {
                name: Some(name.clone()),
                typ: TypeRef::Scalar(*scalar_type),
            })
            .collect::<Vec<_>>();
        inferred.push(PredDecl {
            name: snapshot.clone(),
            types: columns.iter().map(|column| column.typ.clone()).collect(),
            columns,
            is_private: false,
        });
    }
    refinement.predicates.extend(inferred);
    Ok(())
}

fn compile_epistemic_wfs_gpu_plan(program: &Program) -> Result<EpistemicWfsGpuPlan> {
    let negated = wfs_negated_predicates(program);
    let upper_fixed_names = wfs_fixed_names(program, &negated, "__wfs_upper");
    let lower_fixed_names = wfs_fixed_names(program, &negated, "__wfs_lower");
    let source_schemas = infer_wfs_source_schemas(program)?;

    let mut overapprox_program = wfs_transform_program(program, WfsNegationTransform::Drop)?;
    // Constraints do not influence the alternating fixpoint. Evaluate them only in
    // the lower pass, where positive atoms read the true extension and negated atoms
    // read the frozen upper extension.
    overapprox_program.constraints.clear();
    let lower_program = wfs_transform_program(
        program,
        WfsNegationTransform::Rename {
            names: &upper_fixed_names,
            source_schemas: &source_schemas,
        },
    )?;
    let mut upper_program = wfs_transform_program(
        program,
        WfsNegationTransform::Rename {
            names: &lower_fixed_names,
            source_schemas: &source_schemas,
        },
    )?;
    upper_program.constraints.clear();

    Ok(EpistemicWfsGpuPlan {
        overapprox: compile_gpu_ordinary_pass(&overapprox_program)?,
        lower: compile_gpu_ordinary_pass(&lower_program)?,
        upper: compile_gpu_ordinary_pass(&upper_program)?,
        intensional_predicates: wfs_intensional_predicates(program),
        upper_fixed_names,
        lower_fixed_names,
        max_iterations: (program.directives.max_recursion_depth_or_default() as usize).max(1),
    })
}

fn infer_wfs_source_schemas(program: &Program) -> Result<HashMap<String, Schema>> {
    // Ordinary compilation cannot plan a cycle through negation, but its schema
    // inference treats positive and negated atoms identically. Make a monotone copy
    // that retains every atom, compile it through the authoritative compiler path,
    // and use the resulting schemas for the private fixed relations.
    let mut inference_program = program.clone();
    for rule in &mut inference_program.rules {
        for literal in &mut rule.body {
            if let BodyLiteral::Negated(atom) = literal {
                *literal = BodyLiteral::Positive(atom.clone());
            }
        }
    }

    let mut compiler = Compiler::new();
    compiler.compile_prepared_program(&inference_program)?;
    Ok(compiler.schemas().clone())
}

fn compile_gpu_ordinary_pass(program: &Program) -> Result<GpuOrdinaryPass> {
    let mut compiler = Compiler::new();
    let plan = compiler.compile_prepared_program(program)?;
    Ok(GpuOrdinaryPass {
        plan,
        schemas: compiler.schemas().clone(),
        rel_ids: compiler.rel_ids().clone(),
    })
}

fn wfs_transform_program(program: &Program, negation: WfsNegationTransform<'_>) -> Result<Program> {
    let mut out = program.clone();
    out.rules = program
        .rules
        .iter()
        .map(|rule| {
            let mut rule = rule.clone();
            let was_fact = rule.body.is_empty();
            let mut body = transform_wfs_body(&rule.body, &negation)?;
            if !was_fact && body.is_empty() {
                // Dropping every negated literal computes the WFS upper
                // over-approximation. Keep the transformed clause executable as a
                // unit-derived rule; an empty body would otherwise be reclassified as
                // an extensional fact and never derived by the pass.
                body.push(BodyLiteral::Comparison(xlog_logic::ast::Comparison {
                    left: Term::Integer(1),
                    op: xlog_logic::ast::CompOp::Eq,
                    right: Term::Integer(1),
                }));
            }
            rule.body = body;
            Ok(rule)
        })
        .collect::<Result<Vec<_>>>()?;
    out.constraints = program
        .constraints
        .iter()
        .map(|constraint| {
            let mut constraint = constraint.clone();
            constraint.body = transform_wfs_body(&constraint.body, &negation)?;
            Ok(constraint)
        })
        .collect::<Result<Vec<_>>>()?;
    if let WfsNegationTransform::Rename {
        names,
        source_schemas,
    } = negation
    {
        add_wfs_fixed_predicates(&mut out, names, source_schemas)?;
    }
    Ok(out)
}

fn transform_wfs_body(
    body: &[BodyLiteral],
    negation: &WfsNegationTransform<'_>,
) -> Result<Vec<BodyLiteral>> {
    let mut transformed = Vec::with_capacity(body.len());
    for literal in body {
        match (literal, negation) {
            (BodyLiteral::Negated(_), WfsNegationTransform::Drop) => {}
            (BodyLiteral::Negated(atom), WfsNegationTransform::Rename { names, .. }) => {
                let mut atom = atom.clone();
                atom.predicate = names.get(&atom.predicate).cloned().ok_or_else(|| {
                    XlogError::Execution(format!(
                        "missing WFS fixed relation name for {}",
                        atom.predicate
                    ))
                })?;
                if atom.terms.is_empty() {
                    atom.terms.push(Term::Integer(1));
                }
                transformed.push(BodyLiteral::Negated(atom));
            }
            _ => transformed.push(literal.clone()),
        }
    }
    Ok(transformed)
}

fn add_wfs_fixed_predicates(
    program: &mut Program,
    names: &HashMap<String, String>,
    source_schemas: &HashMap<String, Schema>,
) -> Result<()> {
    let existing: BTreeSet<String> = program
        .predicates
        .iter()
        .map(|decl| decl.name.clone())
        .collect();
    for (source, fixed) in names {
        if existing.contains(fixed) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU WFS fixed relation name".to_string(),
                context: format!(
                    "internal fixed relation {fixed} collides with a declared predicate"
                ),
            });
        }
        let Some(schema) = source_schemas.get(source) else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU WFS fixed relation schema".to_string(),
                context: format!(
                    "ordinary schema inference produced no schema for negated predicate {source}"
                ),
            });
        };

        let is_private = program
            .predicates
            .iter()
            .find(|declaration| declaration.name == *source)
            .is_some_and(|declaration| declaration.is_private);
        let columns = if schema.arity() == 0 {
            vec![PredColumn {
                name: Some("present".to_string()),
                typ: TypeRef::Scalar(ScalarType::U32),
            }]
        } else {
            schema
                .columns
                .iter()
                .map(|(name, scalar_type)| PredColumn {
                    name: Some(name.clone()),
                    typ: TypeRef::Scalar(*scalar_type),
                })
                .collect::<Vec<_>>()
        };
        program.predicates.push(PredDecl {
            name: fixed.clone(),
            types: columns.iter().map(|column| column.typ.clone()).collect(),
            columns,
            is_private,
        });
    }
    Ok(())
}

fn wfs_negated_predicates(program: &Program) -> BTreeSet<String> {
    program
        .rules
        .iter()
        .map(|rule| &rule.body)
        .chain(
            program
                .constraints
                .iter()
                .map(|constraint| &constraint.body),
        )
        .flatten()
        .filter_map(|lit| match lit {
            BodyLiteral::Negated(atom) => Some(atom.predicate.clone()),
            _ => None,
        })
        .collect()
}

fn wfs_intensional_predicates(program: &Program) -> Vec<String> {
    program
        .proper_rules()
        .map(|rule| rule.head.predicate.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn wfs_fixed_names(
    program: &Program,
    predicates: &BTreeSet<String>,
    prefix: &str,
) -> HashMap<String, String> {
    let mut reserved: BTreeSet<String> = program
        .predicates
        .iter()
        .map(|decl| decl.name.clone())
        .collect();
    let mut names = HashMap::new();
    for pred in predicates {
        let mut candidate = format!("{prefix}_{pred}");
        if reserved.contains(&candidate) {
            let mut suffix = 0usize;
            loop {
                let suffixed = format!("{prefix}_{suffix}_{pred}");
                if !reserved.contains(&suffixed) {
                    candidate = suffixed;
                    break;
                }
                suffix += 1;
            }
        }
        reserved.insert(candidate.clone());
        names.insert(pred.clone(), candidate);
    }
    names
}

fn wfs_plan_combined_schemas(plan: &EpistemicWfsGpuPlan) -> HashMap<String, Schema> {
    let mut schemas = HashMap::new();
    for ordinary in [&plan.overapprox, &plan.lower, &plan.upper] {
        for (name, schema) in &ordinary.schemas {
            schemas
                .entry(name.clone())
                .or_insert_with(|| schema.clone());
        }
    }
    schemas
}

fn g91_plan_combined_schemas(plan: &EpistemicG91CompatibilityGpuPlan) -> HashMap<String, Schema> {
    let mut schemas = HashMap::new();
    for pass in [&plan.upper_bound, &plan.refinement] {
        for (name, schema) in gpu_evaluation_pass_schemas(pass) {
            schemas.entry(name).or_insert(schema);
        }
    }
    schemas
}

fn gpu_evaluation_pass_schemas(pass: &GpuEvaluationPass) -> HashMap<String, Schema> {
    match pass {
        GpuEvaluationPass::Ordinary(ordinary) => ordinary.schemas.clone(),
        GpuEvaluationPass::Wfs(wfs) => wfs_plan_combined_schemas(wfs),
    }
}

fn g91_plan_combined_rel_ids(plan: &EpistemicG91CompatibilityGpuPlan) -> HashMap<String, RelId> {
    let mut rel_ids = HashMap::new();
    for pass in [&plan.upper_bound, &plan.refinement] {
        for (name, rel_id) in gpu_evaluation_pass_rel_ids(pass) {
            rel_ids.insert(name, rel_id);
        }
    }
    rel_ids
}

fn gpu_evaluation_pass_rel_ids(pass: &GpuEvaluationPass) -> HashMap<String, RelId> {
    match pass {
        GpuEvaluationPass::Ordinary(ordinary) => ordinary.rel_ids.clone(),
        GpuEvaluationPass::Wfs(wfs) => wfs_plan_combined_rel_ids(wfs),
    }
}

fn wfs_plan_combined_rel_ids(plan: &EpistemicWfsGpuPlan) -> HashMap<String, RelId> {
    let mut rel_ids = HashMap::new();
    for ordinary in [&plan.overapprox, &plan.lower, &plan.upper] {
        for (name, rel_id) in &ordinary.rel_ids {
            rel_ids.insert(name.clone(), *rel_id);
        }
    }
    rel_ids
}

fn schema_from_pred_decl(
    decl: &xlog_logic::ast::PredDecl,
    domains: &HashMap<String, ScalarType>,
) -> Result<Schema> {
    let columns = decl.schema_columns();
    let resolved = columns
        .iter()
        .enumerate()
        .map(|(idx, column)| {
            let name = column.name.clone().unwrap_or_else(|| format!("c{idx}"));
            resolve_pred_column_type(&decl.name, idx, &column.typ, domains).map(|typ| (name, typ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Schema::new(resolved))
}

fn resolve_pred_column_type(
    predicate: &str,
    index: usize,
    typ: &TypeRef,
    domains: &HashMap<String, ScalarType>,
) -> Result<ScalarType> {
    match typ {
        TypeRef::Scalar(ty) => Ok(*ty),
        TypeRef::Domain(name) => domains.get(name).copied().ok_or_else(|| {
            XlogError::Compilation(format!(
                "unknown domain alias '{}' in predicate '{}' column {}",
                name, predicate, index
            ))
        }),
        TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
            Ok(ScalarType::U64)
        }
    }
}

fn schema_from_terms(terms: &[Term]) -> Schema {
    let columns = terms
        .iter()
        .enumerate()
        .map(|(idx, term)| (format!("c{idx}"), term.inferred_scalar_type()))
        .collect();
    Schema::new(columns)
}

/// Desugar a shared-variable epistemic constraint — a constraint with at least one
/// epistemic literal and a variable appearing in more than one term position across the body
/// (the join `:- know p(X), possible q(X).`, the diagonal `:- know p(X, X).`, or the
/// negated-difference `:- q(X), not know p(X).`) — into an ordinary extraction rule plus a
/// single-occurrence modal over it:
///
/// ```text
///   :- BodyLit1, BodyLit2, ..., BodyLitN.
///        ==> __epi_join_N(Vars) :- ord(BodyLit1), ..., ord(BodyLitN).
///            :- know __epi_join_N(Vars).
/// ```
///
/// where `ord` ordinary-izes each modal literal (`know/possible r(..)` -> `r(..)`,
/// `not know/possible r(..)` -> `not r(..)`) and keeps non-modal literals unchanged. For a
/// base/EDB or purely-ordinary-derived modal target `know r == possible r == r`, so the
/// ordinary join `__epi_join_N` is exactly the set of variable bindings the constraint
/// forbids; the single-occurrence `:- know __epi_join_N(Vars)` then routes through the
/// existing variable-keyed world-view constraint path, which prunes the world view to empty —
/// with NO new kernel. Applied at the normalization choke point so BOTH the reduced ordinary
/// materialization and the epistemic planner observe the helper relation (an EIR-only rewrite
/// is accepted at planning but the helper is never materialized).
///
/// Guarded to non-modal-derived targets (where the `know == possible == ordinary`
/// equivalence holds); a constraint with a modal-derived target is left unchanged and falls
/// through to the core compiler's existing shared-variable rejection. Single-occurrence
/// variable-keyed modal, distinct-variable multi-literal, and ground constraints have no
/// repeated variable and are likewise untouched.
fn desugar_shared_variable_epistemic_constraints(mut program: Program) -> Program {
    // A predicate defined by any rule carrying an epistemic body literal is "modal-derived":
    // for it `know p`/`possible p` is NOT equal to the ordinary `p`, so ordinary-izing it
    // would be UNSOUND. Restrict the desugaring to base/EDB or purely-ordinary-derived
    // targets (where `know p == possible p == p`), the case for base tuple-key targets.
    let modal_derived: BTreeSet<String> = program
        .rules
        .iter()
        .filter(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
        .map(|rule| rule.head.predicate.clone())
        .collect();
    let mut extraction_rules: Vec<Rule> = Vec::new();
    let mut counter = 0usize;
    for constraint in &mut program.constraints {
        let has_epistemic = constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        if !has_epistemic || !constraint_has_shared_variable(&constraint.body) {
            continue;
        }
        // Sound only when EVERY modal target is non-modal-derived (know == possible == ord).
        let has_modal_derived_target = constraint.body.iter().any(|lit| {
            matches!(lit, BodyLiteral::Epistemic(e) if modal_derived.contains(&e.atom.predicate))
        });
        if has_modal_derived_target {
            continue;
        }
        let distinct = distinct_body_variables(&constraint.body);
        let helper = format!("__epi_join_{counter}");
        counter += 1;
        let helper_terms: Vec<Term> = distinct.iter().map(|v| Term::Variable(v.clone())).collect();
        let helper_body: Vec<BodyLiteral> = constraint
            .body
            .iter()
            .map(ordinaryize_modal_literal)
            .collect();
        extraction_rules.push(Rule {
            head: Atom {
                predicate: helper.clone(),
                terms: helper_terms.clone(),
            },
            body: helper_body,
        });
        // Replace the whole constraint with a single-occurrence modal over the join helper.
        constraint.body = vec![BodyLiteral::Epistemic(EpistemicLiteral {
            op: EpistemicOp::Know,
            negated: false,
            atom: Atom {
                predicate: helper,
                terms: helper_terms,
            },
        })];
    }
    program.rules.extend(extraction_rules);
    program
}

/// Replace a modal literal with its ordinary counterpart (`know/possible r` -> `r`,
/// `not know/possible r` -> `not r`); non-modal literals are returned unchanged. Sound for
/// the shared-variable constraint desugaring when the modal target is non-modal-derived,
/// where `know r == possible r == r`.
fn ordinaryize_modal_literal(lit: &BodyLiteral) -> BodyLiteral {
    match lit {
        BodyLiteral::Epistemic(e) if e.negated => BodyLiteral::Negated(e.atom.clone()),
        BodyLiteral::Epistemic(e) => BodyLiteral::Positive(e.atom.clone()),
        other => other.clone(),
    }
}

/// True if some variable occurs in more than one atom term position across the constraint
/// body — the signature of a join / diagonal / negated-difference the core compiler rejects.
fn constraint_has_shared_variable(body: &[BodyLiteral]) -> bool {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for lit in body {
        if let Some(atom) = lit.atom() {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    *counts.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    counts.values().any(|&count| count > 1)
}

/// Ordered DISTINCT variable names appearing in atom positions across the constraint body
/// (first-appearance order), used as the extracted helper relation's columns.
fn distinct_body_variables(body: &[BodyLiteral]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for lit in body {
        if let Some(atom) = lit.atom() {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    if seen.insert(name.clone()) {
                        order.push(name.clone());
                    }
                }
            }
        }
    }
    order
}

fn augment_same_name_multi_arity_schemas(
    program: &Program,
    schemas: &mut HashMap<String, Schema>,
) -> Result<()> {
    let predicates = if program_has_epistemic_literals(program) {
        epistemic_extensional_multi_arity_predicates(program)
    } else {
        predicate_arities(program)
            .into_iter()
            .filter_map(|(predicate, arities)| (arities.len() > 1).then_some(predicate))
            .collect()
    };
    let domains: HashMap<String, ScalarType> = program
        .domains
        .iter()
        .map(|domain| (domain.name.clone(), domain.typ))
        .collect();

    for decl in &program.predicates {
        if !predicates.contains(&decl.name) {
            continue;
        }
        let key = arity_qualified_name(&decl.name, decl.arity());
        schemas.insert(key, schema_from_pred_decl(decl, &domains)?);
    }

    for fact in program.facts() {
        let pred = fact.head.predicate.as_str();
        let arity = fact.head.terms.len();
        if !predicates.contains(pred) {
            continue;
        }
        let key = arity_qualified_name(pred, arity);
        schemas
            .entry(key)
            .or_insert_with(|| schema_from_terms(&fact.head.terms));
    }

    for rule in &program.rules {
        augment_atom_schema_if_needed(&rule.head, &predicates, schemas);
        for literal in &rule.body {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    augment_atom_schema_if_needed(atom, &predicates, schemas);
                }
                BodyLiteral::Epistemic(epistemic) => {
                    augment_atom_schema_if_needed(&epistemic.atom, &predicates, schemas);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }
    }

    for query in &program.queries {
        augment_atom_schema_if_needed(&query.atom, &predicates, schemas);
    }

    Ok(())
}

/// Arity-qualify same-name multi-arity predicates; `None` when the program has
/// no such overloads (the caller then compiles `program` itself, unchanged).
fn qualify_same_name_multi_arity_program(program: &Program) -> Option<Program> {
    let overloaded = predicate_arities(program)
        .into_iter()
        .filter_map(|(predicate, arities)| (arities.len() > 1).then_some(predicate))
        .collect::<BTreeSet<_>>();
    if overloaded.is_empty() {
        return None;
    }

    let mut qualified = program.clone();
    for declaration in &mut qualified.predicates {
        if overloaded.contains(&declaration.name) {
            declaration.name = arity_qualified_name(&declaration.name, declaration.arity());
        }
    }
    for rule in &mut qualified.rules {
        qualify_atom_arity(&mut rule.head, &overloaded);
        qualify_body_literal_arities(&mut rule.body, &overloaded);
    }
    for constraint in &mut qualified.constraints {
        qualify_body_literal_arities(&mut constraint.body, &overloaded);
    }
    for query in &mut qualified.queries {
        qualify_atom_arity(&mut query.atom, &overloaded);
    }
    Some(qualified)
}

fn qualify_body_literal_arities(literals: &mut [BodyLiteral], overloaded: &BTreeSet<String>) {
    for literal in literals {
        match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                qualify_atom_arity(atom, overloaded);
            }
            BodyLiteral::Epistemic(epistemic) => {
                qualify_atom_arity(&mut epistemic.atom, overloaded);
            }
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
        }
    }
}

fn qualify_atom_arity(atom: &mut Atom, overloaded: &BTreeSet<String>) {
    if overloaded.contains(&atom.predicate) {
        atom.predicate = arity_qualified_name(&atom.predicate, atom.terms.len());
    }
}

fn augment_atom_schema_if_needed(
    atom: &Atom,
    predicates: &BTreeSet<String>,
    schemas: &mut HashMap<String, Schema>,
) {
    if !predicates.contains(&atom.predicate) {
        return;
    }
    let key = arity_qualified_name(&atom.predicate, atom.terms.len());
    schemas
        .entry(key)
        .or_insert_with(|| schema_from_terms(&atom.terms));
}

fn predicate_arities(program: &Program) -> HashMap<String, BTreeSet<usize>> {
    let mut arities = HashMap::new();
    for decl in &program.predicates {
        add_predicate_arity(&mut arities, &decl.name, decl.arity());
    }
    for rule in &program.rules {
        add_predicate_arity(&mut arities, &rule.head.predicate, rule.head.terms.len());
        for literal in &rule.body {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    add_predicate_arity(&mut arities, &atom.predicate, atom.terms.len());
                }
                BodyLiteral::Epistemic(epistemic) => {
                    add_predicate_arity(
                        &mut arities,
                        &epistemic.atom.predicate,
                        epistemic.atom.terms.len(),
                    );
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }
    }
    for query in &program.queries {
        add_predicate_arity(&mut arities, &query.atom.predicate, query.atom.terms.len());
    }
    for constraint in &program.constraints {
        for literal in &constraint.body {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    add_predicate_arity(&mut arities, &atom.predicate, atom.terms.len());
                }
                BodyLiteral::Epistemic(epistemic) => {
                    add_predicate_arity(
                        &mut arities,
                        &epistemic.atom.predicate,
                        epistemic.atom.terms.len(),
                    );
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }
    }
    arities
}

fn add_predicate_arity(
    arities: &mut HashMap<String, BTreeSet<usize>>,
    predicate: &str,
    arity: usize,
) {
    arities
        .entry(predicate.to_string())
        .or_default()
        .insert(arity);
}

fn arity_qualified_name(predicate: &str, arity: usize) -> String {
    format!("{predicate}/{arity}")
}

fn program_has_epistemic_literals(program: &Program) -> bool {
    program.rules.iter().any(|rule| {
        rule.body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    }) || program.constraints.iter().any(|constraint| {
        constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    })
}

fn epistemic_output_head_predicate_count(program: &Program) -> usize {
    program
        .rules
        .iter()
        .filter(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
        .map(|rule| rule.head.predicate.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

/// The user-visible output head predicate(s) of a stratum's epistemic-bearing
/// rules. For a recursive stratum (`reach :- reach, know a`) this is the recursive
/// head whose materialized relation is the stratum's output.
fn epistemic_stratum_output_heads(program: &Program) -> Vec<String> {
    program
        .rules
        .iter()
        .filter(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
        .map(|rule| rule.head.predicate.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn epistemic_relation_ids(plan: &LogicExecutionPlan) -> Result<HashMap<String, RelId>> {
    let mut rel_ids = HashMap::new();
    match plan {
        LogicExecutionPlan::EpistemicSingle(executable) => {
            for (name, rel_id) in &executable.relation_ids {
                insert_epistemic_relation_id(&mut rel_ids, name, *rel_id)?;
            }
        }
        LogicExecutionPlan::EpistemicSplit(split) => {
            for component in &split.components {
                for (name, rel_id) in &component.executable.relation_ids {
                    insert_epistemic_relation_id(&mut rel_ids, name, *rel_id)?;
                }
            }
        }
        LogicExecutionPlan::EpistemicStratified(stratified) => {
            for stratum in &stratified.strata {
                match &stratum.plan {
                    StratumPlanKind::Single(executable) => {
                        for (name, rel_id) in &executable.relation_ids {
                            // Each stratum is a distinct sub-program compiled with a
                            // fresh compiler, so relation ids legitimately differ
                            // across strata; keep the last writer per name.
                            rel_ids.insert(name.clone(), *rel_id);
                        }
                    }
                    StratumPlanKind::Split(split) => {
                        for component in &split.components {
                            for (name, rel_id) in &component.executable.relation_ids {
                                rel_ids.insert(name.clone(), *rel_id);
                            }
                        }
                    }
                    // An ordinary (Case-A recursive) stratum carries no epistemic
                    // relation-id map; its relations are owned by its own ordinary
                    // RIR plan and surfaced from the store after execution.
                    StratumPlanKind::Ordinary { .. } => {}
                }
            }
        }
        LogicExecutionPlan::EpistemicWfsGpu(wfs) => {
            for plan in [&wfs.overapprox, &wfs.lower, &wfs.upper] {
                for (name, rel_id) in &plan.rel_ids {
                    rel_ids.insert(name.clone(), *rel_id);
                }
            }
        }
        LogicExecutionPlan::EpistemicG91Compatibility(g91) => {
            for pass in [&g91.upper_bound, &g91.refinement] {
                for (name, rel_id) in gpu_evaluation_pass_rel_ids(pass) {
                    rel_ids.insert(name, rel_id);
                }
            }
        }
        LogicExecutionPlan::Ordinary(_) => {}
    }
    Ok(rel_ids)
}

fn insert_epistemic_relation_id(
    rel_ids: &mut HashMap<String, RelId>,
    name: &str,
    rel_id: RelId,
) -> Result<()> {
    if let Some(previous) = rel_ids.insert(name.to_string(), rel_id) {
        if previous != rel_id {
            return Err(XlogError::Compilation(format!(
                "epistemic split components assigned conflicting relation ids for {name}: \
                 {previous:?} vs {rel_id:?}"
            )));
        }
    }
    Ok(())
}

fn capacities_for_epistemic_executable(
    executable: &EpistemicExecutablePlan,
) -> Result<EpistemicGpuWorkspaceCapacities> {
    let literal_count = executable.gpu_plan.epistemic_literals.len();
    let max_candidates = 1usize.checked_shl(literal_count as u32).ok_or_else(|| {
        XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU execution candidate generation".to_string(),
            context: format!("literal count {literal_count} exceeds target pointer width"),
        }
    })?;
    Ok(EpistemicGpuWorkspaceCapacities {
        max_candidates,
        max_worlds: 1,
        max_models_per_reduction: DEFAULT_EPISTEMIC_MAX_MODELS_PER_REDUCTION,
    })
}

fn capacities_for_epistemic_split(
    split: &EpistemicSplitExecutablePlan,
) -> Result<EpistemicGpuWorkspaceCapacities> {
    let mut capacities = EpistemicGpuWorkspaceCapacities {
        max_candidates: 1,
        max_worlds: 1,
        max_models_per_reduction: DEFAULT_EPISTEMIC_MAX_MODELS_PER_REDUCTION,
    };
    for component in &split.components {
        let component_capacities = capacities_for_epistemic_executable(&component.executable)?;
        capacities.max_candidates = capacities
            .max_candidates
            .max(component_capacities.max_candidates);
    }
    Ok(capacities)
}

fn epistemic_output_relation_name(executable: &EpistemicExecutablePlan) -> Result<String> {
    executable
        .gpu_plan
        .reductions
        .last()
        .map(|reduction| reduction.head_predicate.clone())
        .ok_or_else(|| XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU reduced output".to_string(),
            context: "executable plan has no epistemic reductions".to_string(),
        })
}

fn epistemic_buffer_to_query_result(relation_name: String, buffer: CudaBuffer) -> LogicQueryResult {
    let schema = buffer.schema();
    let columns = schema
        .columns
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let sort_labels = schema.sort_labels().to_vec();
    LogicQueryResult {
        relation_name,
        columns,
        sort_labels,
        buffer,
    }
}

/// Convert an epistemic GPU execution result into one query result per output head.
///
/// `primary_relation_name` is the primary head (from `final_output`). A
/// JOINT-SOLVED coalesced multi-head component also carries
/// `additional_head_outputs`, each materialized against the SAME accepted world
/// view; every coupled head becomes its own query result so `xlog run` displays
/// all coupled epistemic outputs.
fn epistemic_result_to_query_results(
    primary_relation_name: String,
    result: EpistemicGpuExecutionResult,
) -> Vec<LogicQueryResult> {
    let mut results = Vec::with_capacity(1 + result.additional_head_outputs.len());
    for (head, buffer) in result.additional_head_outputs {
        results.push(epistemic_buffer_to_query_result(head, buffer));
    }
    results.push(epistemic_buffer_to_query_result(
        primary_relation_name,
        result.final_output,
    ));
    results
}

fn is_user_visible_relation(name: &str) -> bool {
    !name.starts_with("__")
}

fn is_list_helper_relation(name: &str) -> bool {
    name.starts_with("__xlog_list_")
}

fn relation_clone_error(context: String, error: XlogError) -> XlogError {
    match error {
        XlogError::ResourceExhausted {
            context: source_context,
            estimated_bytes,
            budget_bytes,
        } => XlogError::ResourceExhausted {
            context: format!("{context}: {source_context}"),
            estimated_bytes,
            budget_bytes,
        },
        XlogError::Kernel(message) => XlogError::Kernel(format!("{context}: {message}")),
        error => error,
    }
}

fn logic_delta_report(
    stats: DeltaRecomputeStats,
    insert_rows: u64,
    delete_rows: u64,
) -> LogicDeltaReport {
    LogicDeltaReport {
        input_delta_count: stats.changed_relations,
        changed_relations: stats.changed_relations,
        changed_relation_names: Vec::new(),
        insert_rows,
        delete_rows,
        has_deletes: stats.has_deletes,
        affected_sccs: stats.affected_sccs,
        recomputed_sccs: stats.recomputed_sccs,
        incremental_sccs: stats.incremental_sccs,
        coalesced_insert_rows: insert_rows,
        coalesced_delete_rows: delete_rows,
        canceled_rows: 0,
        planner_telemetry: DeltaPlannerTelemetry::default(),
        debug_trace: Vec::new(),
    }
}

fn no_op_delta_report(seed: PreparedRelationDeltaReportSeed) -> LogicDeltaReport {
    LogicDeltaReport {
        input_delta_count: seed.input_delta_count,
        changed_relations: 0,
        changed_relation_names: Vec::new(),
        insert_rows: 0,
        delete_rows: 0,
        has_deletes: false,
        affected_sccs: 0,
        recomputed_sccs: 0,
        incremental_sccs: 0,
        coalesced_insert_rows: 0,
        coalesced_delete_rows: 0,
        canceled_rows: seed.canceled_rows,
        planner_telemetry: DeltaPlannerTelemetry {
            fallback_decision: "no_op".to_string(),
            ..DeltaPlannerTelemetry::default()
        },
        debug_trace: vec![format!("canceled_rows={}", seed.canceled_rows)],
    }
}

fn delta_debug_trace(report: &LogicDeltaReport) -> Vec<String> {
    vec![
        format!("changed_relation_names={:?}", report.changed_relation_names),
        format!("affected_sccs={}", report.affected_sccs),
        format!("recomputed_sccs={}", report.recomputed_sccs),
        format!("incremental_sccs={}", report.incremental_sccs),
        format!("insert_rows={}", report.insert_rows),
        format!("delete_rows={}", report.delete_rows),
        format!(
            "planner_fallback_decision={}",
            report.planner_telemetry.fallback_decision
        ),
        format!(
            "estimated_delta_speedup={:?}",
            report.planner_telemetry.estimated_delta_speedup
        ),
    ]
}

fn buffers_gpu_set_equivalent(
    provider: &CudaKernelProvider,
    left: &CudaBuffer,
    right: &CudaBuffer,
) -> Result<bool> {
    if left.schema() != right.schema() {
        return Ok(false);
    }
    let left_rows = provider.device_row_count(left)?;
    let right_rows = provider.device_row_count(right)?;
    if left_rows != right_rows {
        return Ok(false);
    }

    let left_minus_right = provider.diff_full_row(left, right)?;
    if provider.device_row_count(&left_minus_right)? != 0 {
        return Ok(false);
    }
    let right_minus_left = provider.diff_full_row(right, left)?;
    Ok(provider.device_row_count(&right_minus_left)? == 0)
}

fn coalesce_relation_delta_batch_with_cancellation_capture(
    provider: &CudaKernelProvider,
    delta_batch: Vec<(String, RelationDelta)>,
    cancellation_capture_relations: &BTreeSet<String>,
) -> Result<PreparedRelationDeltaBatch> {
    let input_delta_count = delta_batch.len();
    let mut pending_by_relation: HashMap<String, PendingRelationDelta> = HashMap::new();
    let mut cancellations: HashMap<String, Vec<RelationDeltaCancellation>> = HashMap::new();
    let mut canceled_rows = 0u64;

    for (update_index, (name, delta)) in delta_batch.into_iter().enumerate() {
        let capture_cancellations = cancellation_capture_relations.contains(&name);
        let cancellation_relation = capture_cancellations.then(|| name.clone());
        let mut update_cancellations = capture_cancellations.then(Vec::new);
        let pending = pending_by_relation.entry(name).or_default();
        if let Some(insert) = delta.insert {
            merge_insert_delta(
                provider,
                pending,
                insert,
                &mut canceled_rows,
                update_index,
                update_cancellations.as_mut(),
            )?;
        }
        if let Some(delete) = delta.delete {
            merge_delete_delta(
                provider,
                pending,
                delete,
                &mut canceled_rows,
                update_index,
                update_cancellations.as_mut(),
            )?;
        }
        if let Some(mut captured) = update_cancellations.filter(|trace| !trace.is_empty()) {
            cancellations
                .entry(cancellation_relation.expect("capture relation must be retained"))
                .or_default()
                .append(&mut captured);
        }
    }

    let mut deltas = HashMap::new();
    let mut coalesced_insert_rows = 0u64;
    let mut coalesced_delete_rows = 0u64;
    for (name, pending) in pending_by_relation {
        let insert = pending.insert.and_then(non_empty_buffer);
        let delete = pending.delete.and_then(non_empty_buffer);
        if insert.is_none() && delete.is_none() {
            continue;
        }
        coalesced_insert_rows += insert.as_ref().map(buffer_rows).unwrap_or(0);
        coalesced_delete_rows += delete.as_ref().map(buffer_rows).unwrap_or(0);
        deltas.insert(name, RelationDelta::new(insert, delete));
    }

    let changed_relations = deltas.len();
    Ok(PreparedRelationDeltaBatch {
        deltas,
        cancellations,
        report_seed: PreparedRelationDeltaReportSeed {
            input_delta_count,
            changed_relations,
            coalesced_insert_rows,
            coalesced_delete_rows,
            canceled_rows,
        },
    })
}

fn merge_insert_delta(
    provider: &CudaKernelProvider,
    pending: &mut PendingRelationDelta,
    insert: CudaBuffer,
    canceled_rows: &mut u64,
    update_index: usize,
    cancellations: Option<&mut Vec<RelationDeltaCancellation>>,
) -> Result<()> {
    let mut incoming = provider.dedup_full_row(&insert)?;
    if let Some(delete) = pending.delete.take().and_then(non_empty_buffer) {
        let delete_before = buffer_rows(&delete);
        let delete_after = provider.diff_full_row(&delete, &incoming)?;
        let insert_after = provider.diff_full_row(&incoming, &delete)?;
        *canceled_rows += delete_before.saturating_sub(buffer_rows(&delete_after));
        capture_canceled_tuples(
            provider,
            &incoming,
            &insert_after,
            update_index,
            RelationDeltaDirection::Insert,
            cancellations,
        )?;
        pending.delete = non_empty_buffer(delete_after);
        incoming = insert_after;
    }
    pending.insert = merge_optional_buffer(provider, pending.insert.take(), incoming)?;
    Ok(())
}

fn merge_delete_delta(
    provider: &CudaKernelProvider,
    pending: &mut PendingRelationDelta,
    delete: CudaBuffer,
    canceled_rows: &mut u64,
    update_index: usize,
    cancellations: Option<&mut Vec<RelationDeltaCancellation>>,
) -> Result<()> {
    let mut incoming = provider.dedup_full_row(&delete)?;
    if let Some(insert) = pending.insert.take().and_then(non_empty_buffer) {
        let insert_before = buffer_rows(&insert);
        let insert_after = provider.diff_full_row(&insert, &incoming)?;
        let delete_after = provider.diff_full_row(&incoming, &insert)?;
        *canceled_rows += insert_before.saturating_sub(buffer_rows(&insert_after));
        capture_canceled_tuples(
            provider,
            &incoming,
            &delete_after,
            update_index,
            RelationDeltaDirection::Delete,
            cancellations,
        )?;
        pending.insert = non_empty_buffer(insert_after);
        incoming = delete_after;
    }
    pending.delete = merge_optional_buffer(provider, pending.delete.take(), incoming)?;
    Ok(())
}

fn capture_canceled_tuples(
    provider: &CudaKernelProvider,
    incoming: &CudaBuffer,
    incoming_after_cancellation: &CudaBuffer,
    update_index: usize,
    incoming_direction: RelationDeltaDirection,
    cancellations: Option<&mut Vec<RelationDeltaCancellation>>,
) -> Result<()> {
    let Some(cancellations) = cancellations else {
        return Ok(());
    };
    let intersection = provider.diff_full_row(incoming, incoming_after_cancellation)?;
    if let Some(tuples) = non_empty_buffer(intersection) {
        cancellations.push(RelationDeltaCancellation {
            update_index,
            incoming_direction,
            tuples,
        });
    }
    Ok(())
}

fn merge_optional_buffer(
    provider: &CudaKernelProvider,
    existing: Option<CudaBuffer>,
    incoming: CudaBuffer,
) -> Result<Option<CudaBuffer>> {
    let Some(incoming) = non_empty_buffer(incoming) else {
        return Ok(existing.and_then(non_empty_buffer));
    };
    match existing.and_then(non_empty_buffer) {
        Some(existing) => provider
            .union_gpu(&existing, &incoming)
            .map(non_empty_buffer),
        None => Ok(Some(incoming)),
    }
}

fn non_empty_buffer(buffer: CudaBuffer) -> Option<CudaBuffer> {
    if buffer.cached_row_count() == Some(0) || buffer.is_empty() {
        None
    } else {
        Some(buffer)
    }
}

fn buffer_rows(buffer: &CudaBuffer) -> u64 {
    buffer
        .cached_row_count()
        .map(u64::from)
        .unwrap_or_else(|| buffer.num_rows())
}

fn ensure_schema_type_compatible(expected: &Schema, actual: &Schema) -> Result<()> {
    if expected.arity() != actual.arity() {
        return Err(XlogError::Execution(format!(
            "Expected {} columns, got {}",
            expected.arity(),
            actual.arity()
        )));
    }
    for i in 0..expected.arity() {
        let exp = expected.column_type(i).ok_or_else(|| {
            XlogError::Execution(format!("Missing expected type for column {}", i))
        })?;
        let act = actual
            .column_type(i)
            .ok_or_else(|| XlogError::Execution(format!("Missing actual type for column {}", i)))?;
        if exp != act {
            return Err(XlogError::Execution(format!(
                "Column {} type mismatch: expected {:?}, got {:?}",
                i, exp, act
            )));
        }
    }
    Ok(())
}

fn query_output_vars(Query { atom }: &Query) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for term in &atom.terms {
        for name in term.variables() {
            if seen.insert(name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// Epistemic-plan / EIR JSON dump
// --------------------------------------------------------------------------- //

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Deterministic 64-bit FNV-1a hash of a string (stable across runs/processes,
/// unlike `std::hash::DefaultHasher` which is randomized). Used as the stable
/// epistemic plan id so two dumps of the same plan compare equal.
fn fnv1a_64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Extract every `know`/`possible` literal (with negation) from a program's EIR.
/// Used to retain epistemic provenance when a Case-A recursive reduction lowers the
/// program to an ordinary executable plan.
fn collect_eir_epistemic_literals(program: &Program) -> Vec<xlog_ir::EirEpistemicLiteral> {
    let mut lits = Vec::new();
    if let Ok(eir) = xlog_logic::build_eir(program) {
        for rule in &eir.rules {
            for lit in &rule.body {
                if let xlog_ir::EirBodyLiteral::Epistemic(e) = lit {
                    lits.push(e.clone());
                }
            }
        }
    }
    lits
}

/// JSON summary for an epistemic source that reduced to a high-level recursive
/// execution plan without single-pass epistemic GPU candidate units. Case-A/B
/// stratified reductions use the ordinary semi-naive engine; cyclic negated-modal
/// reductions use the GPU-backed WFS alternating-fixpoint plan. In both cases the
/// modal literals and the fail-closed GPU execution policy are recorded.
fn epistemic_provenance_summary_json(
    plan_kind: &str,
    prov: &EpistemicProvenance,
    max_iterations: Option<usize>,
    wfs: Option<&EpistemicWfsGpuPlan>,
) -> String {
    let literals = prov
        .literals
        .iter()
        .map(epistemic_literal_json)
        .collect::<Vec<_>>()
        .join(",");
    let wfs_fixed_relations = wfs
        .map(wfs_fixed_relations_json)
        .unwrap_or_else(|| "null".to_string());
    let wfs_convergence_predicates = wfs
        .map(wfs_convergence_predicates_json)
        .unwrap_or_else(|| "null".to_string());
    let wfs_gpu_passes = if wfs.is_some() {
        "[\"overapprox\",\"lower\",\"upper\"]"
    } else {
        "null"
    };
    let body = format!(
        "{{\"plan_kind\":\"{}\",\"reduction\":\"{}\",\
\"epistemic_literals\":[{}],\"units\":[],\"max_iterations\":{},\
\"wfs_fixed_relations\":{},\"wfs_convergence_predicates\":{},\
\"wfs_gpu_passes\":{},\"execution_backend\":\"{}\",\
\"fallback_policy\":\"{}\"}}",
        json_escape(plan_kind),
        json_escape(prov.reduction),
        literals,
        max_iterations
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        wfs_fixed_relations,
        wfs_convergence_predicates,
        wfs_gpu_passes,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{:016x}\",\"plan_kind\":\"{}\",\
\"reduction\":\"{}\",\"epistemic_literals\":[{}],\"units\":[],\
\"max_iterations\":{},\"wfs_fixed_relations\":{},\
\"wfs_convergence_predicates\":{},\"wfs_gpu_passes\":{},\"execution_backend\":\"{}\",\
\"fallback_policy\":\"{}\"}}",
        plan_id,
        json_escape(plan_kind),
        json_escape(prov.reduction),
        literals,
        max_iterations
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        wfs_fixed_relations,
        wfs_convergence_predicates,
        wfs_gpu_passes,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    )
}

fn g91_compatibility_summary_json(
    plan_kind: &str,
    provenance: &EpistemicProvenance,
    plan: &EpistemicG91CompatibilityGpuPlan,
) -> String {
    let literals = provenance
        .literals
        .iter()
        .map(epistemic_literal_json)
        .collect::<Vec<_>>()
        .join(",");
    let snapshots = plan
        .snapshot_relations
        .iter()
        .map(|(source, snapshot)| {
            format!("\"{}\":\"{}\"", json_escape(source), json_escape(snapshot))
        })
        .collect::<Vec<_>>()
        .join(",");
    let convergence = plan
        .convergence_predicates
        .iter()
        .map(|predicate| format!("\"{}\"", json_escape(predicate)))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        "{{\"plan_kind\":\"{}\",\"reduction\":\"{}\",\
\"epistemic_literals\":[{}],\"units\":[],\"max_iterations\":{},\
\"snapshot_relations\":{{{}}},\"convergence_predicates\":[{}],\
\"gpu_passes\":[\"upper_bound\",\"refinement\"],\
\"execution_backend\":\"{}\",\"fallback_policy\":\"{}\"}}",
        json_escape(plan_kind),
        json_escape(provenance.reduction),
        literals,
        plan.max_iterations,
        snapshots,
        convergence,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{plan_id:016x}\",\"plan_kind\":\"{}\",\
\"reduction\":\"{}\",\"epistemic_literals\":[{}],\"units\":[],\
\"max_iterations\":{},\"snapshot_relations\":{{{}}},\
\"convergence_predicates\":[{}],\
\"gpu_passes\":[\"upper_bound\",\"refinement\"],\
\"execution_backend\":\"{}\",\"fallback_policy\":\"{}\"}}",
        json_escape(plan_kind),
        json_escape(provenance.reduction),
        literals,
        plan.max_iterations,
        snapshots,
        convergence,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    )
}

fn wfs_fixed_relations_json(wfs: &EpistemicWfsGpuPlan) -> String {
    let mut sources: BTreeSet<&str> = BTreeSet::new();
    for source in wfs.upper_fixed_names.keys() {
        sources.insert(source.as_str());
    }
    for source in wfs.lower_fixed_names.keys() {
        sources.insert(source.as_str());
    }
    let entries = sources
        .into_iter()
        .map(|source| {
            let upper = wfs
                .upper_fixed_names
                .get(source)
                .map(String::as_str)
                .unwrap_or("");
            let lower = wfs
                .lower_fixed_names
                .get(source)
                .map(String::as_str)
                .unwrap_or("");
            format!(
                "\"{}\":{{\"upper\":\"{}\",\"lower\":\"{}\"}}",
                json_escape(source),
                json_escape(upper),
                json_escape(lower)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{entries}}}")
}

fn wfs_convergence_predicates_json(wfs: &EpistemicWfsGpuPlan) -> String {
    let entries = wfs
        .intensional_predicates
        .iter()
        .map(|pred| format!("\"{}\"", json_escape(pred)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{entries}]")
}

fn epistemic_literal_json(lit: &xlog_ir::EirEpistemicLiteral) -> String {
    let op = match lit.op {
        xlog_ir::EirEpistemicOp::Know => "know",
        xlog_ir::EirEpistemicOp::Possible => "possible",
    };
    format!(
        "{{\"op\":\"{}\",\"negated\":{},\"predicate\":\"{}\",\"arity\":{}}}",
        op,
        lit.negated,
        json_escape(&lit.atom.predicate),
        lit.atom.arity
    )
}

fn epistemic_execution_backend_json(backend: xlog_ir::EpistemicExecutionBackend) -> &'static str {
    match backend {
        xlog_ir::EpistemicExecutionBackend::Gpu => "gpu",
    }
}

fn epistemic_fallback_policy_json(policy: xlog_ir::EpistemicFallbackPolicy) -> &'static str {
    match policy {
        xlog_ir::EpistemicFallbackPolicy::RejectUnsupported => "reject_unsupported",
    }
}

fn epistemic_gpu_plan_json(plan: &xlog_ir::EpistemicGpuPlan) -> String {
    let mode = match plan.mode {
        xlog_ir::EirEpistemicMode::G91 => "g91",
        xlog_ir::EirEpistemicMode::Faeel => "faeel",
    };
    let literals = plan
        .epistemic_literals
        .iter()
        .map(epistemic_literal_json)
        .collect::<Vec<_>>()
        .join(",");
    let phases = plan
        .required_phases
        .iter()
        .map(|p| format!("\"{:?}\"", p))
        .collect::<Vec<_>>()
        .join(",");
    let kernels = plan
        .required_kernel_phases
        .iter()
        .map(|p| format!("\"{:?}\"", p))
        .collect::<Vec<_>>()
        .join(",");
    let constraints = plan
        .constraints
        .iter()
        .map(|c| {
            let idx = c
                .literal_indices
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"constraint_index\":{},\"literal_indices\":[{}]}}",
                c.constraint_index, idx
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let reductions = plan
        .reductions
        .iter()
        .map(|r| {
            format!(
                "{{\"rule_index\":{},\"head\":\"{}\",\"public_head_arity\":{},\"relational_body_atoms\":{}}}",
                r.rule_index,
                json_escape(&r.head_predicate),
                r.public_head_arity,
                r.relational_body_atoms
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"mode\":\"{}\",\"epistemic_literals\":[{}],\"required_phases\":[{}],\
\"required_kernel_phases\":[{}],\"constraints\":[{}],\"reductions\":[{}],\
\"execution_backend\":\"{}\",\"fallback_policy\":\"{}\"}}",
        mode,
        literals,
        phases,
        kernels,
        constraints,
        reductions,
        epistemic_execution_backend_json(plan.execution_backend),
        epistemic_fallback_policy_json(plan.fallback_policy)
    )
}

fn epistemic_plan_summary_json(
    plan_kind: &str,
    gpu_plans: &[(String, &xlog_ir::EpistemicGpuPlan)],
    has_ordinary_post: bool,
) -> String {
    let mut units = gpu_plans
        .iter()
        .map(|(label, plan)| {
            format!(
                "{{\"unit\":\"{}\",\"plan\":{}}}",
                json_escape(label),
                epistemic_gpu_plan_json(plan)
            )
        })
        .collect::<Vec<_>>();
    if has_ordinary_post {
        units.push(
            "{\"unit\":\"ordinary_post\",\"stage_kind\":\"ordinary_closure_and_constraints\"}"
                .to_string(),
        );
    }
    let units = units.join(",");
    // Canonical body (without the id) hashed for the stable plan id.
    let body = format!(
        "{{\"plan_kind\":\"{}\",\"units\":[{}],\"execution_backend\":\"{}\",\"fallback_policy\":\"{}\"}}",
        json_escape(plan_kind),
        units,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{:016x}\",\"plan_kind\":\"{}\",\"units\":[{}],\"execution_backend\":\"{}\",\"fallback_policy\":\"{}\"}}",
        plan_id,
        json_escape(plan_kind),
        units,
        epistemic_execution_backend_json(xlog_ir::EpistemicExecutionBackend::Gpu),
        epistemic_fallback_policy_json(xlog_ir::EpistemicFallbackPolicy::RejectUnsupported)
    )
}

#[cfg(test)]
fn finish_test_provider_setup<T>(provider: Result<T>, require_cuda: bool) -> Option<T> {
    match provider {
        Ok(provider) => Some(provider),
        Err(error) if require_cuda => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA provider construction failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: no CUDA device available ({error})");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use xlog_core::{symbol, MemoryBudget, ScalarType};
    use xlog_cuda::cuda_graph::CudaGraphNodeKind;
    use xlog_ir::RirNode;
    use xlog_runtime::resident_graph::{
        ResidentGraphDeclineReason, ResidentGraphRouteCertificate, ResidentGraphSchemaCatalog,
        ResidentGraphSelectionKind,
    };

    fn ground_term_encoding_test_provider() -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            Ok(Arc::new(
                xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
                    .build()?,
            ))
        })();

        finish_test_provider_setup(
            provider,
            std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1"),
        )
    }

    fn pinned_corpus_test_provider() -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            Ok(Arc::new(
                xlog_cuda::CudaProviderBuilder::new(
                    0,
                    MemoryBudget::with_limit(2 * 1024 * 1024 * 1024),
                )
                .build()?,
            ))
        })();
        finish_test_provider_setup(
            provider,
            std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1"),
        )
    }

    const PINNED_CORPUS_SHA: &str = "74f2895486737b4caa42229389d309994e7ad3ea";
    const RESIDENT_ENV_NAMES: [&str; 3] = [
        "XLOG_DISABLE_RESIDENT_RECURSION",
        "XLOG_REQUIRE_RESIDENT_RECURSION",
        RESIDENT_LATENCY_DIAGNOSTICS_ENV,
    ];

    fn git_output(corpus: &std::path::Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(corpus)
            .args(args)
            .output()
            .unwrap_or_else(|error| panic!("git {args:?} failed to start: {error}"));
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("git output must be UTF-8")
    }

    fn assert_exact_clean_corpus(corpus: &std::path::Path) {
        assert_eq!(
            git_output(corpus, &["rev-parse", "HEAD"]).trim(),
            PINNED_CORPUS_SHA
        );
        assert!(
            git_output(
                corpus,
                &["status", "--porcelain=v1", "--untracked-files=all"]
            )
            .is_empty(),
            "pinned corpus must have no tracked or untracked modifications"
        );
        assert!(
            git_output(corpus, &["diff", "--no-ext-diff", "--submodule=diff"]).is_empty(),
            "pinned corpus working tree must match HEAD"
        );
        assert!(
            git_output(
                corpus,
                &["diff", "--cached", "--no-ext-diff", "--submodule=diff"]
            )
            .is_empty(),
            "pinned corpus index must match HEAD"
        );
        let submodules = git_output(corpus, &["submodule", "status", "--recursive"]);
        assert!(
            submodules.lines().all(|line| line.starts_with(' ')),
            "every recursive submodule must be initialized at its recorded commit: {submodules}"
        );
        let clean_submodules = std::process::Command::new("git")
            .arg("-C")
            .arg(corpus)
            .args([
                "submodule",
                "foreach",
                "--quiet",
                "--recursive",
                "test -z \"$(git status --porcelain=v1 --untracked-files=all)\"",
            ])
            .status()
            .expect("recursive submodule cleanliness command must start");
        assert!(
            clean_submodules.success(),
            "recursive submodule checkout is dirty"
        );
    }

    struct ResidentEnvGuard {
        old: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl ResidentEnvGuard {
        fn set(active: &[(&'static str, &'static str)]) -> Self {
            let old = RESIDENT_ENV_NAMES
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect();
            for name in RESIDENT_ENV_NAMES {
                // SAFETY: every test using this helper holds `resident_env_lock`.
                unsafe { std::env::remove_var(name) };
            }
            for (name, value) in active {
                // SAFETY: every test using this helper holds `resident_env_lock`.
                unsafe { std::env::set_var(name, value) };
            }
            Self { old }
        }
    }

    impl Drop for ResidentEnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.old.drain(..) {
                match value {
                    Some(value) => {
                        // SAFETY: every test using this helper holds `resident_env_lock`.
                        unsafe { std::env::set_var(name, value) };
                    }
                    None => {
                        // SAFETY: every test using this helper holds `resident_env_lock`.
                        unsafe { std::env::remove_var(name) };
                    }
                }
            }
        }
    }

    fn resident_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn resident_selection_rejects_malformed_boolean_environment() {
        let _lock = resident_env_lock().lock().unwrap();
        let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "sometimes")]);
        assert!(matches!(
            ResidentSelectionMode::from_env(),
            Err(XlogError::Configuration { ref name, .. })
                if name == "XLOG_REQUIRE_RESIDENT_RECURSION"
        ));
    }

    #[test]
    fn resident_selection_defaults_to_auto_and_honors_explicit_policy() {
        let _lock = resident_env_lock().lock().unwrap();

        let automatic = ResidentEnvGuard::set(&[]);
        assert_eq!(
            ResidentSelectionMode::from_env().unwrap(),
            ResidentSelectionMode::Auto
        );
        drop(automatic);

        let disabled = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
        assert_eq!(
            ResidentSelectionMode::from_env().unwrap(),
            ResidentSelectionMode::Disabled
        );
        drop(disabled);

        let _required = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
        assert_eq!(
            ResidentSelectionMode::from_env().unwrap(),
            ResidentSelectionMode::Require
        );
    }

    #[test]
    fn resident_selection_rejects_conflicting_explicit_policy() {
        let _lock = resident_env_lock().lock().unwrap();
        let _env = ResidentEnvGuard::set(&[
            ("XLOG_DISABLE_RESIDENT_RECURSION", "1"),
            ("XLOG_REQUIRE_RESIDENT_RECURSION", "1"),
        ]);
        assert!(matches!(
            ResidentSelectionMode::from_env(),
            Err(XlogError::Execution(ref message))
                if message.contains("mutually exclusive")
        ));
    }

    fn corpus_program(corpus: &std::path::Path) -> Result<LogicProgram> {
        let entry = corpus.join("scenarios/acceptance/issue1/q01_blind.xlog");
        let source = std::fs::read_to_string(&entry).map_err(|error| {
            XlogError::Execution(format!("failed to read {}: {error}", entry.display()))
        })?;
        let resolver = xlog_logic::compile::load_modules(&entry, vec![corpus.join("programs")])
            .map_err(|error| XlogError::Compilation(error.to_string()))?;
        LogicProgram::compile_with_resolver(&source, &resolver)
    }

    fn schema_catalog(program: &LogicProgram) -> ResidentGraphSchemaCatalog {
        ResidentGraphSchemaCatalog::from_named_schemas(program.rel_ids.iter().filter_map(
            |(name, rel)| {
                program
                    .schemas
                    .get(name)
                    .cloned()
                    .map(|schema| (name.clone(), *rel, schema))
            },
        ))
    }

    struct RouteWalk<'a> {
        scc_index: usize,
        rule_index: usize,
        recursive: bool,
        identities: &'a mut BTreeSet<String>,
    }

    impl RouteWalk<'_> {
        fn visit(&mut self, node: &RirNode, path: &str) {
            assert!(
                self.identities.insert(format!(
                    "scc={};rule={};recursive={};path={path}",
                    self.scc_index, self.rule_index, self.recursive
                )),
                "route occurrence paths must be unique"
            );
            match node {
                RirNode::Unit | RirNode::Scan { .. } | RirNode::TensorMaskedJoin { .. } => {}
                RirNode::Filter { input, .. }
                | RirNode::Project { input, .. }
                | RirNode::GroupBy { input, .. }
                | RirNode::Distinct { input, .. } => self.visit(input, &format!("{path}/input")),
                RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                    self.visit(left, &format!("{path}/left"));
                    self.visit(right, &format!("{path}/right"));
                }
                RirNode::ChainJoin {
                    left,
                    right,
                    fallback,
                    ..
                } => {
                    self.visit(left, &format!("{path}/primary/left"));
                    self.visit(right, &format!("{path}/primary/right"));
                    self.visit(fallback, &format!("{path}/alternative/captured_fallback"));
                }
                RirNode::Union { inputs } => {
                    for (index, input) in inputs.iter().enumerate() {
                        self.visit(input, &format!("{path}/input[{index}]"));
                    }
                }
                RirNode::Fixpoint {
                    base, recursive, ..
                } => {
                    self.visit(base, &format!("{path}/base"));
                    self.visit(recursive, &format!("{path}/recursive"));
                }
                RirNode::MultiWayJoin {
                    inputs, fallback, ..
                } => {
                    for (index, input) in inputs.iter().enumerate() {
                        self.visit(input, &format!("{path}/primary/input[{index}]"));
                    }
                    self.visit(fallback, &format!("{path}/alternative/captured_fallback"));
                }
            }
        }
    }

    fn independent_route_identities(plan: &ExecutionPlan) -> BTreeSet<String> {
        let mut identities = BTreeSet::new();
        for (scc_index, scc) in plan.sccs.iter().enumerate() {
            let rules = plan
                .rules_by_scc
                .get(scc_index)
                .unwrap_or_else(|| panic!("missing rule vector for SCC {scc_index}"));
            for (rule_index, rule) in rules.iter().enumerate() {
                RouteWalk {
                    scc_index,
                    rule_index,
                    recursive: scc.is_recursive,
                    identities: &mut identities,
                }
                .visit(&rule.body, "primary/root");
                let rule_identity = format!(
                    "scc={scc_index};rule={rule_index};head={};schema={:#?}",
                    rule.head, rule.meta.schema
                );
                identities.insert(format!("{rule_identity};implicit=rule_result_union"));
                identities.insert(format!("{rule_identity};implicit=full_row_dedup"));
                if scc.is_recursive {
                    identities.insert(format!("{rule_identity};implicit=novel_tuple_difference"));
                    identities.insert(format!("{rule_identity};implicit=device_convergence"));
                }
            }
        }
        identities
    }

    fn op_count(stats: &ExecutionStats, name: &str) -> usize {
        stats
            .strata
            .iter()
            .flat_map(|stratum| &stratum.ops)
            .filter(|op| op.op_name == name)
            .count()
    }

    fn strata_op_profile(stats: &ExecutionStats) -> BTreeMap<String, (usize, u64, u64)> {
        let mut profile = BTreeMap::new();
        for op in stats.strata.iter().flat_map(|stratum| &stratum.ops) {
            let entry = profile.entry(op.op_name.clone()).or_insert((0, 0, 0));
            entry.0 += 1;
            entry.1 += op.input_rows;
            entry.2 += op.output_rows;
        }
        profile
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct HostQuerySnapshot {
        relation_name: String,
        columns: Vec<String>,
        sort_labels: Vec<String>,
        schema: Schema,
        rows: Vec<Vec<u64>>,
    }

    fn snapshot_query_results(
        provider: &CudaKernelProvider,
        result: &LogicEvalResult,
    ) -> Result<Vec<HostQuerySnapshot>> {
        result
            .queries
            .iter()
            .map(|query| {
                let row_count = provider.device_row_count(&query.buffer)?;
                let mut columns = Vec::with_capacity(query.buffer.schema().arity());
                for index in 0..query.buffer.schema().arity() {
                    let ty = query
                        .buffer
                        .schema()
                        .column_type(index)
                        .expect("schema arity checked");
                    let values = match ty {
                        ScalarType::U32 | ScalarType::Symbol => provider
                            .download_column::<u32>(&query.buffer, index)?
                            .into_iter()
                            .map(u64::from)
                            .collect(),
                        ScalarType::U64 => provider.download_column::<u64>(&query.buffer, index)?,
                        ScalarType::I32 => provider
                            .download_column::<i32>(&query.buffer, index)?
                            .into_iter()
                            .map(|value| value as i64 as u64)
                            .collect(),
                        ScalarType::I64 => provider
                            .download_column::<i64>(&query.buffer, index)?
                            .into_iter()
                            .map(|value| value as u64)
                            .collect(),
                        ScalarType::F32 => provider
                            .download_column::<f32>(&query.buffer, index)?
                            .into_iter()
                            .map(|value| u64::from(value.to_bits()))
                            .collect(),
                        ScalarType::F64 => provider
                            .download_column::<f64>(&query.buffer, index)?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect(),
                        ScalarType::Bool => provider
                            .download_column::<u8>(&query.buffer, index)?
                            .into_iter()
                            .map(u64::from)
                            .collect(),
                    };
                    if values.len() != row_count {
                        return Err(XlogError::Execution(format!(
                            "query column {index} has {} rows but metadata reports {row_count}",
                            values.len()
                        )));
                    }
                    columns.push(values);
                }
                let mut rows = (0..row_count)
                    .map(|row| columns.iter().map(|column| column[row]).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                rows.sort_unstable();
                Ok(HostQuerySnapshot {
                    relation_name: query.relation_name.clone(),
                    columns: query.columns.clone(),
                    sort_labels: query.sort_labels.clone(),
                    schema: query.buffer.schema().clone(),
                    rows,
                })
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a serialized release-mode CUDA acceptance run"]
    fn resident_semantic_profile_excludes_noop_recursive_variants() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred seed(u32).
                pred dead(u32).
                pred a(u32).
                pred b(u32).

                seed(1).
                a(X) :- seed(X).
                b(X) :- dead(X).
                a(X) :- b(X), X = 1.
                b(X) :- a(X), X = 1.

                ?- a(X).
                ?- b(X).
            "#,
        )?;
        let empty_recursive_inputs = || -> Result<HashMap<String, CudaBuffer>> {
            Ok(HashMap::from([
                (
                    "a".to_string(),
                    provider.create_empty_buffer(program.schema("a").expect("a schema").clone())?,
                ),
                (
                    "b".to_string(),
                    provider.create_empty_buffer(program.schema("b").expect("b schema").clone())?,
                ),
            ]))
        };
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), empty_recursive_inputs()?, true)?
        };
        let expected = snapshot_query_results(provider.as_ref(), &baseline)?;
        let baseline_stats = baseline.stats.as_ref().expect("baseline profile");
        let baseline_scans = op_count(baseline_stats, "scan");
        let baseline_filters = op_count(baseline_stats, "filter");
        let expected_semantic_scans =
            baseline_scans as u64 + baseline_stats.chain_fallback_scan_equivalents;
        let expected_semantic_filters =
            baseline_filters as u64 + baseline_stats.chain_fallback_filter_equivalents;
        drop(baseline);

        let resident = {
            let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), empty_recursive_inputs()?, true)?
        };
        assert_eq!(
            snapshot_query_results(provider.as_ref(), &resident)?,
            expected
        );
        let resident_stats = resident.stats.as_ref().expect("resident profile");
        let graph = resident_stats
            .resident_graph
            .as_ref()
            .expect("resident telemetry");
        assert_eq!(graph.semantic_scan_invocations, expected_semantic_scans);
        assert_eq!(graph.semantic_filter_invocations, expected_semantic_filters);
        assert_eq!(
            op_count(resident_stats, "scan") as u64,
            graph.device_scan_invocations
        );
        assert_eq!(
            op_count(resident_stats, "filter") as u64,
            graph.device_filter_invocations
        );
        assert!(graph.device_scan_invocations >= graph.semantic_scan_invocations);
        assert!(graph.device_filter_invocations >= graph.semantic_filter_invocations);
        assert!(
            graph.device_scan_invocations > graph.semantic_scan_invocations
                || graph.device_filter_invocations > graph.semantic_filter_invocations,
            "the witness must schedule at least one empty-delta recursive variant"
        );
        Ok(())
    }

    #[test]
    fn resident_latency_phase_accounting_reports_only_unmeasured_host_work() {
        assert_eq!(resident_latency_unattributed_ns(100, &[10, 20, 30]), 40);
        assert_eq!(resident_latency_unattributed_ns(50, &[30, 30]), 0);
    }

    #[test]
    fn resident_certification_cache_is_eager_clone_shared_thread_safe_and_compile_isolated(
    ) -> Result<()> {
        let source = r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#;
        let program = LogicProgram::compile(source)?;
        assert_eq!(program.resident_certification_initializations(), 1);

        let barrier = Arc::new(std::sync::Barrier::new(8));
        let certified = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    let clone = program.clone();
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        clone.resident_certified_plan()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("resident certification worker"))
                .collect::<Result<Vec<_>>>()
        })?;
        assert_eq!(program.resident_certification_initializations(), 1);
        assert!(certified
            .iter()
            .all(|candidate| Arc::ptr_eq(&certified[0], candidate)));

        let outcome_program = LogicProgram::compile(source)?;
        assert_eq!(outcome_program.resident_certification_initializations(), 1);
        let (seeded, cache_was_warm, initialized_here) =
            outcome_program.resident_certified_plan_with_outcome()?;
        assert!(cache_was_warm);
        assert!(!initialized_here);
        let (warm, cache_was_warm, initialized_here) =
            outcome_program.resident_certified_plan_with_outcome()?;
        assert!(cache_was_warm);
        assert!(!initialized_here);
        assert!(Arc::ptr_eq(&seeded, &warm));

        let fresh = LogicProgram::compile(source)?;
        assert_eq!(fresh.resident_certification_initializations(), 1);
        let fresh_certified = fresh.resident_certified_plan()?;
        assert_eq!(fresh.resident_certification_initializations(), 1);
        assert!(!Arc::ptr_eq(&certified[0], &fresh_certified));
        Ok(())
    }

    #[test]
    fn resident_certification_retains_only_query_and_constraint_dependencies() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred base(u32).
                pred edge(u32, u32).
                pred reachable(u32).
                pred audited(u32).
                pred disconnected_seed(u32).
                pred disconnected(u32).

                base(1).
                edge(1, 2).
                reachable(X) :- base(X).
                reachable(Y) :- reachable(X), edge(X, Y).
                audited(X) :- base(X).
                disconnected(X) :- disconnected_seed(X).

                :- audited(99).
                ?- reachable(X).
            "#,
        )?;

        let full = program.ordinary_plan("resident reachability test")?;
        let full_heads = full
            .rules_by_scc
            .iter()
            .flatten()
            .map(|rule| rule.head.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(full_heads.contains("disconnected"));

        let certified = program.resident_certified_plan()?;
        let resident = certified.plan();
        let resident_heads = resident
            .rules_by_scc
            .iter()
            .flatten()
            .map(|rule| rule.head.as_str())
            .collect::<std::collections::BTreeSet<_>>();

        assert!(resident_heads.contains("reachable"));
        assert!(resident_heads.contains("audited"));
        assert!(resident_heads.contains("__xlog_constraint_0"));
        assert!(resident_heads.contains("__xlog_query_0"));
        assert!(!resident_heads.contains("disconnected"));
        assert!(resident.rules_by_scc.len() < full.rules_by_scc.len());
        assert!(resident
            .sccs
            .iter()
            .enumerate()
            .all(|(index, scc)| scc.id == index as u32));
        assert!(resident
            .strata
            .iter()
            .flat_map(|stratum| &stratum.sccs)
            .all(|scc| (*scc as usize) < resident.sccs.len()));
        assert_eq!(resident.generated_query_rules.len(), 1);
        let query = &resident.generated_query_rules[0];
        assert_eq!(query.query_index, 0);
        assert_eq!(
            resident.rules_by_scc[query.scc_index][query.rule_index].head,
            "__xlog_query_0"
        );

        let full_heads_after_certification = full
            .rules_by_scc
            .iter()
            .flatten()
            .map(|rule| rule.head.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(full_heads_after_certification.contains("disconnected"));
        Ok(())
    }

    #[test]
    fn resident_dependency_closure_fails_closed_on_missing_duplicate_or_ambiguous_proof(
    ) -> Result<()> {
        #[derive(Debug, PartialEq, Eq)]
        struct ResidentPlanStructure {
            sccs: Vec<(u32, Vec<String>)>,
            strata: Vec<(u32, Vec<u32>)>,
            rule_heads: Vec<Vec<String>>,
            generated_query_rules: Vec<(usize, usize, usize)>,
        }

        fn plan_structure(plan: &ExecutionPlan) -> ResidentPlanStructure {
            ResidentPlanStructure {
                sccs: plan
                    .sccs
                    .iter()
                    .map(|scc| (scc.id, scc.predicates.clone()))
                    .collect(),
                strata: plan
                    .strata
                    .iter()
                    .map(|stratum| (stratum.id, stratum.sccs.clone()))
                    .collect(),
                rule_heads: plan
                    .rules_by_scc
                    .iter()
                    .map(|rules| rules.iter().map(|rule| rule.head.clone()).collect())
                    .collect(),
                generated_query_rules: plan
                    .generated_query_rules
                    .iter()
                    .map(|query| (query.query_index, query.scc_index, query.rule_index))
                    .collect(),
            }
        }

        let program = LogicProgram::compile(
            r#"
                pred input(u32).
                pred output(u32).
                input(1).
                output(X) :- input(X).
                ?- output(X).
            "#,
        )?;
        let full = program.ordinary_plan("resident fail-closed test")?;

        let mut missing = full.clone();
        missing.generated_query_rules.clear();
        assert_eq!(
            plan_structure(&program.resident_dependency_closed_plan(&missing)),
            plan_structure(&missing)
        );

        let mut duplicate = full.clone();
        duplicate
            .generated_query_rules
            .push(duplicate.generated_query_rules[0].clone());
        assert_eq!(
            plan_structure(&program.resident_dependency_closed_plan(&duplicate)),
            plan_structure(&duplicate)
        );

        let mut ambiguous = full.clone();
        let duplicated_rule = ambiguous
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == "output")
            .expect("output rule")
            .clone();
        let duplicate_scc = ambiguous.sccs.len() as u32;
        ambiguous.sccs.push(xlog_ir::Scc {
            id: duplicate_scc,
            predicates: vec!["output".into()],
            is_recursive: false,
        });
        ambiguous.rules_by_scc.push(vec![duplicated_rule]);
        ambiguous.strata.push(xlog_ir::Stratum {
            id: ambiguous.strata.len() as u32,
            sccs: vec![duplicate_scc],
        });
        assert_eq!(
            plan_structure(&program.resident_dependency_closed_plan(&ambiguous)),
            plan_structure(&ambiguous)
        );

        let mut missing_nonroot_rel_id = full.clone();
        missing_nonroot_rel_id
            .rules_by_scc
            .iter_mut()
            .flatten()
            .find(|rule| rule.head == "output")
            .expect("output rule")
            .head = "missing_nonroot_rel_id".into();
        assert!(program
            .try_resident_dependency_closed_plan(&missing_nonroot_rel_id)
            .is_none());
        assert_eq!(
            plan_structure(&program.resident_dependency_closed_plan(&missing_nonroot_rel_id)),
            plan_structure(&missing_nonroot_rel_id)
        );

        let no_query = LogicProgram::compile(
            r#"
                pred input(u32).
                pred output(u32).
                input(1).
                output(X) :- input(X).
            "#,
        )?;
        let no_query_full = no_query.ordinary_plan("resident no-query test")?;
        assert_eq!(
            plan_structure(&no_query.resident_dependency_closed_plan(no_query_full)),
            plan_structure(no_query_full)
        );
        Ok(())
    }

    #[test]
    fn compile_finalizer_preserves_and_replays_deterministic_certification_errors() -> Result<()> {
        let mut program = LogicProgram::compile(
            r#"
                pred output(u32).
                output(7).
                ?- output(X).
            "#,
        )?;
        program.reusable_state_identity = Arc::new(LogicProgramIdentity::new());
        let first = program
            .reusable_state_identity
            .get_or_init_resident_certification(|| -> Result<ResidentGraphCertifiedPlan> {
                Err(XlogError::Execution(
                    "deterministic certification failure".into(),
                ))
            })
            .expect_err("injected certification must fail");
        let program = program.finalize_compilation();
        let second = program
            .resident_certified_plan()
            .expect_err("cached certification must fail identically");

        assert_eq!(first.to_string(), second.to_string());
        assert_eq!(program.resident_certification_initializations(), 1);
        Ok(())
    }

    #[test]
    fn resident_certification_cache_is_ordinary_only_and_caches_declines_without_policy(
    ) -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        for policy in [
            &[][..],
            &[("XLOG_DISABLE_RESIDENT_RECURSION", "1")][..],
            &[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")][..],
        ] {
            let program = {
                let _env = ResidentEnvGuard::set(policy);
                LogicProgram::compile(
                    r#"
                        pred input(u32).
                        pred output(u32).
                        output(X) :- input(X).
                        ?- output(X).
                    "#,
                )?
            };
            assert_eq!(program.resident_certification_initializations(), 1);
        }

        let epistemic = LogicProgram::compile(
            r#"
                pred p(u32). pred q(u32).
                p(1). q(X) :- p(X), know p(X). ?- q(X).
            "#,
        )?;
        assert!(!matches!(epistemic.plan, LogicExecutionPlan::Ordinary(_)));
        assert!(epistemic.resident_certified_plan().is_err());
        assert_eq!(epistemic.resident_certification_initializations(), 0);

        let reduced_ordinary = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(u32).
                pred seed(u32, u32).
                pred trust(u32, u32).
                pred reach(u32, u32).
                node(1). node(2). node(3).
                seed(1, 2).
                reach(X, Y) :- seed(X, Y).
                reach(X, Z) :- reach(X, Y), trust(Y, Z).
                trust(2, 3) :- know reach(1, 2).
                trust(3, 1) :- know reach(3, 3).
                ?- reach(X, Y).
            "#,
        )?;
        assert!(matches!(
            reduced_ordinary.plan,
            LogicExecutionPlan::Ordinary(_)
        ));
        assert_eq!(reduced_ordinary.resident_certification_initializations(), 1);

        let unsupported = LogicProgram::compile(
            r#"
                pred unsupported(f64).
                unsupported(7.5).
                ?- unsupported(X).
            "#,
        )?;
        let first = unsupported.resident_certified_plan()?;
        let second = unsupported.resident_certified_plan()?;
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!first.certificate().is_supported());
        assert_eq!(unsupported.resident_certification_initializations(), 1);
        Ok(())
    }

    fn median_seconds(samples: &mut [f64]) -> f64 {
        assert!(!samples.is_empty());
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    #[test]
    #[ignore = "requires the exact external issue corpus checkout and CUDA"]
    fn pinned_corpus_prepares_resident_graph_without_launching_it() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let corpus = std::path::PathBuf::from(
            std::env::var("XLOG_PINNED_CORPUS_ROOT")
                .expect("XLOG_PINNED_CORPUS_ROOT must name the pinned corpus checkout"),
        );
        assert_exact_clean_corpus(&corpus);
        let program = corpus_program(&corpus)?;
        let plan = program.ordinary_plan("resident graph preflight")?;
        let certificate = ResidentGraphRouteCertificate::inspect(plan, &schema_catalog(&program))?;
        assert!(certificate.is_supported(), "{:#?}", certificate.declines());

        let Some(provider) = pinned_corpus_test_provider() else {
            return Ok(());
        };
        let resident_provider = Arc::clone(&provider);
        let executor =
            program.prepare_resident_executor(&resident_provider, HashMap::new(), false, plan)?;
        let runtime = resident_provider
            .memory()
            .runtime()
            .expect("canonical provider must own an async runtime");
        let graph_before = runtime.conditional_graph_stats();
        let allocated_before = resident_provider.memory().allocated_bytes();

        let prepared = executor
            .prepare_resident_graph(plan, &certificate, ResidentGraphPrepareOptions::default())
            .map_err(LogicProgram::resident_execution_error)?;
        let report = prepared.preflight_report();
        let graph_after = runtime.conditional_graph_stats();
        assert_eq!(graph_after.launches, graph_before.launches);
        assert_eq!(
            graph_after.terminal_synchronizations,
            graph_before.terminal_synchronizations
        );
        assert_eq!(
            resident_provider
                .memory()
                .allocated_bytes()
                .saturating_sub(allocated_before),
            report.tracked_device_allocation_bytes
        );
        assert!(report.relation_capacity > 0);
        assert_eq!(report.parent_graph_nodes, 5);
        assert_eq!(report.conditional_while_nodes, 2);
        assert_eq!(
            report.parent_graph_node_kinds,
            vec![
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
            ]
        );
        assert_eq!(
            report.conditional_body_node_kinds,
            vec![
                vec![CudaGraphNodeKind::Kernel],
                vec![CudaGraphNodeKind::Kernel],
            ]
        );
        assert_eq!(report.conditional_body_kernel_counts, vec![1, 1]);
        assert_eq!(report.hierarchical_graph_nodes, 7);
        eprintln!(
            "resident corpus preflight: capacity={} estimated_bytes={} available_bytes={} tracked_allocated_bytes={} parent_nodes={} conditional_while_nodes={}",
            report.relation_capacity,
            report.estimated_required_bytes,
            report.available_bytes_at_admission,
            report.tracked_device_allocation_bytes,
            report.parent_graph_nodes,
            report.conditional_while_nodes,
        );
        drop(prepared);
        assert_eq!(
            runtime.conditional_graph_stats().launches,
            graph_before.launches
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the exact external issue corpus checkout and CUDA"]
    fn pinned_corpus_certifies_and_runs_through_the_resident_production_path() -> Result<()> {
        fn fallback_scan_filter_counts(node: &RirNode) -> (usize, usize) {
            match node {
                RirNode::Unit | RirNode::TensorMaskedJoin { .. } => (0, 0),
                RirNode::Scan { .. } => (1, 0),
                RirNode::Filter { input, .. } => {
                    let (scans, filters) = fallback_scan_filter_counts(input);
                    (scans, filters + 1)
                }
                RirNode::Project { input, .. }
                | RirNode::GroupBy { input, .. }
                | RirNode::Distinct { input, .. } => fallback_scan_filter_counts(input),
                RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                    let (left_scans, left_filters) = fallback_scan_filter_counts(left);
                    let (right_scans, right_filters) = fallback_scan_filter_counts(right);
                    (left_scans + right_scans, left_filters + right_filters)
                }
                RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
                    fallback_scan_filter_counts(fallback)
                }
                RirNode::Union { inputs } => inputs.iter().fold((0, 0), |total, input| {
                    let current = fallback_scan_filter_counts(input);
                    (total.0 + current.0, total.1 + current.1)
                }),
                RirNode::Fixpoint {
                    base, recursive, ..
                } => {
                    let (base_scans, base_filters) = fallback_scan_filter_counts(base);
                    let (recursive_scans, recursive_filters) =
                        fallback_scan_filter_counts(recursive);
                    (
                        base_scans + recursive_scans,
                        base_filters + recursive_filters,
                    )
                }
            }
        }

        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let corpus = std::path::PathBuf::from(
            std::env::var("XLOG_PINNED_CORPUS_ROOT")
                .expect("XLOG_PINNED_CORPUS_ROOT must name the pinned corpus checkout"),
        );
        assert_exact_clean_corpus(&corpus);
        let compile_and_certification_started = std::time::Instant::now();
        let program = corpus_program(&corpus)?;
        let compile_and_certification_seconds =
            compile_and_certification_started.elapsed().as_secs_f64();
        assert_eq!(
            program.resident_certification_initializations(),
            1,
            "ordinary compilation must eagerly seed one resident certification"
        );
        let plan = program.ordinary_plan("resident graph capability certificate")?;
        assert_eq!(plan.sccs.iter().filter(|scc| scc.is_recursive).count(), 2);
        assert_eq!(
            plan.sccs.iter().filter(|scc| !scc.is_recursive).count(),
            1_751
        );
        assert_eq!(plan.rules_by_scc.iter().map(Vec::len).sum::<usize>(), 4_559);
        let projected_reference_plan = program.resident_certified_plan()?.plan().clone();
        assert!(projected_reference_plan.sccs.len() < plan.sccs.len());
        assert!(
            projected_reference_plan
                .rules_by_scc
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                < plan.rules_by_scc.iter().map(Vec::len).sum::<usize>()
        );
        let chain_fallbacks = plan
            .rules_by_scc
            .iter()
            .enumerate()
            .flat_map(|(scc_index, rules)| {
                rules.iter().filter_map(move |rule| {
                    let RirNode::ChainJoin { fallback, .. } = &rule.body else {
                        return None;
                    };
                    let (scans, filters) = fallback_scan_filter_counts(fallback.as_ref());
                    Some((
                        scc_index,
                        plan.sccs[scc_index].is_recursive,
                        rule.head.clone(),
                        scans,
                        filters,
                    ))
                })
            })
            .collect::<Vec<_>>();
        let projected_chain_fallbacks = projected_reference_plan
            .rules_by_scc
            .iter()
            .enumerate()
            .flat_map(|(scc_index, rules)| {
                let is_recursive = projected_reference_plan.sccs[scc_index].is_recursive;
                rules.iter().filter_map(move |rule| {
                    let RirNode::ChainJoin { fallback, .. } = &rule.body else {
                        return None;
                    };
                    let (scans, filters) = fallback_scan_filter_counts(fallback);
                    Some((scc_index, is_recursive, rule.head.clone(), scans, filters))
                })
            })
            .collect::<Vec<_>>();
        eprintln!(
            "chain fallback inventory: routes={} scans={} filters={} details={chain_fallbacks:?}",
            chain_fallbacks.len(),
            chain_fallbacks.iter().map(|route| route.3).sum::<usize>(),
            chain_fallbacks.iter().map(|route| route.4).sum::<usize>()
        );

        // Prove occurrence completeness independently without duplicating the
        // certificate's bounded local-node encoding. `matches_plan` and the
        // resident-graph mutation tests verify that semantic binding.
        let expected_route_identities = independent_route_identities(plan);
        let certificate = ResidentGraphRouteCertificate::inspect(plan, &schema_catalog(&program))?;
        assert!(certificate.is_supported(), "{:#?}", certificate.declines());
        assert!(certificate.matches_plan(plan)?);
        let mut covered_structural_bindings = BTreeSet::new();
        let mut covered_physical_route_identities = BTreeSet::new();
        for descriptor in certificate.covered_route_descriptors() {
            if descriptor.starts_with("plan;") {
                covered_structural_bindings.insert(descriptor.clone());
            } else if descriptor.starts_with("scc=") {
                let identity = descriptor
                    .split_once(";node=")
                    .map_or(descriptor.as_str(), |(identity, _)| identity);
                covered_physical_route_identities.insert(identity.to_owned());
            } else {
                panic!("unknown resident certificate descriptor class: {descriptor}");
            }
        }
        assert!(!covered_structural_bindings.is_empty());
        assert!(!covered_physical_route_identities.is_empty());
        assert_eq!(covered_physical_route_identities, expected_route_identities);

        let Some(provider) = pinned_corpus_test_provider() else {
            return Ok(());
        };
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        let baseline_snapshot = snapshot_query_results(provider.as_ref(), &baseline)?;
        let baseline_stats = baseline.stats.as_ref().expect("baseline profile");
        let baseline_scans = op_count(baseline_stats, "scan");
        let baseline_filters = op_count(baseline_stats, "filter");
        let chain_fallback_scan_equivalents = baseline_stats.chain_fallback_scan_equivalents;
        let chain_fallback_filter_equivalents = baseline_stats.chain_fallback_filter_equivalents;
        let full_semantic_scans = baseline_scans as u64 + chain_fallback_scan_equivalents;
        let full_semantic_filters = baseline_filters as u64 + chain_fallback_filter_equivalents;
        assert_eq!(
            chain_fallback_scan_equivalents,
            chain_fallbacks
                .iter()
                .map(|route| route.3 as u64)
                .sum::<u64>()
        );
        assert_eq!(
            chain_fallback_filter_equivalents,
            chain_fallbacks
                .iter()
                .map(|route| route.4 as u64)
                .sum::<u64>()
        );
        eprintln!(
            "full-plan baseline operation profile: physical_scans={baseline_scans} physical_filters={baseline_filters} chain_fallback_scan_equivalents={chain_fallback_scan_equivalents} chain_fallback_filter_equivalents={chain_fallback_filter_equivalents} semantic_scans={full_semantic_scans} semantic_filters={full_semantic_filters} triangle={} four_cycle={} free_join={} factorized_delta={}",
            baseline_stats.wcoj_triangle_dispatch_count,
            baseline_stats.wcoj_4cycle_dispatch_count,
            baseline_stats.free_join_dispatch_count,
            baseline_stats.factorized_delta_dispatch_count
        );
        let mut projected_program = program.clone();
        projected_program.plan =
            LogicExecutionPlan::Ordinary(Box::new(projected_reference_plan.clone()));
        let projected_baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            projected_program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        assert_eq!(
            snapshot_query_results(provider.as_ref(), &projected_baseline)?,
            baseline_snapshot,
            "dependency-closed ordinary reference changed full-plan query semantics"
        );
        let projected_stats = projected_baseline
            .stats
            .as_ref()
            .expect("dependency-closed ordinary reference profile");
        let projected_scans = op_count(projected_stats, "scan");
        let projected_filters = op_count(projected_stats, "filter");
        assert_eq!(
            projected_stats.chain_fallback_scan_equivalents,
            projected_chain_fallbacks
                .iter()
                .map(|route| route.3 as u64)
                .sum::<u64>()
        );
        assert_eq!(
            projected_stats.chain_fallback_filter_equivalents,
            projected_chain_fallbacks
                .iter()
                .map(|route| route.4 as u64)
                .sum::<u64>()
        );
        let expected_semantic_scans =
            projected_scans as u64 + projected_stats.chain_fallback_scan_equivalents;
        let expected_semantic_filters =
            projected_filters as u64 + projected_stats.chain_fallback_filter_equivalents;
        eprintln!(
            "dependency-closed ordinary reference: physical_scans={projected_scans} physical_filters={projected_filters} chain_fallback_scan_equivalents={} chain_fallback_filter_equivalents={} semantic_scans={expected_semantic_scans} semantic_filters={expected_semantic_filters}",
            projected_stats.chain_fallback_scan_equivalents,
            projected_stats.chain_fallback_filter_equivalents,
        );
        drop(projected_baseline);
        assert!(
            baseline_scans >= 9_000,
            "unexpected baseline scan count: {baseline_scans}"
        );
        assert!(
            baseline_filters >= 7_000,
            "unexpected baseline filter count: {baseline_filters}"
        );
        drop(baseline);
        assert_eq!(
            program.resident_certification_initializations(),
            1,
            "external certificate audits and the disabled-resident baseline must reuse the compile-time certification"
        );

        const WARMUP_RUNS: usize = 2;
        const MEASURED_RUNS: usize = 5;
        let mut warmup_seconds = Vec::with_capacity(WARMUP_RUNS);
        let mut warmup_device_seconds = Vec::with_capacity(WARMUP_RUNS);
        let mut resident_seconds = Vec::with_capacity(MEASURED_RUNS);
        let mut device_seconds = Vec::with_capacity(MEASURED_RUNS);
        for run in 0..(WARMUP_RUNS + MEASURED_RUNS) {
            let started = std::time::Instant::now();
            let resident = {
                let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
                program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
            };
            assert_eq!(
                program.resident_certification_initializations(),
                1,
                "resident corpus run {run} must reuse the single cached certification"
            );
            let elapsed_seconds = started.elapsed().as_secs_f64();
            if run < WARMUP_RUNS {
                warmup_seconds.push(elapsed_seconds);
            } else {
                resident_seconds.push(elapsed_seconds);
            }
            assert_eq!(
                snapshot_query_results(provider.as_ref(), &resident)?,
                baseline_snapshot,
                "resident corpus run {run} changed query semantics"
            );

            let resident_stats = resident.stats.as_ref().expect("resident profile");
            let resident_physical_scans = op_count(resident_stats, "scan");
            let resident_physical_filters = op_count(resident_stats, "filter");
            let graph = resident_stats
                .resident_graph
                .as_ref()
                .expect("resident selection telemetry");
            eprintln!(
                "resident operation profile run {run}: semantic_scans={} semantic_filters={} physical_scans={resident_physical_scans} physical_filters={resident_physical_filters}",
                graph.semantic_scan_invocations, graph.semantic_filter_invocations
            );
            assert_eq!(graph.semantic_scan_invocations, expected_semantic_scans);
            assert_eq!(graph.semantic_filter_invocations, expected_semantic_filters);
            assert_eq!(
                resident_physical_scans as u64,
                graph.device_scan_invocations
            );
            assert_eq!(
                resident_physical_filters as u64,
                graph.device_filter_invocations
            );
            assert_eq!(
                graph.selection,
                ResidentGraphSelectionKind::ResidentConditionalGraph
            );
            assert_eq!(graph.conditional_graph_launches, 1);
            assert_eq!(graph.terminal_synchronizations, 1);
            assert_eq!(graph.host_iterations, 0);
            assert_eq!(graph.host_allocations, 0);
            assert_eq!(graph.host_status_injections, 0);
            assert_eq!(graph.deterministic_d2h_violations, 0);
            assert_eq!(graph.host_dispatched_scan_ops, 0);
            assert_eq!(graph.host_dispatched_filter_ops, 0);
            assert!(graph.device_scan_invocations >= graph.semantic_scan_invocations);
            assert!(graph.device_filter_invocations >= graph.semantic_filter_invocations);
            assert_eq!(
                graph.deferred_profile.timed_scan_filter_invocations,
                graph.device_scan_invocations + graph.device_filter_invocations
            );
            assert!(graph.deferred_profile.device_elapsed_ns > 0);
            assert_eq!(graph.deferred_profile.final_sync_misattributed_ns, 0);
            let device_elapsed_seconds =
                graph.deferred_profile.device_elapsed_ns as f64 / 1_000_000_000.0;
            if run < WARMUP_RUNS {
                warmup_device_seconds.push(device_elapsed_seconds);
            } else {
                device_seconds.push(device_elapsed_seconds);
            }
            assert_eq!(graph.core_transfers.tracked_htod_calls, 0);
            assert_eq!(graph.core_transfers.tracked_htod_bytes, 0);
            assert_eq!(graph.core_transfers.tracked_dtoh_calls, 0);
            assert_eq!(graph.core_transfers.tracked_dtoh_bytes, 0);
            assert_eq!(graph.core_transfers.provider_dtoh_calls, 0);
            assert_eq!(graph.core_transfers.untracked_metadata_dtoh_calls, 0);
            assert_eq!(graph.final_observation.dtoh_calls, 1);
            assert_eq!(
                graph.final_observation.dtoh_bytes,
                60 + 8 * graph.staged_store_mutations
            );
            assert_eq!(graph.final_observation.pinned_receipts, 1);
            let json = resident_stats.format_json();
            assert!(json.contains("\"resident_graph\""), "{json}");
            assert!(
                json.contains("\"selection\":\"resident_conditional_graph\""),
                "{json}"
            );
            assert!(
                json.contains(&format!(
                    "\"semantic_scan_invocations\":{expected_semantic_scans}"
                )),
                "{json}"
            );
            assert!(
                json.contains(&format!(
                    "\"semantic_filter_invocations\":{expected_semantic_filters}"
                )),
                "{json}"
            );
            drop(resident);
        }
        assert_eq!(program.resident_certification_initializations(), 1);
        assert_eq!(warmup_seconds.len(), WARMUP_RUNS);
        assert_eq!(resident_seconds.len(), MEASURED_RUNS);
        let compile_plus_first_resident_seconds =
            compile_and_certification_seconds + warmup_seconds[0];
        let max_seconds = warmup_seconds
            .iter()
            .chain(&resident_seconds)
            .copied()
            .fold(0.0_f64, f64::max);
        let mut sorted_resident_seconds = resident_seconds.clone();
        let median_seconds = median_seconds(&mut sorted_resident_seconds);
        eprintln!(
            "resident corpus latency: compile_and_certification_seconds={compile_and_certification_seconds:.6} compile_plus_first_resident_seconds={compile_plus_first_resident_seconds:.6} warmup_end_to_end_seconds={warmup_seconds:?} measured_end_to_end_seconds={resident_seconds:?} warmup_device_event_seconds={warmup_device_seconds:?} measured_device_event_seconds={device_seconds:?} median_measured_end_to_end_seconds={median_seconds:.6} max_all_end_to_end_seconds={max_seconds:.6}"
        );
        assert!(
            median_seconds <= 1.25,
            "five-run steady-state resident corpus median {median_seconds:.6}s exceeds 1.25s: {resident_seconds:?}"
        );
        assert!(
            max_seconds <= 1.75,
            "seven-run resident corpus max {max_seconds:.6}s exceeds 1.75s: warmup={warmup_seconds:?} measured={resident_seconds:?}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the exact external issue corpus checkout and serialized CUDA"]
    fn pinned_corpus_resident_latency_phase_diagnostic() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let corpus = std::path::PathBuf::from(
            std::env::var("XLOG_PINNED_CORPUS_ROOT")
                .expect("XLOG_PINNED_CORPUS_ROOT must name the pinned corpus checkout"),
        );
        assert_exact_clean_corpus(&corpus);
        let compile_started = std::time::Instant::now();
        let program = corpus_program(&corpus)?;
        eprintln!(
            "resident latency setup: compile_ns={}",
            u64::try_from(compile_started.elapsed().as_nanos()).unwrap_or(u64::MAX)
        );
        let Some(provider) = pinned_corpus_test_provider() else {
            return Ok(());
        };
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        let expected = snapshot_query_results(provider.as_ref(), &baseline)?;
        drop(baseline);

        RESIDENT_LATENCY_SAMPLE.store(0, Ordering::Relaxed);
        for run in 0..5 {
            let evaluate_started = std::time::Instant::now();
            let resident = {
                let _env = ResidentEnvGuard::set(&[
                    ("XLOG_REQUIRE_RESIDENT_RECURSION", "1"),
                    (RESIDENT_LATENCY_DIAGNOSTICS_ENV, "1"),
                ]);
                program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
            };
            let evaluate_return_ns =
                u64::try_from(evaluate_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            assert_eq!(
                snapshot_query_results(provider.as_ref(), &resident)?,
                expected,
                "resident latency diagnostic run {run} changed query semantics"
            );
            let graph = resident
                .stats
                .as_ref()
                .and_then(|stats| stats.resident_graph.as_ref())
                .expect("resident latency diagnostic telemetry");
            assert_eq!(
                graph.selection,
                ResidentGraphSelectionKind::ResidentConditionalGraph
            );
            let query_buffers = resident.queries.len();
            let manager_bytes_before_result_drop = provider.memory().allocated_bytes();
            let result_drop_started = std::time::Instant::now();
            drop(resident);
            let result_drop_ns =
                u64::try_from(result_drop_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let manager_bytes_after_result_drop = provider.memory().allocated_bytes();
            let result_manager_bytes_released =
                manager_bytes_before_result_drop.saturating_sub(manager_bytes_after_result_drop);
            eprintln!(
                "resident latency result teardown: sample={run} evaluate_return_ns={evaluate_return_ns} result_drop_ns={result_drop_ns} query_buffers={query_buffers} manager_bytes_before={manager_bytes_before_result_drop} manager_bytes_after={manager_bytes_after_result_drop} manager_bytes_released={result_manager_bytes_released} deallocation_calls=unavailable"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a serialized release-mode CUDA acceptance run"]
    fn resident_disconnected_four_thousand_rule_scaling_acceptance() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let corpus = std::path::PathBuf::from(
            std::env::var("XLOG_PINNED_CORPUS_ROOT")
                .expect("XLOG_PINNED_CORPUS_ROOT must name the pinned corpus checkout"),
        );
        assert_exact_clean_corpus(&corpus);
        let base_program = corpus_program(&corpus)?;
        let entry = corpus.join("scenarios/acceptance/issue1/q01_blind.xlog");
        let mut augmented_source = std::fs::read_to_string(&entry).map_err(|error| {
            XlogError::Execution(format!("failed to read {}: {error}", entry.display()))
        })?;
        augmented_source.push_str("\npred disconnected_seed(u32).\n");
        for family in 0..4_000 {
            augmented_source.push_str(&format!("pred disconnected_family_{family}(u32).\n"));
            augmented_source.push_str(&format!(
                "disconnected_family_{family}(X) :- disconnected_seed(X).\n"
            ));
        }
        let resolver = xlog_logic::compile::load_modules(&entry, vec![corpus.join("programs")])
            .map_err(|error| XlogError::Compilation(error.to_string()))?;
        let augmented_program = LogicProgram::compile_with_resolver(&augmented_source, &resolver)?;
        let Some(provider) = pinned_corpus_test_provider() else {
            return Ok(());
        };

        const WARMUP_PAIRS: usize = 2;
        const MEASURED_PAIRS: usize = 5;

        let mut warmup_base_seconds = Vec::with_capacity(WARMUP_PAIRS);
        let mut warmup_augmented_seconds = Vec::with_capacity(WARMUP_PAIRS);
        let mut base_seconds = Vec::with_capacity(MEASURED_PAIRS);
        let mut augmented_seconds = Vec::with_capacity(MEASURED_PAIRS);
        let mut expected_snapshot = None;
        let mut expected_profile = None;
        for pair in 0..(WARMUP_PAIRS + MEASURED_PAIRS) {
            let augmented_order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for augmented in augmented_order {
                let program = if augmented {
                    &augmented_program
                } else {
                    &base_program
                };
                let started = std::time::Instant::now();
                let result = {
                    let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
                    program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
                };
                let elapsed = started.elapsed().as_secs_f64();
                match (pair < WARMUP_PAIRS, augmented) {
                    (true, false) => warmup_base_seconds.push(elapsed),
                    (true, true) => warmup_augmented_seconds.push(elapsed),
                    (false, false) => base_seconds.push(elapsed),
                    (false, true) => augmented_seconds.push(elapsed),
                }

                let sample_kind = if augmented { "augmented" } else { "base" };
                let snapshot = snapshot_query_results(provider.as_ref(), &result)?;
                if let Some(expected) = &expected_snapshot {
                    assert_eq!(
                        &snapshot, expected,
                        "{sample_kind} resident pair {pair} changed query output"
                    );
                } else {
                    assert!(
                        !augmented,
                        "first resident scaling sample must be the base plan"
                    );
                    expected_snapshot = Some(snapshot);
                }

                let stats = result.stats.as_ref().expect("resident scaling profile");
                let graph = stats
                    .resident_graph
                    .as_ref()
                    .expect("resident scaling graph telemetry");
                assert_eq!(
                    graph.selection,
                    ResidentGraphSelectionKind::ResidentConditionalGraph
                );
                assert_eq!(graph.conditional_graph_launches, 1);
                assert_eq!(
                    op_count(stats, "scan") as u64,
                    graph.device_scan_invocations
                );
                assert_eq!(
                    op_count(stats, "filter") as u64,
                    graph.device_filter_invocations
                );
                let profile = (
                    strata_op_profile(stats),
                    graph.device_scan_invocations,
                    graph.device_filter_invocations,
                    graph.semantic_scan_invocations,
                    graph.semantic_filter_invocations,
                    graph.deferred_profile.timed_scan_filter_invocations,
                );
                if let Some(expected) = &expected_profile {
                    assert_eq!(
                        &profile, expected,
                        "{sample_kind} resident pair {pair} changed semantic or device op counts"
                    );
                } else {
                    assert!(
                        !augmented,
                        "first resident scaling profile must be the base plan"
                    );
                    expected_profile = Some(profile);
                }
                drop(result);
            }
        }

        let mut base_median_samples = base_seconds.clone();
        let mut augmented_median_samples = augmented_seconds.clone();
        let base_median = median_seconds(&mut base_median_samples);
        let augmented_median = median_seconds(&mut augmented_median_samples);
        let paired_deltas = augmented_seconds
            .iter()
            .zip(&base_seconds)
            .map(|(augmented, base)| augmented - base)
            .collect::<Vec<_>>();
        let mut paired_delta_samples = paired_deltas.clone();
        let paired_delta_median = median_seconds(&mut paired_delta_samples);
        let allowed_delta = (base_median * 0.10).max(0.100);
        eprintln!(
            "disconnected 4,000-rule resident timings: warmup_base={warmup_base_seconds:?} warmup_augmented={warmup_augmented_seconds:?} measured_base={base_seconds:?} measured_augmented={augmented_seconds:?} paired_deltas={paired_deltas:?} base_median={base_median:.6}s augmented_median={augmented_median:.6}s paired_delta_median={paired_delta_median:.6}s allowed_delta={allowed_delta:.6}s"
        );
        assert!(
            augmented_median - base_median <= allowed_delta,
            "disconnected 4,000-rule resident median delta {:.6}s exceeds {:.6}s: base={base_seconds:?} augmented={augmented_seconds:?}",
            augmented_median - base_median,
            allowed_delta,
        );
        assert!(
            paired_delta_median <= allowed_delta,
            "disconnected 4,000-rule resident paired median delta {paired_delta_median:.6}s exceeds {allowed_delta:.6}s: base={base_seconds:?} augmented={augmented_seconds:?} paired={paired_deltas:?}"
        );
        Ok(())
    }

    fn assert_required_resident_semantics(
        source: &str,
        case: &str,
        expected_query_schema: Option<&Schema>,
        expected_query_types: Option<&[ScalarType]>,
        expected_query_rows: Option<usize>,
    ) -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let program = LogicProgram::compile(source)?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        let expected = snapshot_query_results(provider.as_ref(), &baseline)?;
        if let Some(expected_query_schema) = expected_query_schema {
            assert_eq!(expected.len(), 1, "schema witness must have one query");
            assert_eq!(
                &expected[0].schema, expected_query_schema,
                "pre-change legacy query schema witness changed for {case}"
            );
        }
        if let Some(expected_query_types) = expected_query_types {
            assert_eq!(expected.len(), 1, "type witness must have one query");
            assert_eq!(
                expected[0]
                    .schema
                    .columns
                    .iter()
                    .map(|(_, scalar)| *scalar)
                    .collect::<Vec<_>>(),
                expected_query_types,
                "pre-change legacy query types changed for {case}"
            );
        }
        if let Some(expected_query_rows) = expected_query_rows {
            assert_eq!(expected.len(), 1, "row witness must have one query");
            assert_eq!(
                expected[0].rows.len(),
                expected_query_rows,
                "pre-change legacy query row count changed for {case}"
            );
        }
        drop(baseline);
        let resident = {
            let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        assert_eq!(
            snapshot_query_results(provider.as_ref(), &resident)?,
            expected,
            "resident semantic case {case} diverged"
        );
        let graph = resident
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident semantic telemetry");
        assert_eq!(
            graph.selection,
            ResidentGraphSelectionKind::ResidentConditionalGraph,
            "resident semantic case {case} did not use the production graph"
        );
        assert_eq!(graph.conditional_graph_launches, 1);
        assert_eq!(graph.core_transfers.tracked_htod_calls, 0);
        assert_eq!(graph.core_transfers.tracked_htod_bytes, 0);
        assert_eq!(graph.core_transfers.tracked_dtoh_calls, 0);
        assert_eq!(graph.core_transfers.tracked_dtoh_bytes, 0);
        assert_eq!(graph.core_transfers.provider_dtoh_calls, 0);
        assert_eq!(graph.core_transfers.untracked_metadata_dtoh_calls, 0);
        assert_eq!(graph.deterministic_d2h_violations, 0);
        assert_eq!(graph.final_observation.dtoh_calls, 1);
        assert_eq!(graph.final_observation.pinned_receipts, 1);
        Ok(())
    }

    fn program_with_authored_query_prefix(
        source: &str,
        original_head: &str,
        authored_head: &str,
    ) -> Result<Program> {
        let mut program = xlog_logic::parse_program(source)?;
        let mut renamed_declarations = 0usize;
        for declaration in &mut program.predicates {
            if declaration.name == original_head {
                declaration.name = authored_head.to_string();
                renamed_declarations += 1;
            }
        }
        let mut renamed_rules = 0usize;
        for rule in &mut program.rules {
            if rule.head.predicate == original_head {
                rule.head.predicate = authored_head.to_string();
                renamed_rules += 1;
            }
        }
        let mut renamed_queries = 0usize;
        for query in &mut program.queries {
            if query.atom.predicate == original_head {
                query.atom.predicate = authored_head.to_string();
                renamed_queries += 1;
            }
        }
        assert_eq!(renamed_declarations, 1);
        assert!(renamed_rules >= 1);
        assert_eq!(renamed_queries, 1);
        Ok(program)
    }

    fn compile_program_with_authored_query_prefix(
        source: &str,
        original_head: &str,
        authored_head: &str,
    ) -> Result<LogicProgram> {
        LogicProgram::compile_program(program_with_authored_query_prefix(
            source,
            original_head,
            authored_head,
        )?)
    }

    fn assert_required_resident_authored_prefix_semantics(source: &str, case: &str) -> Result<()> {
        const AUTHORED_HEAD: &str = "__xlog_query_authored";
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let program = compile_program_with_authored_query_prefix(source, "answer", AUTHORED_HEAD)?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let explicit_schema = Schema::new(vec![("external_value".to_string(), ScalarType::Symbol)]);
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(
                provider.clone(),
                HashMap::from([(
                    AUTHORED_HEAD.to_string(),
                    provider.create_empty_buffer(explicit_schema.clone())?,
                )]),
                true,
            )?
        };
        let expected = snapshot_query_results(provider.as_ref(), &baseline)?;
        assert_eq!(expected.len(), 1);
        drop(baseline);
        let resident = {
            let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(
                provider.clone(),
                HashMap::from([(
                    AUTHORED_HEAD.to_string(),
                    provider.create_empty_buffer(explicit_schema)?,
                )]),
                true,
            )?
        };
        assert_eq!(
            snapshot_query_results(provider.as_ref(), &resident)?,
            expected,
            "resident authored-prefix case {case} diverged"
        );
        let graph = resident
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident authored-prefix telemetry");
        assert_eq!(
            graph.selection,
            ResidentGraphSelectionKind::ResidentConditionalGraph,
            "resident authored-prefix case {case} declined"
        );
        assert_eq!(graph.conditional_graph_launches, 1);
        Ok(())
    }

    #[test]
    fn required_resident_preserves_programmatic_authored_query_prefix_empty_input() -> Result<()> {
        assert_required_resident_authored_prefix_semantics(
            r#"
                pred source(symbol).
                pred answer(symbol).
                answer(X) :- source(X).
                ?- answer(X).
            "#,
            "one authored rule",
        )
    }

    #[test]
    fn required_resident_does_not_treat_programmatic_authored_prefix_rules_as_generated(
    ) -> Result<()> {
        assert_required_resident_authored_prefix_semantics(
            r#"
                pred left_source(symbol).
                pred right_source(symbol).
                pred answer(symbol).
                answer(X) :- left_source(X).
                answer(X) :- right_source(X).
                ?- answer(X).
            "#,
            "two authored rules",
        )
    }

    #[test]
    fn compile_program_rejects_exact_generated_query_head_collision() -> Result<()> {
        let program = program_with_authored_query_prefix(
            r#"
                pred source(symbol).
                pred answer(symbol).
                answer(X) :- source(X).
                ?- answer(X).
            "#,
            "answer",
            "__xlog_query_0",
        )?;
        let error = match LogicProgram::compile_program(program) {
            Ok(_) => panic!("exact compiler-generated query head collision must be rejected"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("authored relation __xlog_query_0 collides with generated query head"),
            "unexpected collision error: {error}"
        );
        Ok(())
    }

    #[test]
    fn compiler_generated_query_relation_validation_is_exact_and_host_only() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                ?- source(X).
            "#,
        )?;

        let error = program
            .reject_compiler_generated_query_relation_names(["__xlog_query_0"], "persistent caller")
            .expect_err("an exact compiler-generated query head must be rejected");
        match error {
            XlogError::Execution(message) => assert_eq!(
                message,
                "persistent caller relation __xlog_query_0 collides with generated query head"
            ),
            other => panic!("expected typed execution rejection, got {other:?}"),
        }

        program.reject_compiler_generated_query_relation_names(
            ["__xlog_query_authored"],
            "persistent caller",
        )?;
        Ok(())
    }

    #[test]
    fn compiler_generated_query_relation_validation_rejects_provenance_mutation() -> Result<()> {
        let source = r#"
            pred source(symbol).
            ?- source(X).
        "#;

        let mut omitted = LogicProgram::compile(source)?;
        let LogicExecutionPlan::Ordinary(plan) = &mut omitted.plan else {
            panic!("ordinary program must compile to an ordinary plan");
        };
        plan.generated_query_rules.clear();
        let error = omitted
            .reject_compiler_generated_query_relation_names(std::iter::empty(), "caller input")
            .expect_err("omitted compiler provenance must be rejected");
        assert!(error.to_string().contains(
            "compiler-generated query provenance count 0 does not match authored query count 1"
        ));

        let mut repositioned = LogicProgram::compile(source)?;
        let LogicExecutionPlan::Ordinary(plan) = &mut repositioned.plan else {
            panic!("ordinary program must compile to an ordinary plan");
        };
        plan.generated_query_rules[0].query_index = 1;
        let error = repositioned
            .reject_compiler_generated_query_relation_names(std::iter::empty(), "caller input")
            .expect_err("repositioned compiler provenance must be rejected");
        assert!(error
            .to_string()
            .contains("compiler-generated query provenance position 0 carries query index 1"));

        let mut renamed = LogicProgram::compile(source)?;
        let LogicExecutionPlan::Ordinary(plan) = &mut renamed.plan else {
            panic!("ordinary program must compile to an ordinary plan");
        };
        let provenance = &plan.generated_query_rules[0];
        plan.rules_by_scc[provenance.scc_index][provenance.rule_index].head =
            "__xlog_query_spoof".to_string();
        let error = renamed
            .reject_compiler_generated_query_relation_names(std::iter::empty(), "caller input")
            .expect_err("renamed compiler-generated query head must be rejected");
        assert!(error.to_string().contains(
            "compiler-generated query provenance 0 expects head __xlog_query_0 but references authored head __xlog_query_spoof"
        ));
        Ok(())
    }

    #[test]
    fn cloned_program_shares_reusable_state_identity_but_recompile_does_not() -> Result<()> {
        let source = r#"
            pred source(symbol).
            ?- source(X).
        "#;
        let original = LogicProgram::compile(source)?;
        let cloned = original.clone();
        let recompiled = LogicProgram::compile(source)?;

        cloned.validate_reusable_state_identity(
            &original.reusable_state_identity,
            "materialized cache",
        )?;
        let error = recompiled
            .validate_reusable_state_identity(
                &original.reusable_state_identity,
                "materialized cache",
            )
            .expect_err("independent compilation must have a distinct reusable-state identity");
        assert!(matches!(error, XlogError::Execution(_)));
        assert_eq!(
            error.to_string(),
            "Execution error: materialized cache belongs to a different compiled logic program"
        );
        Ok(())
    }

    #[test]
    fn foreign_cache_and_runtime_are_rejected_before_evaluation_work() -> Result<()> {
        let source = r#"
            pred source(u32).
            pred out(u32).
            out(X) :- source(X).
            ?- out(X).
        "#;
        let program = LogicProgram::compile(source)?;
        let foreign_program = LogicProgram::compile(source)?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let base_store = program.create_relation_store(provider.clone())?;
        let (_, cache) =
            program.evaluate_with_relation_store_and_cache(provider.clone(), &base_store, false)?;
        let allocations_before_cache = provider.memory().alloc_count();

        let error = match foreign_program.evaluate_cached_relation_store(provider.clone(), &cache) {
            Ok(_) => panic!("independently compiled program must reject a foreign cache"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "Execution error: materialized cache belongs to a different compiled logic program"
        );
        assert_eq!(provider.memory().alloc_count(), allocations_before_cache);

        let mut runtime = program.create_session_runtime(provider.clone(), &base_store, false)?;
        let runtime_store_before = std::ptr::from_ref(runtime.executor.store());
        let source_version_before = runtime.executor.store().version("source");
        let allocations_before_runtime = provider.memory().alloc_count();
        let error =
            match foreign_program.evaluate_with_session_runtime(provider.clone(), &mut runtime) {
                Ok(_) => panic!("independently compiled program must reject a foreign runtime"),
                Err(error) => error,
            };
        assert_eq!(
            error.to_string(),
            "Execution error: session runtime belongs to a different compiled logic program"
        );
        assert_eq!(provider.memory().alloc_count(), allocations_before_runtime);
        assert_eq!(
            std::ptr::from_ref(runtime.executor.store()),
            runtime_store_before
        );
        assert_eq!(
            runtime.executor.store().version("source"),
            source_version_before
        );
        Ok(())
    }

    #[test]
    fn foreign_reusable_state_is_rejected_before_delta_take_or_device_work() -> Result<()> {
        let source = r#"
            pred source(u32).
            pred out(u32).
            source(1).
            out(X) :- source(X).
            ?- out(X).
        "#;
        let program = LogicProgram::compile(source)?;
        let foreign_program = LogicProgram::compile(source)?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let mut base_store = program.create_relation_store(provider.clone())?;
        let (_, initial_cache) =
            program.evaluate_with_relation_store_and_cache(provider.clone(), &base_store, false)?;
        let mut cache = Some(initial_cache);
        let mut runtime = None;
        let cache_before = cache.as_ref().map(std::ptr::from_ref);
        let source_version_before = base_store.version("source");
        let mut raw_delta_store = program.create_relation_store(provider.clone())?;
        let raw_insert = raw_delta_store
            .remove("source")
            .expect("inline source fact must materialize a nonempty raw delta");
        let raw_deltas = HashMap::from([(
            "source".to_string(),
            RelationDelta::new(Some(raw_insert), None),
        )]);
        let allocations_before_raw = provider.memory().alloc_count();

        let error = match foreign_program.prepare_relation_deltas_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            raw_deltas,
        ) {
            Ok(_) => panic!("raw delta preparation must reject a foreign cache"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "Execution error: materialized cache belongs to a different compiled logic program"
        );
        assert_eq!(provider.memory().alloc_count(), allocations_before_raw);
        assert_eq!(cache.as_ref().map(std::ptr::from_ref), cache_before);
        assert!(runtime.is_none());
        assert_eq!(base_store.version("source"), source_version_before);

        let mut no_cache = None;
        let mut foreign_runtime =
            Some(program.create_session_runtime(provider.clone(), &base_store, false)?);
        let runtime_before = foreign_runtime.as_ref().map(std::ptr::from_ref);
        let mut prepared_delta_store = program.create_relation_store(provider.clone())?;
        let prepared_insert = prepared_delta_store
            .remove("source")
            .expect("inline source fact must materialize a nonempty prepared delta");
        let prepared_batch = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![(
                "source".to_string(),
                RelationDelta::new(Some(prepared_insert), None),
            )],
            &BTreeSet::new(),
        )?;
        let allocations_before_prepared = provider.memory().alloc_count();
        let error = match foreign_program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut no_cache,
            &mut foreign_runtime,
            prepared_batch,
        ) {
            Ok(_) => panic!("prepared delta commit must reject a foreign runtime"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "Execution error: session runtime belongs to a different compiled logic program"
        );
        assert_eq!(provider.memory().alloc_count(), allocations_before_prepared);
        assert!(no_cache.is_none());
        assert_eq!(
            foreign_runtime.as_ref().map(std::ptr::from_ref),
            runtime_before
        );
        assert_eq!(base_store.version("source"), source_version_before);

        let mut ordered_delta_store = program.create_relation_store(provider.clone())?;
        let ordered_insert = ordered_delta_store
            .remove("source")
            .expect("inline source fact must materialize a nonempty ordered delta");
        let allocations_before_ordered = provider.memory().alloc_count();
        let error = match foreign_program.apply_relation_delta_batch(
            provider.clone(),
            &mut base_store,
            &mut cache,
            vec![(
                "source".to_string(),
                RelationDelta::new(Some(ordered_insert), None),
            )],
        ) {
            Ok(_) => panic!("ordered delta application must reject a foreign cache"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "Execution error: materialized cache belongs to a different compiled logic program"
        );
        assert_eq!(
            provider.memory().alloc_count(),
            allocations_before_ordered,
            "identity validation must run before ordered device coalescing"
        );
        assert_eq!(cache.as_ref().map(std::ptr::from_ref), cache_before);
        assert_eq!(base_store.version("source"), source_version_before);
        Ok(())
    }

    #[test]
    fn persistent_relation_store_rejects_generated_query_head_before_setup() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                ?- source(X).
            "#,
        )?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let mut store = program.create_relation_store(provider.clone())?;
        let query_schema = program
            .schemas
            .get("__xlog_query_0")
            .expect("compiler-generated query schema")
            .clone();
        store.put(
            "__xlog_query_0",
            provider.create_empty_buffer(query_schema)?,
        );
        let mut store_before = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_before.sort_unstable();
        let allocations_before = provider.memory().alloc_count();

        let error = match program.evaluate_with_relation_store(provider.clone(), &store, false) {
            Ok(_) => panic!("persistent caller must not seed a generated query head"),
            Err(error) => error,
        };
        assert!(matches!(error, XlogError::Execution(_)));
        assert!(error.to_string().contains(
            "persistent caller relation __xlog_query_0 collides with generated query head"
        ));
        assert_eq!(provider.memory().alloc_count(), allocations_before);
        let mut store_after = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_after.sort_unstable();
        assert_eq!(store_after, store_before);
        Ok(())
    }

    #[test]
    fn persistent_session_rejects_generated_query_head_before_setup() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                ?- source(X).
            "#,
        )?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let mut store = program.create_relation_store(provider.clone())?;
        let query_schema = program
            .schemas
            .get("__xlog_query_0")
            .expect("compiler-generated query schema")
            .clone();
        store.put(
            "__xlog_query_0",
            provider.create_empty_buffer(query_schema)?,
        );
        let mut store_before = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_before.sort_unstable();
        let allocations_before = provider.memory().alloc_count();

        let error = match program.create_session_runtime(provider.clone(), &store, false) {
            Ok(_) => panic!("persistent session must not seed a generated query head"),
            Err(error) => error,
        };
        assert!(matches!(error, XlogError::Execution(_)));
        assert!(error.to_string().contains(
            "persistent caller relation __xlog_query_0 collides with generated query head"
        ));
        assert_eq!(provider.memory().alloc_count(), allocations_before);
        let mut store_after = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_after.sort_unstable();
        assert_eq!(store_after, store_before);
        Ok(())
    }

    #[test]
    fn raw_delta_preparation_rejects_generated_query_head_without_consuming_state() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                ?- source(X).
            "#,
        )?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let mut store = program.create_relation_store(provider.clone())?;
        let (_, cached_store) =
            program.evaluate_with_relation_store_and_cache(provider.clone(), &store, false)?;
        let mut cached_store = Some(cached_store);
        let cached_store_before = cached_store.as_ref().map(std::ptr::from_ref);
        let mut session_runtime = None;
        let mut store_before = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_before.sort_unstable();
        let allocations_before = provider.memory().alloc_count();
        let deltas =
            HashMap::from([("__xlog_query_0".to_string(), RelationDelta::new(None, None))]);

        let error = match program.prepare_relation_deltas_commit_with_session_runtime(
            provider.clone(),
            &mut store,
            &mut cached_store,
            &mut session_runtime,
            deltas,
        ) {
            Ok(_) => panic!("delta preparation must reject a generated query head"),
            Err(error) => error,
        };
        assert!(matches!(error, XlogError::Execution(_)));
        assert!(error
            .to_string()
            .contains("caller delta relation __xlog_query_0 collides with generated query head"));
        assert_eq!(provider.memory().alloc_count(), allocations_before);
        assert_eq!(
            cached_store.as_ref().map(std::ptr::from_ref),
            cached_store_before
        );
        assert!(session_runtime.is_none());
        let mut store_after = store
            .names()
            .map(|name| {
                (
                    name.to_string(),
                    store.get(name).expect("named relation").num_rows(),
                )
            })
            .collect::<Vec<_>>();
        store_after.sort_unstable();
        assert_eq!(store_after, store_before);
        Ok(())
    }

    #[test]
    fn ordered_delta_preparation_rejects_generated_query_head_before_device_work() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                source("payload").
                ?- source(X).
            "#,
        )?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let mut insert_store = program.create_relation_store(provider.clone())?;
        let insert = insert_store
            .remove("source")
            .expect("inline source fact must materialize a nonempty insert");
        let mut delete_store = program.create_relation_store(provider.clone())?;
        let delete = delete_store
            .remove("source")
            .expect("inline source fact must materialize a nonempty delete");
        let allocations_before = provider.memory().alloc_count();

        let error = match program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "__xlog_query_0".to_string(),
                    RelationDelta::new(Some(insert), None),
                ),
                (
                    "__xlog_query_0".to_string(),
                    RelationDelta::new(None, Some(delete)),
                ),
            ],
            &BTreeSet::from(["__xlog_query_0".to_string()]),
        ) {
            Ok(_) => panic!("ordered delta preparation must reject a generated query head"),
            Err(error) => error,
        };
        assert!(matches!(error, XlogError::Execution(_)));
        assert!(error
            .to_string()
            .contains("caller delta relation __xlog_query_0 collides with generated query head"));
        assert_eq!(provider.memory().alloc_count(), allocations_before);
        Ok(())
    }

    #[test]
    fn resident_rejects_caller_input_for_exact_generated_query_head_before_setup() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                ?- source(X).
            "#,
        )?;
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let generated_schema = program
            .schemas
            .get("__xlog_query_0")
            .expect("generated query schema")
            .clone();
        for policy in [
            &[][..],
            &[("XLOG_DISABLE_RESIDENT_RECURSION", "1")][..],
            &[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")][..],
        ] {
            let input = provider.create_empty_buffer(generated_schema.clone())?;
            let allocations_before = provider.memory().alloc_count();
            let _env = ResidentEnvGuard::set(policy);
            let error = match program.evaluate_with_options(
                provider.clone(),
                HashMap::from([("__xlog_query_0".to_string(), input)]),
                true,
            ) {
                Ok(_) => panic!("caller input must not occupy a generated query head"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains(
                    "caller input relation __xlog_query_0 collides with generated query head"
                ),
                "unexpected caller-input collision error: {error}"
            );
            assert_eq!(
                provider.memory().alloc_count(),
                allocations_before,
                "generated-head collision must fail before resident setup"
            );
        }
        Ok(())
    }

    #[test]
    fn resident_executor_distinguishes_derived_placeholders_from_explicit_empty_inputs(
    ) -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(symbol).
                pred answer(symbol, symbol).
                pred seeded_derived(symbol).
                seeded_derived("seed").
                seeded_derived(X) :- source(X).
                answer("yes", X) :- source(X).
                ?- answer(Outcome, Claim).
            "#,
        )?;
        let LogicExecutionPlan::Ordinary(plan) = &program.plan else {
            panic!("ordinary test program must compile to an ordinary plan");
        };
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };

        let compiler_seeded =
            program.prepare_resident_executor(&provider, HashMap::new(), false, plan)?;
        assert!(compiler_seeded.store().get("source").is_some());
        assert!(compiler_seeded.store().get("answer").is_none());
        assert_eq!(
            compiler_seeded
                .store()
                .get("seeded_derived")
                .expect("derived relation with inline facts must be seeded")
                .cached_row_count(),
            Some(1),
        );

        let explicit_schema = Schema::new(vec![
            ("external_outcome".into(), ScalarType::Symbol),
            ("external_claim".into(), ScalarType::Symbol),
        ]);
        let explicit_empty = provider.create_empty_buffer(explicit_schema.clone())?;
        let explicitly_seeded = program.prepare_resident_executor(
            &provider,
            HashMap::from([("answer".to_string(), explicit_empty)]),
            false,
            plan,
        )?;
        assert_eq!(
            explicitly_seeded
                .store()
                .get("answer")
                .expect("explicit empty input must be retained")
                .schema(),
            &explicit_schema,
        );
        Ok(())
    }

    #[test]
    fn required_resident_query_schema_matches_legacy_for_nonempty_result() -> Result<()> {
        let expected_schema = Schema::new(vec![
            ("computed_0".into(), ScalarType::Symbol),
            ("c0".into(), ScalarType::Symbol),
        ]);
        assert_required_resident_semantics(
            r#"
                pred source(symbol).
                pred answer(symbol, symbol).
                source("claim").
                answer("yes", X) :- source(X).
                ?- answer(Outcome, Claim).
            "#,
            "nonempty synthetic query schema",
            Some(&expected_schema),
            None,
            None,
        )
    }

    #[test]
    fn required_resident_query_schema_matches_legacy_for_empty_result() -> Result<()> {
        let expected_schema = Schema::new(vec![
            ("computed_0".into(), ScalarType::Symbol),
            ("c0".into(), ScalarType::Symbol),
        ]);
        assert_required_resident_semantics(
            r#"
                pred source(symbol).
                pred answer(symbol, symbol).
                answer("yes", X) :- source(X).
                ?- answer(Outcome, Claim).
            "#,
            "empty synthetic query schema",
            Some(&expected_schema),
            None,
            None,
        )
    }

    #[test]
    fn required_resident_recursive_query_schema_matches_legacy() -> Result<()> {
        let expected_schema = Schema::new(vec![("c0".into(), ScalarType::U32)]);
        assert_required_resident_semantics(
            r#"
                pred seed(u32).
                pred edge(u32, u32).
                pred reach(u32).
                seed(1).
                edge(1, 2).
                reach(X) :- seed(X).
                reach(X) :- reach(X), edge(X, Y).
                ?- reach(X).
            "#,
            "recursive synthetic query schema",
            Some(&expected_schema),
            None,
            None,
        )
    }

    #[test]
    fn required_resident_recursive_constant_projection_schema_matches_legacy() -> Result<()> {
        let expected_schema = Schema::new(vec![
            ("item".into(), ScalarType::U32),
            ("computed_1".into(), ScalarType::U32),
        ]);
        assert_required_resident_semantics(
            r#"
                pred seed(item: u32).
                pred path(item: u32, category: u32).
                seed(7).
                path(X, 1) :- seed(X).
                path(X, 2) :- path(X, 1).
                ?- path(Item, Category).
            "#,
            "recursive constant-projection schema",
            Some(&expected_schema),
            Some(&[ScalarType::U32, ScalarType::U32]),
            Some(2),
        )
    }

    #[test]
    #[ignore = "requires a serialized release-mode CUDA acceptance run"]
    fn resident_semantic_acceptance_matrix() -> Result<()> {
        let ordinary_cases = [
            (
                "recursion",
                r#"
                    pred edge(u32, u32).
                    pred reach(u32, u32).
                    edge(1, 2). edge(2, 3).
                    reach(X, Y) :- edge(X, Y).
                    reach(X, Z) :- reach(X, Y), edge(Y, Z).
                    ?- reach(X, Y).
                "#,
            ),
            (
                "negation",
                r#"
                    pred item(u32). pred blocked(u32). pred visible(u32).
                    item(1). item(2). blocked(2).
                    visible(X) :- item(X), not blocked(X).
                    ?- visible(X).
                "#,
            ),
            (
                "constraint",
                r#"
                    pred safe(u32).
                    safe(1).
                    :- safe(2).
                    ?- safe(X).
                "#,
            ),
            (
                "multiple queries",
                r#"
                    pred seed(u32). pred left(u32). pred right(u32).
                    seed(7).
                    left(X) :- seed(X).
                    right(X) :- seed(X).
                    ?- left(X).
                    ?- right(X).
                "#,
            ),
            (
                "nullary set",
                r#"
                    pred ready(). pred answer().
                    ready().
                    answer() :- ready().
                    ?- answer().
                "#,
            ),
            (
                "same name with multiple arities",
                r#"
                    pred item(u32). pred item(u32, u32).
                    pred unary(u32). pred binary(u32, u32).
                    item(1). item(1, 2).
                    unary(X) :- item(X).
                    binary(X, Y) :- item(X, Y).
                    ?- unary(X).
                    ?- binary(X, Y).
                "#,
            ),
        ];
        for (case, source) in ordinary_cases {
            assert_required_resident_semantics(source, case, None, None, None)?;
        }

        let output_cases: [(&str, &str, &[ScalarType], usize); 7] = [
            (
                "zero-row output",
                r#"
                    pred empty(u32).
                    ?- empty(X).
                "#,
                &[ScalarType::U32],
                0,
            ),
            (
                "identity output",
                r#"
                    pred source(u32). pred copied(u32).
                    source(2). source(1).
                    copied(X) :- source(X).
                    ?- copied(X).
                "#,
                &[ScalarType::U32],
                2,
            ),
            (
                "u64 output",
                r#"
                    pred wide(u64).
                    wide(5000000000).
                    ?- wide(X).
                "#,
                &[ScalarType::U64],
                1,
            ),
            (
                "symbol output",
                r#"
                    pred labeled(symbol).
                    labeled("claim").
                    ?- labeled(X).
                "#,
                &[ScalarType::Symbol],
                1,
            ),
            (
                "mixed output",
                r#"
                    pred mixed(u32, u64, symbol).
                    mixed(7, 5000000000, "claim").
                    ?- mixed(Small, Wide, Label).
                "#,
                &[ScalarType::U32, ScalarType::U64, ScalarType::Symbol],
                1,
            ),
            (
                "nullary output",
                r#"
                    pred ready(). pred answer().
                    ready().
                    answer() :- ready().
                    ?- answer().
                "#,
                &[],
                1,
            ),
            (
                "arity-seventeen output",
                r#"
                    pred wide(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32).
                    wide(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17).
                    ?- wide(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q).
                "#,
                &[ScalarType::U32; 17],
                1,
            ),
        ];
        for (case, source, expected_types, expected_rows) in output_cases {
            assert_required_resident_semantics(
                source,
                case,
                None,
                Some(expected_types),
                Some(expected_rows),
            )?;
        }

        assert!(
            LogicProgram::compile(
                r#"
                    pred seed(u32). pred invalid(u32).
                    seed(1).
                    invalid(X) :- seed(X), X = "wrong type".
                    ?- seed(X).
                "#,
            )
            .is_err(),
            "invalid unreachable rules must still be validated"
        );

        required_resident_evaluation_canonicalizes_a_caller_input_only_relation()?;
        complete_store_evaluation_declines_resident_partial_execution_and_materializes_all_heads()?;

        {
            let _env_lock = resident_env_lock().lock().expect("resident env lock");
            let Some(provider) = ground_term_encoding_test_provider() else {
                return Ok(());
            };
            let program = LogicProgram::compile(
                r#"
                    pred seed(u32). pred out(u32).
                    seed(1). out(X) :- seed(X). ?- out(X).
                "#,
            )?;
            let store = program.create_relation_store(provider.clone())?;
            let mut session = program.create_session_runtime(provider.clone(), &store, true)?;
            let (result, _) = program.evaluate_with_session_runtime(provider, &mut session)?;
            let graph = result
                .stats
                .as_ref()
                .and_then(|stats| stats.resident_graph.as_ref())
                .expect("session resident decline telemetry");
            assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
            assert_eq!(
                graph.decline,
                Some(ResidentGraphDeclineReason::FullStoreRequested)
            );
            assert_eq!(graph.conditional_graph_launches, 0);
        }

        {
            let _env_lock = resident_env_lock().lock().expect("resident env lock");
            let Some(provider) = ground_term_encoding_test_provider() else {
                return Ok(());
            };
            let program = LogicProgram::compile(
                r#"
                    pred p(u32). pred q(u32).
                    p(1). q(X) :- p(X), know p(X). ?- q(X).
                "#,
            )?;
            let baseline = {
                let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
                program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
            };
            let expected = snapshot_query_results(provider.as_ref(), &baseline)?;
            drop(baseline);
            let automatic =
                program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
            assert_eq!(
                snapshot_query_results(provider.as_ref(), &automatic)?,
                expected
            );
            let graph = automatic
                .stats
                .as_ref()
                .and_then(|stats| stats.resident_graph.as_ref())
                .expect("nonordinary resident decline telemetry");
            assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
            assert_eq!(
                graph.decline,
                Some(ResidentGraphDeclineReason::NonOrdinaryPlan)
            );
            assert_eq!(graph.conditional_graph_launches, 0);
        }
        Ok(())
    }

    #[test]
    fn complete_store_evaluation_declines_resident_partial_execution_and_materializes_all_heads(
    ) -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred base(u32).
                pred edge(u32, u32).
                pred queried(u32).
                pred disconnected(u32).
                base(7).
                edge(7, 8).
                queried(X) :- base(X).
                queried(Y) :- queried(X), edge(X, Y).
                disconnected(X) :- base(X).
                ?- queried(X).
            "#,
        )?;
        let seed = program.create_relation_store(provider.clone())?;
        let (result, store) =
            program.evaluate_with_relation_store_and_cache(provider.clone(), &seed, true)?;

        assert_eq!(
            provider.download_column::<u32>(&result.queries[0].buffer, 0)?,
            vec![7, 8]
        );
        assert_eq!(
            provider.download_column::<u32>(
                store
                    .as_relation_store()
                    .get("disconnected")
                    .expect("complete disconnected head"),
                0,
            )?,
            vec![7]
        );
        let graph = result
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident decline telemetry");
        assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
        assert_eq!(
            graph.decline,
            Some(ResidentGraphDeclineReason::FullStoreRequested)
        );
        assert_eq!(graph.conditional_graph_launches, 0);
        assert_eq!(graph.staged_store_mutations, 0);
        assert!(result
            .stats
            .as_ref()
            .expect("profile")
            .format_json()
            .contains("resident_graph_declined"));
        Ok(())
    }

    #[test]
    fn no_query_program_bypasses_resident_certification_and_executes_the_full_plan() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let program = LogicProgram::compile(
            r#"
                pred seed(u32).
                pred disconnected(u32).
                seed(1).
                disconnected(X) :- seed(X).
            "#,
        )?;
        assert_eq!(program.resident_certification_initializations(), 0);
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };

        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        assert!(baseline.queries.is_empty());
        let baseline_profile =
            strata_op_profile(baseline.stats.as_ref().expect("baseline profile"));
        assert_eq!(op_count(baseline.stats.as_ref().unwrap(), "scan"), 1);
        assert_eq!(program.resident_certification_initializations(), 0);
        drop(baseline);

        let automatic = program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
        assert!(automatic.queries.is_empty());
        assert_eq!(
            strata_op_profile(automatic.stats.as_ref().expect("automatic profile")),
            baseline_profile
        );
        let graph = automatic
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident decline telemetry");
        assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
        assert_eq!(
            graph.decline,
            Some(ResidentGraphDeclineReason::FullStoreRequested)
        );
        assert_eq!(graph.conditional_graph_launches, 0);
        assert_eq!(program.resident_certification_initializations(), 0);
        drop(automatic);

        let required_result = {
            let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider, HashMap::new(), true)
        };
        let required = match required_result {
            Ok(_) => panic!("no-query evaluation cannot use the resident partial-result route"),
            Err(error) => error,
        };
        assert!(required
            .to_string()
            .contains("resident conditional-graph execution was required but declined"));
        assert!(required.to_string().contains("FullStoreRequested"));
        assert_eq!(program.resident_certification_initializations(), 0);
        Ok(())
    }

    #[test]
    fn automatic_resident_decline_executes_the_untouched_full_plan() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred unsupported(f64).
                pred seed(u32).
                pred disconnected(u32).
                unsupported(7.5).
                seed(1).
                disconnected(X) :- seed(X).
                ?- unsupported(X).
            "#,
        )?;
        let baseline = {
            let _env = ResidentEnvGuard::set(&[("XLOG_DISABLE_RESIDENT_RECURSION", "1")]);
            program.evaluate_with_options(provider.clone(), HashMap::new(), true)?
        };
        let baseline_snapshot = snapshot_query_results(provider.as_ref(), &baseline)?;
        let baseline_profile =
            strata_op_profile(baseline.stats.as_ref().expect("baseline profile"));
        assert_eq!(op_count(baseline.stats.as_ref().unwrap(), "scan"), 2);
        drop(baseline);

        let automatic = program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
        assert_eq!(
            snapshot_query_results(provider.as_ref(), &automatic)?,
            baseline_snapshot
        );
        assert_eq!(
            strata_op_profile(automatic.stats.as_ref().expect("automatic profile")),
            baseline_profile
        );
        let graph = automatic
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident decline telemetry");
        assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
        assert!(graph.decline.is_some());
        assert_eq!(graph.conditional_graph_launches, 0);
        Ok(())
    }

    #[test]
    fn ordinary_unsupported_scalar_types_decline_before_launch() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let programs = [
            r#"
                pred unsupported(i64).
                unsupported(7).
                ?- unsupported(X).
            "#,
            r#"
                pred unsupported(f64).
                unsupported(7.5).
                ?- unsupported(X).
            "#,
            r#"
                pred unsupported(u32, f64).
                unsupported(7, 8.5).
                ?- unsupported(X, Y).
            "#,
        ];
        for source in programs {
            let program = LogicProgram::compile(source)?;
            let automatic =
                program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
            assert_eq!(automatic.queries[0].buffer.cached_row_count(), Some(1));
            let graph = automatic
                .stats
                .as_ref()
                .and_then(|stats| stats.resident_graph.as_ref())
                .expect("resident decline telemetry");
            assert_eq!(graph.selection, ResidentGraphSelectionKind::ExistingGpu);
            assert!(graph.decline.is_some());
            assert_eq!(graph.conditional_graph_launches, 0);
            drop(automatic);

            let error = {
                let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
                match program.evaluate_with_options(provider.clone(), HashMap::new(), true) {
                    Ok(_) => {
                        panic!("required resident execution must reject a prelaunch decline")
                    }
                    Err(error) => error,
                }
            };
            assert!(error
                .to_string()
                .contains("resident conditional-graph execution was required but declined"));
            assert_eq!(program.resident_certification_initializations(), 1);
        }

        let mut cached_failure = LogicProgram::compile(
            r#"
                pred output(u32).
                output(7).
                ?- output(X).
            "#,
        )?;
        assert_eq!(cached_failure.resident_certification_initializations(), 1);
        cached_failure.reusable_state_identity = Arc::new(LogicProgramIdentity::new());
        cached_failure
            .reusable_state_identity
            .get_or_init_resident_certification(|| -> Result<ResidentGraphCertifiedPlan> {
                Err(XlogError::Execution(
                    "deterministic certification failure".into(),
                ))
            })
            .expect_err("injected certification must fail");
        let automatic =
            cached_failure.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
        assert_eq!(automatic.queries[0].buffer.cached_row_count(), Some(1));
        let decline = automatic
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .and_then(|stats| stats.decline.as_ref())
            .expect("automatic certification failure must report its decline");
        assert!(format!("{decline:?}").contains("deterministic certification failure"));
        drop(automatic);

        let required = {
            let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
            match cached_failure.evaluate_with_options(provider, HashMap::new(), true) {
                Ok(_) => panic!("required resident execution must reject cached certification"),
                Err(error) => error,
            }
        };
        assert!(required
            .to_string()
            .contains("resident conditional-graph execution was required but declined"));
        assert!(required
            .to_string()
            .contains("deterministic certification failure"));
        assert_eq!(cached_failure.resident_certification_initializations(), 1);
        Ok(())
    }

    #[test]
    fn ordinary_evaluation_uses_the_required_resident_graph_on_the_callers_cuda_context(
    ) -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        assert!(provider.memory().runtime().is_some());
        let program = LogicProgram::compile(
            r#"
                pred seed(u32).
                pred edge(u32, u32).
                pred reach(u32).
                seed(1).
                edge(1, 2).
                edge(2, 3).
                reach(X) :- seed(X).
                reach(Y) :- reach(X), edge(X, Y), Y >= 2.
                ?- reach(X).
            "#,
        )?;
        let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
        let result = program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
        assert_eq!(
            provider.download_column::<u32>(&result.queries[0].buffer, 0)?,
            vec![1, 2, 3]
        );
        let graph = result
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident selection telemetry");
        assert_eq!(
            graph.selection,
            ResidentGraphSelectionKind::ResidentConditionalGraph
        );
        assert_eq!(graph.conditional_graph_launches, 1);
        assert!(graph.device_scan_invocations > 0);
        assert!(graph.device_filter_invocations > 0);
        assert_eq!(graph.core_transfers.tracked_htod_calls, 0);
        assert_eq!(graph.core_transfers.tracked_dtoh_calls, 0);
        assert_eq!(graph.core_transfers.provider_dtoh_calls, 0);
        assert_eq!(graph.core_transfers.untracked_metadata_dtoh_calls, 0);
        assert_eq!(graph.final_observation.dtoh_calls, 1);
        assert_eq!(graph.final_observation.pinned_receipts, 1);
        assert!(graph.deferred_profile.device_elapsed_ns > 0);
        Ok(())
    }

    #[test]
    fn required_resident_evaluation_canonicalizes_a_caller_input_only_relation() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred input(u32).
                pred output(u32).
                output(X) :- input(X).
                ?- output(X).
            "#,
        )?;
        let input = provider.create_buffer_from_slice::<u32>(
            &[2, 1, 2],
            Schema::new(vec![("x".into(), ScalarType::U32)]),
        )?;
        assert!(!input.canonical_full_row_set_certified());
        let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
        let result = program.evaluate_with_options(
            provider.clone(),
            HashMap::from([("input".to_string(), input)]),
            true,
        )?;
        assert_eq!(
            provider.download_column::<u32>(&result.queries[0].buffer, 0)?,
            vec![1, 2]
        );
        let graph = result
            .stats
            .as_ref()
            .and_then(|stats| stats.resident_graph.as_ref())
            .expect("resident selection telemetry");
        assert_eq!(
            graph.selection,
            ResidentGraphSelectionKind::ResidentConditionalGraph
        );
        assert_eq!(graph.conditional_graph_launches, 1);
        Ok(())
    }

    #[test]
    fn incompatible_caller_input_type_fails_before_resident_allocation_or_launch() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred input(u32).
                pred output(u32).
                output(X) :- input(X).
                ?- output(X).
            "#,
        )?;
        let input = provider.create_buffer_from_slice::<u64>(
            &[1],
            Schema::new(vec![("x".into(), ScalarType::U64)]),
        )?;
        provider.memory().reset_alloc_count();
        let allocated_before = provider.memory().allocated_bytes();
        let _env = ResidentEnvGuard::set(&[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")]);
        let error = match program.evaluate_with_options(
            provider.clone(),
            HashMap::from([("input".to_string(), input)]),
            false,
        ) {
            Err(error) => error,
            Ok(_) => panic!("incompatible input type unexpectedly reached resident setup"),
        };
        assert!(error.to_string().contains("schema mismatch"));
        assert!(provider.memory().allocated_bytes() <= allocated_before);
        assert_eq!(provider.memory().alloc_count(), 0);
        assert!(provider.memory().runtime().is_some());
        Ok(())
    }

    #[test]
    fn malformed_caller_device_count_fails_before_resident_allocation_or_launch() -> Result<()> {
        let _env_lock = resident_env_lock().lock().expect("resident env lock");
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred input(u32).
                pred output(u32).
                output(X) :- input(X).
                ?- output(X).
            "#,
        )?;
        for (selection, policy) in [
            ("automatic", &[][..]),
            ("required", &[("XLOG_REQUIRE_RESIDENT_RECURSION", "1")][..]),
        ] {
            let mut column = provider.memory().alloc::<u8>(4)?;
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&7u32.to_ne_bytes(), &mut column)
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            let mut device_count = provider.memory().alloc::<u32>(1)?;
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&[2], &mut device_count)
                .map_err(|error| XlogError::Kernel(error.to_string()))?;
            let input = CudaBuffer::from_columns(
                vec![column.into()],
                1,
                device_count,
                Schema::new(vec![("x".into(), ScalarType::U32)]),
            );
            provider.memory().reset_alloc_count();
            let allocated_before = provider.memory().allocated_bytes();
            let _env = ResidentEnvGuard::set(policy);
            let error = match program.evaluate_with_options(
                provider.clone(),
                HashMap::from([("input".to_string(), input)]),
                false,
            ) {
                Err(error) => error,
                Ok(_) => panic!("malformed input count unexpectedly reached resident setup"),
            };
            assert!(
                error
                    .to_string()
                    .contains("Logical row count 2 exceeds row capacity 1"),
                "unexpected error for {selection}: {error}"
            );
            assert!(provider.memory().allocated_bytes() <= allocated_before);
            assert_eq!(provider.memory().alloc_count(), 0);
            assert!(provider.memory().runtime().is_some());
        }
        Ok(())
    }

    #[test]
    fn program_fact_loader_uses_shared_ground_term_encoding() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred encoded(u32, u64, i32, i64, f32, f64, bool, bool, symbol, symbol).
                encoded(42, 43, -44, -45, 1.5, 2.25, true, 0, "hello", world).
            "#,
        )?;
        let store = program.create_relation_store(provider.clone())?;
        let encoded = store
            .get("encoded")
            .ok_or_else(|| XlogError::Execution("missing encoded fact buffer".to_string()))?;

        assert_eq!(provider.download_column::<u32>(encoded, 0)?, vec![42]);
        assert_eq!(provider.download_column::<u64>(encoded, 1)?, vec![43]);
        assert_eq!(provider.download_column::<i32>(encoded, 2)?, vec![-44]);
        assert_eq!(provider.download_column::<i64>(encoded, 3)?, vec![-45]);
        assert_eq!(provider.download_column::<f32>(encoded, 4)?, vec![1.5]);
        assert_eq!(provider.download_column::<f64>(encoded, 5)?, vec![2.25]);
        assert_eq!(provider.download_column::<bool>(encoded, 6)?, vec![true]);
        assert_eq!(provider.download_column::<bool>(encoded, 7)?, vec![false]);
        assert_eq!(
            provider.download_column::<u32>(encoded, 8)?,
            vec![symbol::intern("hello")]
        );
        assert_eq!(
            provider.download_column::<u32>(encoded, 9)?,
            vec![symbol::intern("world")]
        );

        let invalid = LogicProgram::compile(
            r#"
                pred invalid(u32).
                invalid(X).
            "#,
        )?;
        let error = match invalid.create_relation_store(provider) {
            Ok(_) => panic!("a variable in a fact must be rejected"),
            Err(error) => error,
        };
        let XlogError::Execution(message) = error else {
            panic!("fact encoding must remain an Execution error, got {error:?}");
        };
        assert_eq!(
            message,
            "Failed to encode fact for predicate invalid at column 0: Fact cannot contain variable X"
        );
        Ok(())
    }

    #[test]
    fn grouped_facts_preserve_fact_and_rule_results_without_seed_operations() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred base(u32).
                pred derived(u32).
                base(1).
                base(1).
                base(2).
                derived(X) :- base(X).
                ?- base(X).
                ?- derived(X).
            "#,
        )?;
        let LogicExecutionPlan::Ordinary(plan) = &program.plan else {
            panic!("ordinary source must compile to an ordinary plan");
        };
        let executable_rule_count = plan.rules_by_scc.iter().map(Vec::len).sum::<usize>();
        assert_eq!(program.program.facts().count(), 3);
        let expected_executable_rules =
            program.program.proper_rules().count() + program.program.queries.len();
        assert_eq!(
            executable_rule_count, expected_executable_rules,
            "compiled rules must correspond only to executable source and query rules"
        );

        provider.reset_host_transfer_stats();
        let mut executor = program.prepare_executor(&provider, HashMap::new(), true)?;
        let fact_load_transfers = provider.host_transfer_stats();
        assert_eq!(fact_load_transfers.htod_calls, 1);
        assert_eq!(
            fact_load_transfers.htod_bytes,
            3 * std::mem::size_of::<u32>() as u64
        );
        assert_eq!(fact_load_transfers.dtoh_calls, 0);
        assert_eq!(fact_load_transfers.dtoh_bytes, 0);
        let base = executor
            .store()
            .get("base")
            .ok_or_else(|| XlogError::Execution("missing grouped base facts".to_string()))?;
        assert_eq!(base.cached_row_count(), Some(2));
        let mut materialized_base = provider.download_column::<u32>(base, 0)?;
        materialized_base.sort_unstable();
        assert_eq!(materialized_base, vec![1, 2]);

        executor.execute_plan(plan)?;
        for query_index in 0..2 {
            let relation_name = format!("__xlog_query_{query_index}");
            let query = executor.store().get(&relation_name).ok_or_else(|| {
                XlogError::Execution(format!("missing query relation {relation_name}"))
            })?;
            let mut rows = provider.download_column::<u32>(query, 0)?;
            rows.sort_unstable();
            assert_eq!(rows, vec![1, 2]);
        }

        let stats = executor.execution_stats(4);
        let scan_count = stats
            .strata
            .iter()
            .flat_map(|stratum| &stratum.ops)
            .filter(|op| op.op_name == "scan")
            .count();
        let union_count = stats
            .strata
            .iter()
            .flat_map(|stratum| &stratum.ops)
            .filter(|op| op.op_name == "union")
            .count();
        assert_eq!(scan_count, executable_rule_count);
        assert_eq!(union_count, executable_rule_count);
        Ok(())
    }

    #[test]
    fn ordinary_execution_after_fact_setup_has_no_host_transfers() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred base(u32).
                pred derived(u32).
                base(7).
                derived(X) :- base(X).
                ?- derived(X).
            "#,
        )?;
        let LogicExecutionPlan::Ordinary(plan) = &program.plan else {
            panic!("ordinary source must compile to an ordinary plan");
        };

        let mut executor = program.prepare_executor(&provider, HashMap::new(), false)?;
        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        provider.reset_deterministic_d2h_violations();
        executor.execute_plan(plan)?;

        let transfers = provider.host_transfer_stats();
        assert_eq!(
            transfers.htod_calls, 0,
            "execution must not upload host data"
        );
        assert_eq!(
            transfers.htod_bytes, 0,
            "execution must not upload host bytes"
        );
        assert_eq!(
            transfers.dtoh_calls, 0,
            "execution must not download device data"
        );
        assert_eq!(
            transfers.dtoh_bytes, 0,
            "execution must not download device bytes"
        );
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        assert_eq!(provider.deterministic_d2h_violation_count(), 0);

        let query = executor
            .store()
            .get("__xlog_query_0")
            .ok_or_else(|| XlogError::Execution("missing query result".to_string()))?;
        assert_eq!(provider.download_column::<u32>(query, 0)?, vec![7]);
        Ok(())
    }

    #[test]
    fn grouped_fact_loading_preserves_arity_qualified_relations() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred source(symbol, i64).
                pred source(u32).
                pred result(symbol).
                node(key).
                source(key, 5000000000).
                source(key, 5000000000).
                source(1).
                source(1).
                result(X) :- node(X), know source(X, Y).
                ?- result(X).
            "#,
        )?;
        let store = program.create_relation_store(provider.clone())?;
        let unary = store
            .get("source/1")
            .ok_or_else(|| XlogError::Execution("missing unary source facts".to_string()))?;
        let binary = store
            .get("source/2")
            .ok_or_else(|| XlogError::Execution("missing binary source facts".to_string()))?;

        assert_eq!(unary.cached_row_count(), Some(1));
        assert_eq!(binary.cached_row_count(), Some(1));
        assert_eq!(provider.download_column::<u32>(unary, 0)?, vec![1]);
        assert_eq!(
            provider.download_column::<u32>(binary, 0)?,
            vec![symbol::intern("key")]
        );
        assert_eq!(
            provider.download_column::<i64>(binary, 1)?,
            vec![5_000_000_000]
        );

        let evidence = program.execute_epistemic_evidence(provider.clone(), HashMap::new())?;
        assert_eq!(
            evidence.final_output.schema().arity(),
            1,
            "the epistemic evidence path must project the public result arity"
        );
        assert_eq!(
            provider.download_column::<u32>(&evidence.final_output, 0)?,
            vec![symbol::intern("key")],
            "the epistemic evidence path must execute the binary modal source facts"
        );
        Ok(())
    }

    #[test]
    fn ordinary_compile_qualifies_same_name_multi_arity_relations() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred item(u32). pred item(u32, u32).
                pred unary(u32). pred binary(u32, u32).
                item(1). item(1, 2).
                unary(X) :- item(X).
                binary(X, Y) :- item(X, Y).
                ?- unary(X).
                ?- binary(X, Y).
            "#,
        )?;

        let unary_item = *program
            .rel_ids
            .get("item/1")
            .expect("ordinary compiler registers item/1");
        let binary_item = *program
            .rel_ids
            .get("item/2")
            .expect("ordinary compiler registers item/2");
        assert_ne!(unary_item, binary_item);

        let LogicExecutionPlan::Ordinary(plan) = &program.plan else {
            panic!("ordinary source must compile to an ordinary execution plan");
        };
        let unary_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == "unary")
            .expect("compiled unary rule");
        let binary_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == "binary")
            .expect("compiled binary rule");
        assert_eq!(unary_rule.body.referenced_relations(), vec![unary_item]);
        assert_eq!(binary_rule.body.referenced_relations(), vec![binary_item]);
        Ok(())
    }

    #[test]
    fn recursive_execution_preserves_inline_fact_semantics() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred edge(u32, u32).
                pred reach(u32, u32).
                edge(1, 2).
                edge(2, 3).
                reach(X, Y) :- edge(X, Y).
                reach(X, Z) :- reach(X, Y), edge(Y, Z).
                ?- reach(X, Z).
            "#,
        )?;

        let result = program.evaluate(provider.clone(), HashMap::new())?;
        assert_eq!(result.queries.len(), 1);
        let xs = provider.download_column::<u32>(&result.queries[0].buffer, 0)?;
        let zs = provider.download_column::<u32>(&result.queries[0].buffer, 1)?;
        let mut rows = xs.into_iter().zip(zs).collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(
            rows,
            vec![(1, 2), (1, 3), (2, 3)],
            "recursive production execution must derive the transitive inline-fact result"
        );
        Ok(())
    }

    #[test]
    fn nullary_execution_preserves_asserted_inline_fact_truth() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred enabled().
                pred result().
                enabled().
                result() :- enabled().
                ?- result().
            "#,
        )?;

        let result = program.evaluate(provider.clone(), HashMap::new())?;
        assert_eq!(result.queries.len(), 1);
        assert_eq!(
            provider.device_row_count(&result.queries[0].buffer)?,
            1,
            "an asserted nullary fact must make the derived nullary query true"
        );
        Ok(())
    }

    #[test]
    fn caller_input_is_unioned_with_inline_facts_before_execution() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred source(u32).
                pred result(u32).
                source(1).
                result(X) :- source(X).
                ?- result(X).
            "#,
        )?;
        let source_schema = program
            .schema("source")
            .ok_or_else(|| XlogError::Execution("missing source schema".to_string()))?
            .clone();
        let caller_value = 2u32.to_le_bytes();
        let caller_input =
            provider.create_buffer_from_slices(&[caller_value.as_slice()], source_schema)?;

        let result = program.evaluate(
            provider.clone(),
            HashMap::from([("source".to_string(), caller_input)]),
        )?;
        let mut rows = provider.download_column::<u32>(&result.queries[0].buffer, 0)?;
        rows.sort_unstable();
        assert_eq!(
            rows,
            vec![1, 2],
            "caller-provided rows and inline facts must both reach the executable plan"
        );
        Ok(())
    }

    struct RecursiveDuplicateFactProfile {
        executable_rule_count: usize,
        scan_count: usize,
        union_count: usize,
        rows: Vec<(u32, u32)>,
    }

    fn recursive_duplicate_fact_profile(
        provider: Arc<CudaKernelProvider>,
        fact_count: usize,
    ) -> Result<RecursiveDuplicateFactProfile> {
        let facts = "edge(1, 2).\n".repeat(fact_count);
        let program = LogicProgram::compile(&format!(
            r#"
                pred edge(u32, u32).
                pred reach(u32, u32).
                {facts}
                reach(X, Y) :- edge(X, Y).
                reach(X, Z) :- reach(X, Y), edge(Y, Z).
                ?- reach(X, Z).
            "#
        ))?;
        assert_eq!(program.program.facts().count(), fact_count);
        assert_eq!(
            program.program.proper_rules().count(),
            2,
            "the recursive rule shape must remain constant across fact counts"
        );
        let LogicExecutionPlan::Ordinary(plan) = &program.plan else {
            panic!("recursive source must compile to an ordinary plan");
        };
        let executable_rule_count = plan.rules_by_scc.iter().map(Vec::len).sum::<usize>();

        let result = program.evaluate_with_options(provider.clone(), HashMap::new(), true)?;
        let stats = result
            .stats
            .as_ref()
            .ok_or_else(|| XlogError::Execution("missing execution profile".to_string()))?;
        let scan_count = stats
            .strata
            .iter()
            .flat_map(|stratum| &stratum.ops)
            .filter(|op| op.op_name == "scan")
            .count();
        let union_count = stats
            .strata
            .iter()
            .flat_map(|stratum| &stratum.ops)
            .filter(|op| op.op_name == "union")
            .count();
        let xs = provider.download_column::<u32>(&result.queries[0].buffer, 0)?;
        let ys = provider.download_column::<u32>(&result.queries[0].buffer, 1)?;
        let mut rows = xs.into_iter().zip(ys).collect::<Vec<_>>();
        rows.sort_unstable();
        Ok(RecursiveDuplicateFactProfile {
            executable_rule_count,
            scan_count,
            union_count,
            rows,
        })
    }

    #[test]
    fn recursive_plan_operations_are_invariant_to_duplicate_fact_count() -> Result<()> {
        let Some(provider) = ground_term_encoding_test_provider() else {
            return Ok(());
        };

        let one_fact = recursive_duplicate_fact_profile(provider.clone(), 1)?;
        let many_facts = recursive_duplicate_fact_profile(provider, 64)?;
        assert_eq!(one_fact.rows, vec![(1, 2)]);
        assert_eq!(many_facts.rows, one_fact.rows);
        assert_eq!(
            many_facts.executable_rule_count, one_fact.executable_rule_count,
            "executable rule count must not scale with source fact count"
        );
        assert_eq!(
            many_facts.scan_count, one_fact.scan_count,
            "executable scan count must not scale with source fact count"
        );
        assert_eq!(
            many_facts.union_count, one_fact.union_count,
            "executable union count must not scale with source fact count"
        );
        Ok(())
    }

    #[test]
    fn g91_compatibility_plan_records_gpu_upper_and_refinement_passes() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = g91
                pred domain(u32).
                pred p(u32).
                pred q(u32).
                domain(7).
                p(X) :- domain(X), possible q(X).
                q(X) :- domain(X), possible p(X).
                ?- p(X).
                ?- q(X).
            "#,
        )?;

        let LogicExecutionPlan::EpistemicG91Compatibility(plan) = &program.plan else {
            panic!("mutual G91 possibility cycle must select compatibility iteration");
        };
        assert_eq!(plan.snapshot_relations.len(), 2);
        assert_eq!(plan.convergence_predicates, vec!["p", "q"]);
        let summary = program
            .epistemic_plan_json()
            .expect("epistemic compatibility summary");
        assert!(summary.contains("epistemic_g91_compatibility_gpu"));
        assert!(summary.contains("\"gpu_passes\":[\"upper_bound\",\"refinement\"]"));
        assert!(summary.contains("\"execution_backend\":\"gpu\""));
        assert!(summary.contains("\"fallback_policy\":\"reject_unsupported\""));
        assert!(!summary.contains("\"cpu_fallback_total_zero\""));
        assert!(!summary.contains("\"cpu_fallback_is_zero\""));
        Ok(())
    }

    #[test]
    fn g91_compatibility_infers_undeclared_snapshot_schema() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = g91
                domain(1).
                p(X) :- domain(X), possible p(X).
                ?- p(X).
            "#,
        )?;

        let LogicExecutionPlan::EpistemicG91Compatibility(plan) = &program.plan else {
            panic!("undeclared G91 relation must select compatibility iteration");
        };
        let snapshot = plan
            .snapshot_relations
            .get("p")
            .expect("snapshot name for p");
        let refinement_schemas = gpu_evaluation_pass_schemas(&plan.refinement);
        let schema = refinement_schemas
            .get(snapshot)
            .expect("inferred snapshot schema");
        assert_eq!(schema.arity(), 1);
        assert_eq!(schema.column_type(0), Some(ScalarType::U32));
        Ok(())
    }

    #[test]
    fn g91_compatibility_compile_rejects_a_recursive_aggregate_component() {
        let result = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = g91
                pred seed(u32).
                pred p(u32).
                pred totals(u64).
                seed(1).
                p(X) :- seed(X), possible p(X).
                p(X) :- seed(X), totals(_).
                totals(count(X)) :- p(X).
                ?- p(X).
            "#,
        );
        let error = match result {
            Ok(_) => panic!("recursive aggregation must not enter compatibility refinement"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains("Gelfond-1991 compatibility"), "{message}");
        assert!(message.contains("aggregate"), "{message}");
        assert!(message.contains("totals"), "{message}");
    }

    #[test]
    fn g91_introspection_preserves_authored_rules_and_queries() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = g91
                pred base(u32).
                pred p(u32).
                base(1).
                p(X) :- base(X), possible p(X).
                ?- p(X).
            "#,
        )?;

        let provenance = program.rule_provenance();
        let p_rule = provenance
            .iter()
            .find(|rule| rule.head == "p(X)")
            .expect("authored p rule provenance");
        assert_eq!(p_rule.support_relation_ids, vec!["base", "p"]);
        assert!(provenance.iter().all(|rule| {
            rule.support_relation_ids
                .iter()
                .all(|relation| !relation.starts_with("__xlog_"))
        }));

        let traces = program.proof_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].query, "p(X)");
        assert!(traces[0].source_facts.iter().any(|fact| fact == "base(1)."));
        Ok(())
    }

    #[test]
    fn predicate_function_proof_traces_preserve_source_names_and_normalized_support() -> Result<()>
    {
        let program = LogicProgram::compile(
            r#"
                pred candidate(i32, i32).
                pred blocked(i32).
                pred forbidden(i32).
                pred answer(i32).
                candidate(1, 2).
                func visible(X) = Y :- candidate(X, Y), not blocked(Y).
                answer(Y) :- Y is visible(1), not forbidden(Y).
                ?- answer(Y).
            "#,
        )?;

        let provenance = program.rule_provenance();
        let answer_rule = provenance
            .iter()
            .find(|rule| rule.head == "answer(Y)")
            .expect("answer rule provenance");
        assert!(
            ["candidate", "blocked", "forbidden"]
                .iter()
                .all(|relation| answer_rule
                    .support_relation_ids
                    .iter()
                    .any(|id| id == relation)),
            "{:?}",
            answer_rule.support_relation_ids
        );

        let traces = program.proof_traces();
        assert_eq!(traces.len(), 1);
        assert!(traces[0]
            .rejected_alternatives
            .iter()
            .any(|alternative| alternative == "not blocked(Y)"));
        assert!(traces[0]
            .rejected_alternatives
            .iter()
            .any(|alternative| alternative == "not forbidden(Y)"));
        assert!(traces[0]
            .source_facts
            .iter()
            .any(|fact| fact == "candidate(1, 2)."));
        assert!(!format!("{provenance:?}{traces:?}").contains("__XLOG_FUNCTION"));
        Ok(())
    }

    #[test]
    fn relation_clone_context_preserves_cuda_error_variants() {
        let kernel = relation_clone_error(
            "cloning relation 'fact'".to_string(),
            XlogError::Kernel("launch failed".to_string()),
        );
        assert!(matches!(
            kernel,
            XlogError::Kernel(message)
                if message == "cloning relation 'fact': launch failed"
        ));

        let exhausted = relation_clone_error(
            "cloning relation 'fact'".to_string(),
            XlogError::ResourceExhausted {
                context: "GPU memory pressure: layer=manager_alloc current_bytes=60 requested_bytes=4 required_bytes=64 required_u64_overflow=false budget_bytes=63 prior_peak_bytes=60".to_string(),
                estimated_bytes: 64,
                budget_bytes: 63,
            },
        );
        match exhausted {
            XlogError::ResourceExhausted {
                context,
                estimated_bytes,
                budget_bytes,
            } => {
                assert_eq!(
                    context,
                    "cloning relation 'fact': GPU memory pressure: layer=manager_alloc current_bytes=60 requested_bytes=4 required_bytes=64 required_u64_overflow=false budget_bytes=63 prior_peak_bytes=60"
                );
                assert_eq!(estimated_bytes, 64);
                assert_eq!(budget_bytes, 63);
            }
            error => panic!("expected resource exhaustion, got {error}"),
        }
    }

    #[test]
    fn compiled_argument_preserves_declared_names_domains_and_order() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                domain party: u32.
                pred transfer(giver: party, receiver: party, asset: u32, time: i64).
                pred positional(u32, i64, symbol).
            "#,
        )?;

        let transfer = program
            .argument_schema("transfer")
            .expect("compiled transfer argument schema");
        assert_eq!(
            transfer
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![
                ("giver", true, Some("party"), ScalarType::U32),
                ("receiver", true, Some("party"), ScalarType::U32),
                ("asset", true, None, ScalarType::U32),
                ("time", true, None, ScalarType::I64),
            ]
        );

        let positional = program
            .argument_schema("positional")
            .expect("compiled positional argument schema");
        assert_eq!(
            positional
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![
                ("c0", false, None, ScalarType::U32),
                ("c1", false, None, ScalarType::I64),
                ("c2", false, None, ScalarType::Symbol),
            ]
        );

        Ok(())
    }

    #[test]
    fn compiled_argument_uses_schema_metadata_for_inferred_relations() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                pred source(value: i64).
                inferred(X) :- source(X).
            "#,
        )?;

        let inferred = program
            .argument_schema("inferred")
            .expect("compiled inferred argument schema");
        assert_eq!(
            inferred
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![("c0", false, None, ScalarType::I64)]
        );
        assert!(program.argument_schema("unknown").is_none());

        Ok(())
    }

    #[test]
    fn compiled_argument_recovers_source_metadata_for_arity_qualified_relations() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                domain identity: u32.
                pred joined(u32, u32).
                pred seed(u32, u32).
                pred polymorphic(value: identity).
                pred polymorphic(left: identity, right: identity).

                joined(X, Y) :- seed(X, Y), know polymorphic(X), possible polymorphic(X, Y).
            "#,
        )?;

        let unary = program
            .argument_schema("polymorphic/1")
            .expect("compiled arity-qualified argument schema");
        assert_eq!(
            unary
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![("value", true, Some("identity"), ScalarType::U32)]
        );

        Ok(())
    }

    #[test]
    fn epistemic_compile_qualifies_extensional_signatures_used_by_reduction() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred source(symbol, i64).
                pred source(u32).
                pred result(symbol).
                node(key).
                source(key, 5000000000).
                source(1).
                result(X) :- node(X), know source(X, Y).
                ?- result(X).
            "#,
        )?;

        let source = program
            .schema("source/2")
            .expect("missing arity-qualified binary source schema");
        assert_eq!(source.column_type(0), Some(ScalarType::Symbol));
        assert_eq!(source.column_type(1), Some(ScalarType::I64));
        assert!(program.schema("source/1").is_some());
        assert_eq!(
            program
                .schema("result")
                .expect("missing augmented result schema")
                .arity(),
            2
        );

        Ok(())
    }

    #[test]
    fn epistemic_compile_infers_hidden_column_types_from_runtime_binders() -> Result<()> {
        let inferred_source = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred result(symbol).
                node(key).
                raw(key, 5000000000).
                edge(X, Y) :- raw(X, Y).
                result(X) :- node(X), know edge(X, Y).
                ?- result(X).
            "#,
        )?;
        assert_eq!(
            inferred_source
                .schema("result")
                .expect("missing inferred-source result schema")
                .column_type(1),
            Some(ScalarType::I64)
        );

        let arithmetic_binding = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred allowed(u64).
                pred result(symbol).
                node(key).
                allowed(1).
                result(X) :- node(X), Y is cast(1, u64), not know allowed(Y).
                ?- result(X).
            "#,
        )?;
        assert_eq!(
            arithmetic_binding
                .schema("result")
                .expect("missing arithmetic-bound result schema")
                .column_type(1),
            Some(ScalarType::U64)
        );

        Ok(())
    }

    #[test]
    fn epistemic_compile_uses_one_extensional_arity_census() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                p(a).
                result(X) :- p(X), know p(X).
                :- p(X, Y).
                ?- result(X).
            "#,
        )?;

        assert!(program.schema("p/1").is_some());
        assert!(program.schema("p/2").is_some());
        Ok(())
    }

    #[test]
    fn epistemic_compile_preserves_recursive_stratum_schema() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(u32).
                pred edge(u32, u32).
                pred accepted_edge(u32, u32).
                pred reach(u32, u32).
                node(1).
                node(2).
                node(3).
                edge(1, 2).
                edge(2, 3).
                accepted_edge(X, Y) :- node(X), node(Y), know edge(X, Y).
                reach(X, Y) :- node(X), node(Y), know accepted_edge(X, Y).
                reach(X, Z) :- reach(X, Y), node(Z), know accepted_edge(Y, Z).
                ?- reach(X, Z).
            "#,
        )?;

        assert_eq!(
            program
                .schema("reach")
                .expect("missing recursive reach schema")
                .arity(),
            2
        );
        assert_eq!(program.plan_kind_label(), "epistemic_stratified");
        Ok(())
    }

    #[test]
    fn epistemic_compile_rejects_unscoped_same_head_rule_unions() {
        let error = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred p().
                pred q().
                pred result(symbol).
                q().
                result(a) :- know p().
                result(b) :- know q().
                ?- result(X).
            "#,
        )
        .err()
        .expect("same-head modal clauses require per-clause provenance");
        let message = error.to_string();
        assert!(message.contains("epistemic rule-union materialization"));
        assert!(message.contains("result/1"), "{message}");
    }

    #[test]
    fn epistemic_compile_rejects_derived_source_arity_collisions() {
        let sources = [
            r#"
                #pragma epistemic_mode = faeel
                unary(a).
                binary(a, b).
                result(X) :- unary(X), know unary(X).
                result(X, Y) :- binary(X, Y), know binary(X, Y).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                ?- result(A, B).
            "#,
        ];

        for source in sources {
            let error = match LogicProgram::compile(source) {
                Ok(_) => panic!("derived source-arity collisions must fail compilation"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    &error,
                    XlogError::UnsupportedEpistemicConstruct { construct, .. }
                        if construct == "epistemic derived predicate schema"
                ),
                "{error}"
            );
        }
    }

    #[test]
    fn epistemic_compile_rejects_constrained_augmented_head_query() {
        let error = match LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                ?- result(other).
            "#,
        ) {
            Ok(_) => panic!("a constrained augmented-head query must fail compilation"),
            Err(error) => error,
        };

        assert!(
            matches!(
                &error,
                XlogError::UnsupportedEpistemicConstruct { construct, .. }
                    if construct == "epistemic augmented head query"
            ),
            "{error}"
        );
    }

    #[test]
    fn epistemic_compile_rejects_divergent_ordinary_bound_internal_arities() {
        let error = match LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred edge(symbol, i64).
                pred allowed(i64).
                pred result(symbol).
                node(key).
                edge(key, 5000000000).
                allowed(5000000000).
                result(X) :- node(X).
                result(X) :- node(X), edge(X, Y), know allowed(Y).
                ?- result(X).
            "#,
        ) {
            Ok(_) => panic!("divergent internal arities must fail compilation"),
            Err(error) => error,
        };

        assert!(
            matches!(
                &error,
                XlogError::UnsupportedEpistemicConstruct { construct, .. }
                    if construct == "epistemic augmented predicate schema"
            ),
            "{error}"
        );
    }

    #[test]
    fn compiled_argument_preserves_declared_prefix_when_schema_is_wider() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                domain identity: u32.
                pred node(id: identity).
                pred edge(source: identity, target: identity).
                pred one_hop(node: identity).

                one_hop(X) :- node(X), know edge(X, Y).
            "#,
        )?;

        let one_hop = program
            .argument_schema("one_hop")
            .expect("compiled widened argument schema");
        assert_eq!(
            one_hop
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![
                ("node", true, Some("identity"), ScalarType::U32),
                ("c1", false, None, ScalarType::U32),
            ]
        );

        Ok(())
    }

    #[test]
    fn compiled_argument_uses_the_declaration_selected_by_compilation() -> Result<()> {
        let program = LogicProgram::compile(
            r#"
                domain first: u32.
                domain second: i64.
                pred duplicate(value: first).
                pred duplicate(value: second).
            "#,
        )?;

        let duplicate = program
            .argument_schema("duplicate")
            .expect("compiled duplicate argument schema");
        assert_eq!(
            duplicate
                .iter()
                .map(|argument| (
                    argument.name(),
                    argument.source_named(),
                    argument.sort(),
                    argument.scalar_type(),
                ))
                .collect::<Vec<_>>(),
            vec![("value", true, Some("second"), ScalarType::I64)]
        );

        Ok(())
    }
}

#[cfg(test)]
mod relation_delta_coalesce_tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType};

    fn test_provider() -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            Ok(Arc::new(
                xlog_cuda::CudaProviderBuilder::new(
                    0,
                    MemoryBudget::with_limit(1024 * 1024 * 1024),
                )
                .build()?,
            ))
        })();

        finish_test_provider_setup(
            provider,
            std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1"),
        )
    }

    #[test]
    #[should_panic(expected = "XLOG_REQUIRE_CUDA=1 but CUDA provider construction failed")]
    fn required_cuda_provider_failure_is_not_silently_skipped() {
        finish_test_provider_setup::<()>(
            Err(XlogError::Execution("forced provider failure".to_string())),
            true,
        );
    }

    fn test_buffer(provider: &CudaKernelProvider, rows: &[u32]) -> CudaBuffer {
        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("upload rows");
        let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc rows");
        let row_count = rows.len() as u32;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[row_count], &mut d_num_rows)
            .expect("upload row count");
        CudaBuffer::from_columns(vec![col.into()], rows.len() as u64, d_num_rows, schema)
    }

    fn read_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<u32> {
        provider
            .download_column::<u32>(buffer, 0)
            .expect("download")
    }

    fn sorted_query_rows(provider: &CudaKernelProvider, result: &LogicEvalResult) -> Vec<u32> {
        let mut rows = read_u32(provider, &result.queries[0].buffer);
        rows.sort_unstable();
        rows
    }

    fn assert_empty_modal_cycle_query_result(
        provider: &CudaKernelProvider,
        result: &LogicEvalResult,
    ) -> Result<()> {
        assert_eq!(result.queries.len(), 1);
        let query = &result.queries[0];
        assert_eq!(query.relation_name, "p");
        assert!(query.columns.is_empty());
        assert_eq!(query.buffer.schema().arity(), 0);
        assert_eq!(provider.device_row_count(&query.buffer)?, 0);
        Ok(())
    }

    #[test]
    fn modal_cycle_query_presentation_is_consistent_across_evaluation_apis() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                #pragma epistemic_mode = faeel
                pred p().
                p() :- possible p().
                ?- p().
            "#,
        )?;

        let direct = program.evaluate(provider.clone(), HashMap::new())?;
        assert_empty_modal_cycle_query_result(provider.as_ref(), &direct)?;

        let relation_store = program.create_relation_store(provider.clone())?;
        let (from_store, cached_store) = program.evaluate_with_relation_store_and_cache(
            provider.clone(),
            &relation_store,
            false,
        )?;
        assert_empty_modal_cycle_query_result(provider.as_ref(), &from_store)?;

        let cached = program.evaluate_cached_relation_store(provider.clone(), &cached_store)?;
        assert_empty_modal_cycle_query_result(provider.as_ref(), &cached)?;

        let mut runtime =
            program.create_session_runtime(provider.clone(), &relation_store, false)?;
        let (from_session, _) =
            program.evaluate_with_session_runtime(provider.clone(), &mut runtime)?;
        assert_empty_modal_cycle_query_result(provider.as_ref(), &from_session)?;
        Ok(())
    }

    #[test]
    fn coalesce_batch_cancels_insert_delete_pairs_on_device() {
        let provider = match test_provider() {
            Some(provider) => provider,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let batch = vec![
            (
                "streamed_fact".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[7, 8])), None),
            ),
            (
                "streamed_fact".to_string(),
                RelationDelta::new(None, Some(test_buffer(&provider, &[8]))),
            ),
            (
                "streamed_fact".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[9])), None),
            ),
        ];

        let report = coalesce_relation_delta_batch_with_cancellation_capture(
            provider.as_ref(),
            batch,
            &BTreeSet::new(),
        )
        .expect("coalesce relation delta batch");
        let delta = report
            .deltas
            .get("streamed_fact")
            .expect("coalesced relation");
        let insert = delta.insert.as_ref().expect("coalesced insert");
        assert_eq!(read_u32(&provider, insert), vec![7, 9]);
        assert!(delta.delete.as_ref().map(|b| b.is_empty()).unwrap_or(true));
        assert_eq!(report.report_seed.input_delta_count, 3);
        assert_eq!(report.report_seed.changed_relations, 1);
        assert_eq!(report.report_seed.coalesced_insert_rows, 2);
        assert_eq!(report.report_seed.coalesced_delete_rows, 0);
        assert_eq!(report.report_seed.canceled_rows, 1);
    }

    #[test]
    fn relation_delta_batch_updates_runtime_store_and_reports_coalesced_counts() -> Result<()> {
        let Some(provider) = test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return Ok(());
        };

        let source = r#"
            pred streamed_fact(u32).
            pred out(u32).

            out(X) :- streamed_fact(X).

            ?- out(X).
        "#;
        let program = LogicProgram::compile(source)?;
        let mut coalesced_store = program.create_relation_store(provider.clone())?;
        let mut coalesced_cache = None;

        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        let report = program.apply_relation_delta_batch(
            provider.clone(),
            &mut coalesced_store,
            &mut coalesced_cache,
            vec![
                (
                    "streamed_fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1, 2, 3])), None),
                ),
                (
                    "streamed_fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
                ),
                (
                    "streamed_fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[4])), None),
                ),
            ],
        )?;
        let transfer_stats = provider.host_transfer_stats();

        assert_eq!(report.input_delta_count, 3);
        assert_eq!(report.changed_relations, 1);
        assert_eq!(report.insert_rows, 3);
        assert_eq!(report.delete_rows, 0);
        assert_eq!(report.coalesced_insert_rows, 3);
        assert_eq!(report.coalesced_delete_rows, 0);
        assert_eq!(report.canceled_rows, 1);
        assert_eq!(transfer_stats.dtoh_bytes, 0);
        assert_eq!(transfer_stats.dtoh_calls, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);

        let coalesced = program.evaluate_cached_relation_store(
            provider.clone(),
            coalesced_cache
                .as_ref()
                .expect("cached store after delta batch"),
        )?;
        let coalesced_rows = sorted_query_rows(&provider, &coalesced);

        let mut sequential_store = program.create_relation_store(provider.clone())?;
        let mut sequential_cache = None;
        for delta in [
            RelationDelta::new(Some(test_buffer(&provider, &[1, 2, 3])), None),
            RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
            RelationDelta::new(Some(test_buffer(&provider, &[4])), None),
        ] {
            program.apply_relation_deltas(
                provider.clone(),
                &mut sequential_store,
                &mut sequential_cache,
                HashMap::from([("streamed_fact".to_string(), delta)]),
            )?;
        }
        let sequential = program.evaluate_cached_relation_store(
            provider.clone(),
            sequential_cache
                .as_ref()
                .expect("cached store after sequential deltas"),
        )?;
        let sequential_rows = sorted_query_rows(&provider, &sequential);

        let mut replacement_store = program.create_relation_store(provider.clone())?;
        replacement_store.put("streamed_fact", test_buffer(&provider, &[1, 3, 4]));
        let replacement =
            program.evaluate_with_relation_store(provider.clone(), &replacement_store, false)?;
        let replacement_rows = sorted_query_rows(&provider, &replacement);

        assert_eq!(coalesced_rows, vec![1, 3, 4]);
        assert_eq!(coalesced_rows, sequential_rows);
        assert_eq!(coalesced_rows, replacement_rows);
        Ok(())
    }
}

#[cfg(test)]
mod relation_delta_preparation_tests {
    use super::*;
    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType};

    fn test_provider_with_budget(limit: u64) -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            Ok(Arc::new(
                xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(limit)).build()?,
            ))
        })();

        match provider {
            Ok(provider) => Some(provider),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!(
                    "XLOG_REQUIRE_CUDA=1 but CUDA provider construction failed: {}",
                    error
                )
            }
            Err(error) => {
                eprintln!("Skipping test: no CUDA device available ({error})");
                None
            }
        }
    }

    fn test_provider() -> Option<Arc<CudaKernelProvider>> {
        test_provider_with_budget(1024 * 1024 * 1024)
    }

    fn test_buffer(provider: &CudaKernelProvider, rows: &[u32]) -> CudaBuffer {
        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
        let mut column = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut column)
            .expect("upload rows");
        let mut device_row_count = provider.memory().alloc::<u32>(1).expect("alloc rows");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[rows.len() as u32], &mut device_row_count)
            .expect("upload row count");
        CudaBuffer::from_columns(
            vec![column.into()],
            rows.len() as u64,
            device_row_count,
            schema,
        )
    }

    fn sorted_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<u32> {
        let mut rows = provider
            .download_column::<u32>(buffer, 0)
            .expect("download rows");
        rows.sort_unstable();
        rows
    }

    fn insert_rows(
        provider: &CudaKernelProvider,
        prepared: &PreparedRelationDeltaBatch,
        relation: &str,
    ) -> Vec<u32> {
        let delta = prepared
            .net_deltas()
            .get(relation)
            .expect("prepared relation delta");
        sorted_u32(
            provider,
            delta.insert.as_ref().expect("prepared net insert buffer"),
        )
    }

    fn cancellation_batch(provider: &CudaKernelProvider) -> Vec<(String, RelationDelta)> {
        vec![
            (
                "fact".to_string(),
                RelationDelta::new(Some(test_buffer(provider, &[5])), None),
            ),
            (
                "fact".to_string(),
                RelationDelta::new(None, Some(test_buffer(provider, &[5]))),
            ),
            (
                "fact".to_string(),
                RelationDelta::new(Some(test_buffer(provider, &[6])), None),
            ),
        ]
    }

    fn report_counts(report: &LogicDeltaReport) -> (usize, usize, u64, u64, u64) {
        (
            report.input_delta_count,
            report.changed_relations,
            report.coalesced_insert_rows,
            report.coalesced_delete_rows,
            report.canceled_rows,
        )
    }

    #[test]
    fn prepared_batch_exposes_net_deltas_and_ordered_cancellation_buffers() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile("pred fact(u32).")?;

        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1, 2])), None),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[3]))),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[3, 4])), None),
                ),
            ],
            &BTreeSet::from(["fact".to_string()]),
        )?;

        assert_eq!(insert_rows(&provider, &prepared, "fact"), vec![1, 4]);
        let cancellations = prepared
            .cancellations()
            .get("fact")
            .expect("fact cancellation trace");
        assert_eq!(cancellations.len(), 2);
        assert_eq!(cancellations[0].update_index(), 1);
        assert_eq!(
            cancellations[0].incoming_direction(),
            RelationDeltaDirection::Delete
        );
        assert_eq!(sorted_u32(&provider, cancellations[0].tuples()), vec![2]);
        assert_eq!(cancellations[1].update_index(), 3);
        assert_eq!(
            cancellations[1].incoming_direction(),
            RelationDeltaDirection::Insert
        );
        assert_eq!(sorted_u32(&provider, cancellations[1].tuples()), vec![3]);
        Ok(())
    }

    #[test]
    fn cancellation_trace_distinguishes_canceled_and_surviving_insert_occurrences() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile("pred fact(u32).")?;

        let insert_delete_insert = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[7])), None),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[7]))),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[7])), None),
                ),
            ],
            &BTreeSet::from(["fact".to_string()]),
        )?;
        assert_eq!(
            insert_rows(&provider, &insert_delete_insert, "fact"),
            vec![7]
        );
        let first_trace = insert_delete_insert
            .cancellations()
            .get("fact")
            .expect("first cancellation trace");
        assert_eq!(first_trace.len(), 1);
        assert_eq!(first_trace[0].update_index(), 1);
        assert_eq!(
            first_trace[0].incoming_direction(),
            RelationDeltaDirection::Delete
        );
        assert_eq!(sorted_u32(&provider, first_trace[0].tuples()), vec![7]);

        let delete_insert_insert = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[7]))),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[7])), None),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[7])), None),
                ),
            ],
            &BTreeSet::from(["fact".to_string()]),
        )?;
        assert_eq!(
            insert_rows(&provider, &delete_insert_insert, "fact"),
            vec![7]
        );
        let second_trace = delete_insert_insert
            .cancellations()
            .get("fact")
            .expect("second cancellation trace");
        assert_eq!(second_trace.len(), 1);
        assert_eq!(second_trace[0].update_index(), 1);
        assert_eq!(
            second_trace[0].incoming_direction(),
            RelationDeltaDirection::Insert
        );
        assert_eq!(sorted_u32(&provider, second_trace[0].tuples()), vec![7]);
        Ok(())
    }

    #[test]
    fn raw_combined_delta_preserves_delete_then_insert_while_batch_cancels() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile("pred fact(u32).")?;

        let mut raw_store = program.create_relation_store(provider.clone())?;
        raw_store.put("fact", test_buffer(&provider, &[7]));
        let mut raw_cache = None;
        let mut raw_runtime = None;
        let mut raw_delta = HashMap::new();
        raw_delta.insert(
            "fact".to_string(),
            RelationDelta::new(
                Some(test_buffer(&provider, &[7])),
                Some(test_buffer(&provider, &[7])),
            ),
        );

        let raw_commit = program.prepare_relation_deltas_commit_with_session_runtime(
            provider.clone(),
            &mut raw_store,
            &mut raw_cache,
            &mut raw_runtime,
            raw_delta,
        )?;
        let raw_report = raw_commit.commit();
        assert_eq!(
            sorted_u32(&provider, raw_store.get("fact").unwrap()),
            vec![7]
        );
        assert_eq!(raw_report.insert_rows, 1);
        assert_eq!(raw_report.delete_rows, 1);
        assert_eq!(raw_report.canceled_rows, 0);
        assert_eq!(raw_report.changed_relations, 1);

        let mut batch_store = program.create_relation_store(provider.clone())?;
        batch_store.put("fact", test_buffer(&provider, &[7]));
        let before_batch_version = batch_store.version("fact");
        let mut batch_cache = None;
        let mut batch_runtime = None;
        let batch = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "fact".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[7])), None),
                ),
                (
                    "fact".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[7]))),
                ),
            ],
            &BTreeSet::from(["fact".to_string()]),
        )?;
        let batch_commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut batch_store,
            &mut batch_cache,
            &mut batch_runtime,
            batch,
        )?;
        let batch_report = batch_commit.commit();
        assert_eq!(batch_store.version("fact"), before_batch_version);
        assert_eq!(
            sorted_u32(&provider, batch_store.get("fact").unwrap()),
            vec![7]
        );
        assert_eq!(batch_report.insert_rows, 0);
        assert_eq!(batch_report.delete_rows, 0);
        assert_eq!(batch_report.canceled_rows, 1);
        assert_eq!(batch_report.changed_relations, 0);
        Ok(())
    }

    #[test]
    fn cancellation_capture_is_scoped_to_selected_relations() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred evidenced(u32).
                pred positional(u32).
            "#,
        )?;
        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "evidenced".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1])), None),
                ),
                (
                    "positional".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[2])), None),
                ),
                (
                    "evidenced".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[1]))),
                ),
                (
                    "positional".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
                ),
                (
                    "evidenced".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[3]))),
                ),
                (
                    "evidenced".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[3])), None),
                ),
            ],
            &BTreeSet::from(["evidenced".to_string()]),
        )?;

        assert_eq!(
            prepared
                .cancellations()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["evidenced"])
        );
        let evidenced = prepared
            .cancellations()
            .get("evidenced")
            .expect("selected relation cancellation trace");
        assert_eq!(
            evidenced
                .iter()
                .map(RelationDeltaCancellation::update_index)
                .collect::<Vec<_>>(),
            vec![2, 5]
        );
        assert_eq!(prepared.report_seed.canceled_rows, 3);
        Ok(())
    }

    #[test]
    fn disabled_cancellation_capture_preserves_net_data_and_stats_without_trace_work() -> Result<()>
    {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred fact(u32).
                pred out(u32).
                out(X) :- fact(X).
                ?- out(X).
            "#,
        )?;
        let mut uncaptured_store = program.create_relation_store(provider.clone())?;
        let mut uncaptured_cache = None;
        let mut uncaptured_runtime = None;
        let mut captured_store = program.create_relation_store(provider.clone())?;
        let mut captured_cache = None;
        let mut captured_runtime = None;

        let uncaptured_batch = cancellation_batch(&provider);
        provider.memory().reset_alloc_count();
        let uncaptured = program.prepare_relation_delta_batch(
            provider.as_ref(),
            uncaptured_batch,
            &BTreeSet::new(),
        )?;
        let uncaptured_allocations = provider.memory().alloc_count();
        assert!(uncaptured.cancellations().is_empty());
        assert_eq!(insert_rows(&provider, &uncaptured, "fact"), vec![6]);
        let uncaptured_commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut uncaptured_store,
            &mut uncaptured_cache,
            &mut uncaptured_runtime,
            uncaptured,
        )?;
        let uncaptured_report = uncaptured_commit.commit();

        let captured_batch = cancellation_batch(&provider);
        provider.memory().reset_alloc_count();
        let captured = program.prepare_relation_delta_batch(
            provider.as_ref(),
            captured_batch,
            &BTreeSet::from(["fact".to_string()]),
        )?;
        let captured_allocations = provider.memory().alloc_count();
        assert_eq!(insert_rows(&provider, &captured, "fact"), vec![6]);
        assert_eq!(
            captured
                .cancellations()
                .get("fact")
                .expect("captured cancellation")
                .len(),
            1
        );
        assert!(
            captured_allocations > uncaptured_allocations,
            "capturing cancellation tuples should add device allocation requests in this fixture: captured={captured_allocations}, uncaptured={uncaptured_allocations}"
        );
        let captured_commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut captured_store,
            &mut captured_cache,
            &mut captured_runtime,
            captured,
        )?;
        let captured_report = captured_commit.commit();

        assert_eq!(
            report_counts(&uncaptured_report),
            report_counts(&captured_report)
        );
        assert_eq!(report_counts(&uncaptured_report), (3, 1, 1, 0, 1));
        assert_eq!(
            sorted_u32(
                &provider,
                uncaptured_store.get("fact").expect("uncaptured base fact")
            ),
            vec![6]
        );
        assert_eq!(
            sorted_u32(
                &provider,
                captured_store.get("fact").expect("captured base fact")
            ),
            vec![6]
        );
        Ok(())
    }

    #[test]
    fn retained_preparation_failure_discards_mutated_runtime_without_base_puts() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred safe(u32).
                pred forbidden(u32).
                pred out(u32).
                out(X) :- safe(X).
                :- forbidden(X).
                ?- out(X).
            "#,
        )?;
        let mut base_store = program.create_relation_store(provider.clone())?;
        let safe_version = base_store.version("safe").expect("safe version");
        let forbidden_version = base_store.version("forbidden").expect("forbidden version");
        let mut runtime =
            Some(program.create_session_runtime(provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "safe".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1])), None),
                ),
                (
                    "forbidden".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[9])), None),
                ),
            ],
            &BTreeSet::new(),
        )?;

        let error = match program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared,
        ) {
            Ok(_) => panic!("constraint-violating preparation must fail"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("Constraint 0 violated"),
            "unexpected preparation error: {error}"
        );
        assert!(cache.is_none(), "failed preparation must discard cache");
        assert!(runtime.is_none(), "failed preparation must discard runtime");
        assert_eq!(base_store.version("safe"), Some(safe_version));
        assert_eq!(base_store.version("forbidden"), Some(forbidden_version));
        assert_eq!(base_store.get("safe").expect("safe base").num_rows(), 0);
        assert_eq!(
            base_store
                .get("forbidden")
                .expect("forbidden base")
                .num_rows(),
            0
        );
        Ok(())
    }

    #[test]
    fn successful_preparation_binds_and_stages_all_state_until_infallible_commit() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred left_input(u32).
                pred right_input(u32).
                pred out(u32).
                out(X) :- left_input(X).
                out(X) :- right_input(X).
                ?- out(X).
            "#,
        )?;
        let mut base_store = program.create_relation_store(provider.clone())?;
        let left_version = base_store
            .version("left_input")
            .expect("left input version");
        let right_version = base_store
            .version("right_input")
            .expect("right input version");
        let mut runtime =
            Some(program.create_session_runtime(provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        let authoritative_store_pointer = &mut base_store as *mut RelationStore;
        let cache_slot_pointer = &mut cache as *mut Option<LogicMaterializedStore>;
        let runtime_slot_pointer = &mut runtime as *mut Option<LogicSessionRuntime>;
        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![
                (
                    "left_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1])), None),
                ),
                (
                    "right_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[2])), None),
                ),
            ],
            &BTreeSet::new(),
        )?;

        let commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared,
        )?;

        assert!(
            std::ptr::eq(
                &*commit.authoritative_relation_store,
                authoritative_store_pointer
            ),
            "prepared commit must remain bound to its authoritative store"
        );
        assert!(
            std::ptr::eq(&*commit.cached_store_slot, cache_slot_pointer),
            "prepared commit must remain bound to its cache slot"
        );
        assert!(
            std::ptr::eq(&*commit.session_runtime_slot, runtime_slot_pointer),
            "prepared commit must remain bound to its runtime slot"
        );
        assert!(
            commit.cached_store_slot.is_none(),
            "prepared cache must be transaction-owned"
        );
        assert!(
            commit.session_runtime_slot.is_none(),
            "prepared runtime must be transaction-owned"
        );
        assert_eq!(
            commit.authoritative_relation_store.version("left_input"),
            Some(left_version)
        );
        assert_eq!(
            commit.authoritative_relation_store.version("right_input"),
            Some(right_version)
        );
        assert_eq!(
            commit
                .authoritative_relation_store
                .get("left_input")
                .expect("left input base")
                .num_rows(),
            0
        );
        assert_eq!(
            commit
                .authoritative_relation_store
                .get("right_input")
                .expect("right input base")
                .num_rows(),
            0
        );
        assert_eq!(commit.staged_base_updates.len(), 2);
        assert!(commit.prospective_cached_store.is_some());
        assert!(commit.prospective_session_runtime.is_some());

        let prospective_store = commit.prospective_derived_store();
        assert_eq!(
            sorted_u32(
                &provider,
                prospective_store
                    .get("left_input")
                    .expect("prospective left input")
            ),
            vec![1]
        );
        assert_eq!(
            sorted_u32(
                &provider,
                prospective_store
                    .get("right_input")
                    .expect("prospective right input")
            ),
            vec![2]
        );

        provider.memory().reset_alloc_count();
        let report = commit.commit();
        assert_eq!(
            provider.memory().alloc_count(),
            0,
            "commit must issue zero GPU allocation requests because preparation already staged every buffer"
        );

        assert_eq!(base_store.version("left_input"), Some(left_version + 1));
        assert_eq!(base_store.version("right_input"), Some(right_version + 1));
        assert_eq!(
            sorted_u32(
                &provider,
                base_store.get("left_input").expect("committed left input")
            ),
            vec![1]
        );
        assert_eq!(
            sorted_u32(
                &provider,
                base_store
                    .get("right_input")
                    .expect("committed right input")
            ),
            vec![2]
        );
        let result = program.evaluate_cached_relation_store(
            provider.clone(),
            cache.as_ref().expect("committed cache"),
        )?;
        assert_eq!(sorted_u32(&provider, &result.queries[0].buffer), vec![1, 2]);
        assert!(runtime.is_some(), "commit must install the runtime");
        assert_eq!(report_counts(&report), (2, 2, 2, 0, 0));
        Ok(())
    }

    #[test]
    fn prospective_base_snapshot_recomputes_deletion_without_stale_derived_rows() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred fact(u32).
                pred out(u32).
                out(X) :- fact(X).
                ?- out(X).
            "#,
        )?;
        let mut base_store = program.create_relation_store(provider.clone())?;
        base_store.put("fact", test_buffer(&provider, &[1, 2]));
        let mut runtime =
            Some(program.create_session_runtime(provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        drop(base_store.remove("fact").expect("authoritative fact"));

        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![(
                "fact".to_string(),
                RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
            )],
            &BTreeSet::new(),
        )?;
        let commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared,
        )?;

        let prospective_derived = commit.prospective_derived_store();
        assert_eq!(
            sorted_u32(
                &provider,
                prospective_derived
                    .get("__xlog_query_0")
                    .expect("prospective query")
            ),
            vec![1]
        );

        provider.memory().reset_alloc_count();
        let prospective_base = commit.clone_prospective_base_store()?;
        assert_eq!(
            provider.memory().alloc_count(),
            4,
            "one-column authoritative and staged relations should each be cloned exactly once"
        );
        assert_eq!(
            sorted_u32(
                &provider,
                prospective_base
                    .get("fact")
                    .expect("staged missing base relation")
            ),
            vec![1]
        );
        assert_eq!(
            prospective_base
                .get("out")
                .expect("empty authoritative derived relation")
                .num_rows(),
            0
        );
        let (_, independently_recomputed) = program.evaluate_with_relation_store_and_cache(
            provider.clone(),
            &prospective_base,
            false,
        )?;
        assert!(program.relation_stores_query_equivalent(
            provider.as_ref(),
            independently_recomputed.as_relation_store(),
            prospective_derived,
        )?);

        let mut stale_derived_seed = commit.clone_prospective_base_store()?;
        stale_derived_seed.put("out", test_buffer(&provider, &[1, 2]));
        let (_, stale_seed_recompute) = program.evaluate_with_relation_store_and_cache(
            provider.clone(),
            &stale_derived_seed,
            false,
        )?;
        assert_eq!(
            sorted_u32(
                &provider,
                stale_seed_recompute
                    .as_relation_store()
                    .get("__xlog_query_0")
                    .expect("stale-seeded query")
            ),
            vec![1, 2],
            "seeding full recompute with an intensional head retains the deleted row"
        );
        Ok(())
    }

    #[test]
    fn prospective_base_snapshot_skips_superseded_authoritative_buffer_clone() -> Result<()> {
        let Some(provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred fact(u32).
                pred out(u32).
                out(X) :- fact(X).
                ?- out(X).
            "#,
        )?;
        let mut base_store = program.create_relation_store(provider.clone())?;
        base_store.put("fact", test_buffer(&provider, &[1, 2]));
        let mut runtime =
            Some(program.create_session_runtime(provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        let prepared = program.prepare_relation_delta_batch(
            provider.as_ref(),
            vec![(
                "fact".to_string(),
                RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
            )],
            &BTreeSet::new(),
        )?;
        let commit = program.prepare_relation_delta_commit_with_session_runtime(
            provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared,
        )?;

        provider.memory().reset_alloc_count();
        let prospective_base = commit.clone_prospective_base_store()?;
        assert_eq!(
            provider.memory().alloc_count(),
            4,
            "the empty authoritative head and final staged base must each be cloned once"
        );
        assert_eq!(
            sorted_u32(
                &provider,
                prospective_base.get("fact").expect("prospective fact")
            ),
            vec![1]
        );
        Ok(())
    }

    #[test]
    fn prospective_base_clone_budget_failure_discards_prepared_transaction() -> Result<()> {
        let Some(calibration_provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred alpha_input(u32).
                pred stable_input(u32).
            "#,
        )?;
        let stable_rows = (10_000..75_536).collect::<Vec<u32>>();

        let mut calibration_store = program.create_relation_store(calibration_provider.clone())?;
        calibration_store.put(
            "stable_input",
            test_buffer(&calibration_provider, &stable_rows),
        );
        let mut calibration_runtime = Some(program.create_session_runtime(
            calibration_provider.clone(),
            &calibration_store,
            false,
        )?);
        let (_, calibration_cache) = program.evaluate_with_session_runtime(
            calibration_provider.clone(),
            calibration_runtime.as_mut().expect("calibration runtime"),
        )?;
        let mut calibration_cache = Some(calibration_cache);
        let calibration_batch = program.prepare_relation_delta_batch(
            calibration_provider.as_ref(),
            vec![(
                "alpha_input".to_string(),
                RelationDelta::new(Some(test_buffer(&calibration_provider, &[1])), None),
            )],
            &BTreeSet::new(),
        )?;
        calibration_provider.memory().reset_peak();
        let calibration_commit = program.prepare_relation_delta_commit_with_session_runtime(
            calibration_provider.clone(),
            &mut calibration_store,
            &mut calibration_cache,
            &mut calibration_runtime,
            calibration_batch,
        )?;
        assert_eq!(calibration_commit.staged_base_updates.len(), 1);
        assert_eq!(calibration_commit.staged_base_updates[0].0, "alpha_input");
        assert_eq!(calibration_commit.staged_base_updates[0].1.num_rows(), 1);
        let preparation_peak = calibration_provider.memory().peak_bytes();
        drop(calibration_commit);
        drop(calibration_store);
        drop(calibration_cache);
        drop(calibration_runtime);
        drop(calibration_provider);

        let tight_budget = preparation_peak
            .checked_add(4096)
            .expect("calibrated preparation budget must fit in u64");
        let tight_provider = test_provider_with_budget(tight_budget)
            .expect("calibrated byte budget must construct a CUDA provider");
        let mut base_store = program.create_relation_store(tight_provider.clone())?;
        base_store.put("stable_input", test_buffer(&tight_provider, &stable_rows));
        let authoritative_gpu_bytes = tight_provider.memory().allocated_bytes();
        let alpha_version = base_store.version("alpha_input").expect("alpha version");
        let stable_version = base_store.version("stable_input").expect("stable version");
        let mut runtime =
            Some(program.create_session_runtime(tight_provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            tight_provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        let prepared_batch = program.prepare_relation_delta_batch(
            tight_provider.as_ref(),
            vec![(
                "alpha_input".to_string(),
                RelationDelta::new(Some(test_buffer(&tight_provider, &[1])), None),
            )],
            &BTreeSet::new(),
        )?;
        tight_provider.memory().reset_peak();
        let prepared_commit = program.prepare_relation_delta_commit_with_session_runtime(
            tight_provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared_batch,
        )?;
        assert_eq!(prepared_commit.staged_base_updates.len(), 1);
        assert_eq!(prepared_commit.staged_base_updates[0].0, "alpha_input");
        assert_eq!(prepared_commit.staged_base_updates[0].1.num_rows(), 1);
        assert!(
            tight_provider.memory().peak_bytes() <= preparation_peak,
            "identical preparation must fit within the calibrated peak"
        );

        let stable_clone_bytes = u64::try_from(stable_rows.len())
            .expect("stable row count must fit in u64")
            .checked_mul(u64::try_from(std::mem::size_of::<u32>()).expect("u32 width fits in u64"))
            .and_then(|bytes| {
                bytes.checked_add(
                    u64::try_from(std::mem::size_of::<u32>())
                        .expect("device row-count width fits in u64"),
                )
            })
            .expect("stable clone size must fit in u64");
        let staged_column_bytes =
            u64::try_from(std::mem::size_of::<u32>()).expect("staged column width fits in u64");
        let clone_headroom = stable_clone_bytes
            .checked_add(staged_column_bytes - 1)
            .expect("clone headroom must fit in u64");
        let pressure_bytes = tight_provider
            .memory()
            .remaining_bytes()
            .checked_sub(clone_headroom)
            .expect("calibrated provider must have room for the authoritative clone");
        let pressure_len = usize::try_from(pressure_bytes)
            .expect("calibrated pressure allocation must fit in usize");
        let pressure_guard = tight_provider.memory().alloc::<u8>(pressure_len)?;
        let clone_start_bytes = tight_provider.memory().allocated_bytes();
        let expected_current_bytes = clone_start_bytes
            .checked_add(stable_clone_bytes)
            .expect("current bytes before the refused staged clone must fit in u64");
        let expected_required_bytes = expected_current_bytes
            .checked_add(staged_column_bytes)
            .expect("cumulative required bytes must fit in u64");
        assert_eq!(
            expected_required_bytes,
            tight_budget + 1,
            "the calibrated request must exceed the configured budget by exactly one byte"
        );

        let error = match prepared_commit.clone_prospective_base_store() {
            Ok(_) => panic!("one-byte-tight prospective base cloning must fail"),
            Err(error) => error,
        };
        let XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } = &error
        else {
            panic!("expected GPU resource exhaustion, got {error}");
        };
        assert_eq!(*budget_bytes, tight_budget);
        assert_eq!(*estimated_bytes, expected_required_bytes);
        assert_eq!(
            context,
            &format!(
                "cloning staged prospective base relation 'alpha_input': GPU memory pressure: layer=manager_alloc current_bytes={expected_current_bytes} requested_bytes={staged_column_bytes} required_bytes={expected_required_bytes} required_u64_overflow=false budget_bytes={tight_budget} prior_peak_bytes={expected_current_bytes}"
            ),
            "the authoritative base clone must complete before the staged overlay exhausts memory"
        );
        assert_eq!(
            tight_provider.memory().peak_bytes(),
            expected_current_bytes,
            "the refused request must not enter the admitted allocation high-water mark"
        );
        drop(prepared_commit);

        assert!(cache.is_none(), "failed diagnostic must discard its cache");
        assert!(
            runtime.is_none(),
            "failed diagnostic must discard its retained runtime"
        );
        assert_eq!(
            tight_provider.memory().allocated_bytes(),
            authoritative_gpu_bytes + pressure_bytes,
            "dropping the prepared transaction must release every prospective buffer while preserving external memory pressure"
        );
        drop(pressure_guard);
        assert_eq!(
            tight_provider.memory().allocated_bytes(),
            authoritative_gpu_bytes,
            "releasing the pressure allocation must leave only authoritative data"
        );
        assert_eq!(base_store.version("alpha_input"), Some(alpha_version));
        assert_eq!(base_store.version("stable_input"), Some(stable_version));
        assert_eq!(
            base_store
                .get("alpha_input")
                .expect("authoritative alpha")
                .num_rows(),
            0
        );
        assert_eq!(
            sorted_u32(
                &tight_provider,
                base_store
                    .get("stable_input")
                    .expect("authoritative stable input")
            ),
            stable_rows
        );
        Ok(())
    }

    #[test]
    fn later_snapshot_clone_failure_discards_staged_updates_and_derived_state() -> Result<()> {
        let Some(calibration_provider) = test_provider() else {
            return Ok(());
        };
        let program = LogicProgram::compile(
            r#"
                pred alpha_input(u32).
                pred omega_input(u32).
                pred stable_input(u32).
                pred out(u32).
                out(X) :- alpha_input(X).
                out(X) :- omega_input(X).
                ?- out(X).
            "#,
        )?;
        let alpha_rows = (0..512).collect::<Vec<u32>>();
        let omega_rows = (10_000..10_512).collect::<Vec<u32>>();
        let stable_rows = (20_000..85_536).collect::<Vec<u32>>();

        let mut calibration_store = program.create_relation_store(calibration_provider.clone())?;
        calibration_store.put(
            "stable_input",
            test_buffer(&calibration_provider, &stable_rows),
        );
        let mut calibration_runtime = Some(program.create_session_runtime(
            calibration_provider.clone(),
            &calibration_store,
            false,
        )?);
        let (_, calibration_cache) = program.evaluate_with_session_runtime(
            calibration_provider.clone(),
            calibration_runtime.as_mut().expect("calibration runtime"),
        )?;
        let mut calibration_cache = Some(calibration_cache);
        let calibration_batch = program.prepare_relation_delta_batch(
            calibration_provider.as_ref(),
            vec![
                (
                    "alpha_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&calibration_provider, &alpha_rows)), None),
                ),
                (
                    "omega_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&calibration_provider, &omega_rows)), None),
                ),
            ],
            &BTreeSet::new(),
        )?;
        calibration_provider.memory().reset_peak();
        calibration_provider.memory().reset_alloc_count();
        let calibration_commit = program.prepare_relation_delta_commit_with_session_runtime(
            calibration_provider.clone(),
            &mut calibration_store,
            &mut calibration_cache,
            &mut calibration_runtime,
            calibration_batch,
        )?;
        let successful_peak = calibration_provider.memory().peak_bytes();
        let successful_preparation_allocations = calibration_provider.memory().alloc_count();
        assert!(
            successful_peak >= calibration_provider.memory().allocated_bytes(),
            "the calibrated peak must cover every live staged GPU allocation"
        );
        drop(calibration_commit);
        drop(calibration_store);
        drop(calibration_cache);
        drop(calibration_runtime);
        drop(calibration_provider);

        let tight_budget = successful_peak
            .checked_sub(1)
            .expect("successful preparation must allocate GPU memory");
        let tight_provider = test_provider_with_budget(tight_budget)
            .expect("calibrated budget must still construct a CUDA provider");
        let mut base_store = program.create_relation_store(tight_provider.clone())?;
        base_store.put("stable_input", test_buffer(&tight_provider, &stable_rows));
        let authoritative_gpu_bytes = tight_provider.memory().allocated_bytes();
        let alpha_version = base_store.version("alpha_input").expect("alpha version");
        let omega_version = base_store.version("omega_input").expect("omega version");
        let stable_version = base_store.version("stable_input").expect("stable version");
        let mut runtime =
            Some(program.create_session_runtime(tight_provider.clone(), &base_store, false)?);
        let (_, initial_cache) = program.evaluate_with_session_runtime(
            tight_provider.clone(),
            runtime.as_mut().expect("initial runtime"),
        )?;
        let mut cache = Some(initial_cache);
        let prepared = program.prepare_relation_delta_batch(
            tight_provider.as_ref(),
            vec![
                (
                    "alpha_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&tight_provider, &alpha_rows)), None),
                ),
                (
                    "omega_input".to_string(),
                    RelationDelta::new(Some(test_buffer(&tight_provider, &omega_rows)), None),
                ),
            ],
            &BTreeSet::new(),
        )?;

        tight_provider.memory().reset_alloc_count();
        let final_row_count_clone_bytes =
            u64::try_from(std::mem::size_of::<u32>()).expect("device row-count width fits in u64");
        let expected_current_bytes = tight_budget
            .checked_sub(final_row_count_clone_bytes - 1)
            .expect("one-byte-tight budget must cover earlier snapshot clones");
        let expected_required_bytes = expected_current_bytes
            .checked_add(final_row_count_clone_bytes)
            .expect("cumulative required bytes must fit in u64");
        assert_eq!(
            expected_required_bytes,
            tight_budget + 1,
            "the calibrated final clone must exceed the configured budget by exactly one byte"
        );
        let error = match program.prepare_relation_delta_commit_with_session_runtime(
            tight_provider.clone(),
            &mut base_store,
            &mut cache,
            &mut runtime,
            prepared,
        ) {
            Ok(_) => panic!("one-byte-tight preparation must fail during the final clone"),
            Err(error) => error,
        };
        let XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } = &error
        else {
            panic!("expected GPU resource exhaustion, got {error}");
        };
        assert_eq!(*budget_bytes, tight_budget);
        assert_eq!(*estimated_bytes, expected_required_bytes);
        assert_eq!(
            context,
            &format!(
                "cloning prospective relation snapshot 'stable_input': GPU memory pressure: layer=manager_alloc current_bytes={expected_current_bytes} requested_bytes={final_row_count_clone_bytes} required_bytes={expected_required_bytes} required_u64_overflow=false budget_bytes={tight_budget} prior_peak_bytes={expected_current_bytes}"
            )
        );
        assert_eq!(
            tight_provider.memory().peak_bytes(),
            expected_current_bytes,
            "the refused final clone must leave the high-water mark at the last admitted allocation"
        );
        let failed_preparation_allocations = tight_provider.memory().alloc_count();
        assert!(
            successful_preparation_allocations > 4,
            "calibration must include both staged-base and snapshot clones"
        );
        assert_eq!(
            failed_preparation_allocations,
            successful_preparation_allocations,
            "the one-byte-tight run must reach the final calibrated clone allocation after every earlier staged clone succeeds"
        );
        assert!(cache.is_none(), "failed preparation must discard its cache");
        assert!(
            runtime.is_none(),
            "failed preparation must discard its runtime"
        );
        assert_eq!(
            tight_provider.memory().allocated_bytes(),
            authoritative_gpu_bytes,
            "failed preparation must release every transaction-owned GPU allocation"
        );
        assert_eq!(base_store.version("alpha_input"), Some(alpha_version));
        assert_eq!(base_store.version("omega_input"), Some(omega_version));
        assert_eq!(base_store.version("stable_input"), Some(stable_version));
        assert_eq!(
            base_store
                .get("alpha_input")
                .expect("authoritative alpha")
                .num_rows(),
            0
        );
        assert_eq!(
            base_store
                .get("omega_input")
                .expect("authoritative omega")
                .num_rows(),
            0
        );
        assert_eq!(
            sorted_u32(
                &tight_provider,
                base_store
                    .get("stable_input")
                    .expect("authoritative stable input")
            ),
            stable_rows
        );
        Ok(())
    }
}
