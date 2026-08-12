//! GPU-accelerated evaluation of compiled Datalog programs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use xlog_core::{RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
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
use xlog_runtime::{
    DeltaRecomputeStats, EpistemicGpuExecutionResult, EpistemicGpuWorkspaceCapacities,
    ExecutionStats, Executor, RelationDelta, RelationStore,
};

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
    executor: Executor,
    profiling: bool,
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
///     LogicProgram, LogicSessionRuntime, PreparedRelationDeltaBatch,
/// };
/// use xlog_runtime::RelationStore;
///
/// fn mutate_after_prepare(
///     program: &LogicProgram,
///     provider: Arc<CudaKernelProvider>,
///     store: &mut RelationStore,
///     cache: &mut Option<RelationStore>,
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
    cached_store_slot: &'a mut Option<RelationStore>,
    session_runtime_slot: &'a mut Option<LogicSessionRuntime>,
    staged_base_updates: Vec<(String, CudaBuffer)>,
    prospective_cached_store: Option<RelationStore>,
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
            return store;
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
    EpistemicStratified(Vec<StratumExecutable>),
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
    pub fn compile_program(program: Program) -> Result<Self> {
        let source_program = program.clone();
        let normalized = normalize_program_for_execution(program)?;
        Self::compile_normalized_program(normalized, source_program)
    }

    fn compile_normalized_program(normalized: Program, source_program: Program) -> Result<Self> {
        // Function, meta-term, list, and shared-variable normalization preserve
        // constraint count and source order. Keep the authored snapshot only
        // while that one-to-one invariant remains observable.
        let authored_constraints = (source_program.constraints.len()
            == normalized.constraints.len())
        .then(|| source_program.constraints.clone());
        if program_has_epistemic_literals(&normalized) {
            return Self::compile_epistemic_program(
                normalized,
                source_program,
                authored_constraints,
            );
        }
        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&normalized)?;
        Ok(Self {
            source_program,
            program: normalized,
            authored_constraints,
            plan: LogicExecutionPlan::Ordinary(Box::new(plan)),
            schemas: compiler.schemas().clone(),
            rel_ids: compiler.rel_ids().clone(),
            epistemic_provenance: None,
        })
    }

    fn compile_epistemic_program(
        normalized: Program,
        source_program: Program,
        authored_constraints: Option<Vec<Constraint>>,
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
            schema_compiler.compile_program(&reduced)?;
            let mut schemas = schema_compiler.schemas().clone();
            augment_same_name_multi_arity_schemas(active_program, &mut schemas)?;

            let mut strata = Vec::with_capacity(stratified.strata.len());
            for stratum in &stratified.strata {
                strata.push(StratumExecutable {
                    plan: Self::compile_stratum_plan(&stratum.program)?,
                });
            }
            let plan = LogicExecutionPlan::EpistemicStratified(strata);
            let rel_ids = epistemic_relation_ids(&plan)?;
            return Ok(Self {
                source_program,
                program: normalized,
                authored_constraints,
                plan,
                schemas,
                rel_ids,
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "stratified",
                    literals: provenance_literals,
                    surface_source_queries: false,
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
            let plan = compiler.compile_program(&recursive_reduced)?;
            return Ok(Self {
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
        schema_compiler.compile_program(&reduced)?;
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
            let plan = compiler.compile_program(&case_a_reduced)?;
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
    /// summaries, the forbidden CPU-fallback counters (which must all be zero on
    /// the accepted GPU hot path), and a deterministic plan id (a stable hash of
    /// the canonical summary). This is the epistemic-plan/EIR dump surface:
    /// it lets an external caller (pyxlog or CLI consumer) read the accepted
    /// world-view structure and assert `cpu_fallback == 0` off a real run.
    pub fn epistemic_plan_json(&self) -> Option<String> {
        let gpu_plans: Vec<(String, &xlog_ir::EpistemicGpuPlan)> = match &self.plan {
            // A program whose source was epistemic but whose executable plan is
            // ordinary either resolved admissible recursive modal literals into joins
            // or removed every unfounded FAEEL modal rule. It carries no epistemic GPU
            // plan and executes through the ordinary GPU engine with no epistemic CPU
            // fallback. Emit a provenance summary with a stable id so the reduction is
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
            LogicExecutionPlan::EpistemicStratified(strata) => {
                let mut plans = Vec::new();
                for (i, stratum) in strata.iter().enumerate() {
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
                        // a materialized base); they contribute no fallback counters.
                        StratumPlanKind::Ordinary { .. } => {}
                    }
                }
                plans
            }
        };
        Some(epistemic_plan_summary_json(
            self.plan_kind_label(),
            &gpu_plans,
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
    ) -> Result<(LogicEvalResult, RelationStore)> {
        let mut executor =
            self.executor_from_relation_store(provider.clone(), relation_store, profiling)?;
        executor.execute_plan(self.ordinary_plan("relation-store evaluation")?)?;
        self.enforce_constraints(&provider, &executor)?;

        let total_output_rows = self.total_query_rows(executor.store())?;
        let stats = if profiling {
            Some(executor.execution_stats(total_output_rows))
        } else {
            None
        };

        let cached_store = self.clone_relation_store(&provider, executor.store())?;
        let result = self.logic_result_from_store(provider.as_ref(), &cached_store, stats)?;
        Ok((result, cached_store))
    }

    /// Create retained runtime state for a persistent relation session.
    pub fn create_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
        profiling: bool,
    ) -> Result<LogicSessionRuntime> {
        self.ordinary_plan("persistent relation session")?;
        Ok(LogicSessionRuntime {
            executor: self.executor_from_relation_store(provider, relation_store, profiling)?,
            profiling,
        })
    }

    /// Evaluate with retained session runtime state and return a materialized store snapshot.
    pub fn evaluate_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        runtime: &mut LogicSessionRuntime,
    ) -> Result<(LogicEvalResult, RelationStore)> {
        runtime.executor.set_profiling(runtime.profiling);
        runtime
            .executor
            .execute_plan(self.ordinary_plan("session runtime evaluation")?)?;
        self.enforce_constraints(&provider, &runtime.executor)?;

        let total_output_rows = self.total_query_rows(runtime.executor.store())?;
        let stats = if runtime.profiling {
            Some(runtime.executor.execution_stats(total_output_rows))
        } else {
            None
        };

        let cached_store = self.clone_relation_store(&provider, runtime.executor.store())?;
        let result = self.logic_result_from_store(provider.as_ref(), &cached_store, stats)?;
        Ok((result, cached_store))
    }

    /// Build query results from an already materialized runtime store.
    pub fn evaluate_cached_relation_store(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &RelationStore,
    ) -> Result<LogicEvalResult> {
        self.logic_result_from_store(provider.as_ref(), relation_store, None)
    }

    /// Apply relation deltas to a persistent session store through the runtime delta path.
    ///
    /// If preparation fails, the authoritative relation store is unchanged but
    /// any prior derived cache consumed by preparation is discarded. The caller
    /// must rebuild that cache on its next evaluation.
    pub fn apply_relation_deltas(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
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
    /// If preparation fails, the authoritative relation store is unchanged but
    /// the derived cache and retained runtime slots are left empty. The caller
    /// must rebuild them on its next evaluation.
    pub fn apply_relation_deltas_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
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
    /// The current derived cache and runtime are consumed during preparation;
    /// on error their caller slots remain empty while the authoritative store
    /// remains unchanged.
    pub fn prepare_relation_deltas_commit_with_session_runtime<'a>(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &'a mut RelationStore,
        cached_store: &'a mut Option<RelationStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
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
        coalesce_relation_delta_batch_with_cancellation_capture(
            provider,
            delta_batch,
            cancellation_capture_relations,
        )
    }

    /// Prepare a fully staged retained-runtime commit from a coalesced batch.
    ///
    /// The current derived runtime and cache are moved into the transaction. If
    /// preparation fails, those partially updated values are discarded and the
    /// caller slots remain empty; the authoritative base store is never changed.
    pub fn prepare_relation_delta_commit_with_session_runtime<'a>(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &'a mut RelationStore,
        cached_store: &'a mut Option<RelationStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        prepared_batch: PreparedRelationDeltaBatch,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
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
        cached_store: &'a mut Option<RelationStore>,
        session_runtime: &'a mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
        report_seed: Option<PreparedRelationDeltaReportSeed>,
    ) -> Result<PreparedRelationDeltaCommit<'a>> {
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
                let seed_store = prior_cached_store.as_ref().unwrap_or(relation_store);
                self.create_session_runtime(provider.clone(), seed_store, false)?
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
        let prospective_cached_store = Some(
            self.clone_prepared_relation_snapshot(&provider, working_runtime.executor.store())?,
        );

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
    /// fails, the authoritative store remains unchanged and any consumed
    /// derived cache is discarded.
    pub fn apply_relation_delta_batch(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
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
    /// and retained runtime slots are left empty.
    pub fn apply_relation_delta_batch_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
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
        let mut executor = self.prepare_executor(&provider, inputs, profiling)?;

        if let LogicExecutionPlan::EpistemicG91Compatibility(g91_plan) = &self.plan {
            return self
                .evaluate_g91_compatibility_gpu_program(provider, executor, g91_plan, profiling);
        }

        if let LogicExecutionPlan::EpistemicWfsGpu(wfs_plan) = &self.plan {
            return self.evaluate_wfs_gpu_program(provider, executor, wfs_plan, profiling);
        }

        let LogicExecutionPlan::Ordinary(plan) = &self.plan else {
            return self.evaluate_epistemic_with_executor(executor, profiling);
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

    fn executor_from_relation_store(
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
        let (relation_name, buffer) = if surface_source_query {
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
            (presentation_query.atom.predicate.clone(), buffer)
        } else {
            (internal_relation_name, buffer)
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
            if pass.schemas.contains_key(name) {
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
        mut executor: Executor,
        profiling: bool,
    ) -> Result<LogicEvalResult> {
        let mut queries = Vec::new();
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
                let batch = executor.execute_epistemic_gpu_execution_batch_with_trace(
                    &executables,
                    capacities_for_epistemic_split(split)?,
                )?;
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
            LogicExecutionPlan::EpistemicStratified(strata) => {
                // Execute strata in topological order on the SAME executor. After
                // each stratum, write its GATED head output(s) into the store as
                // base relations so the NEXT stratum's `know`/`possible` over a
                // lower head reads the gated extension through the existing tuple-key
                // membership filter (or, once the head is a materialized base
                // relation, Case-A resolve-into-body) — never double-gating against
                // a still-modal relation.
                //
                // A head is surfaced as a user-visible query result when the source
                // program explicitly queries it (`?- head(...)`), regardless of
                // which stratum produced it; otherwise only the TOP stratum's heads
                // are surfaced (lower-stratum heads are intermediate, materialized
                // for gating only).
                let queried_predicates: BTreeSet<&str> = self
                    .program
                    .queries
                    .iter()
                    .map(|query| query.atom.predicate.as_str())
                    .collect();
                let stratum_count = strata.len();
                for (stratum_index, stratum) in strata.iter().enumerate() {
                    let is_last = stratum_index + 1 == stratum_count;
                    match &stratum.plan {
                        StratumPlanKind::Single(executable) => {
                            let result = executor.execute_epistemic_gpu_execution(
                                executable,
                                capacities_for_epistemic_executable(executable)?,
                            )?;
                            result.require_runtime_dispatch_certification()?;
                            let primary_head = epistemic_output_relation_name(executable)?;
                            Self::materialize_and_surface_epistemic_stratum_result(
                                &mut executor,
                                primary_head,
                                result,
                                is_last,
                                &queried_predicates,
                                &mut queries,
                            )?;
                        }
                        StratumPlanKind::Split(split) => {
                            let executables: Vec<_> = split
                                .components
                                .iter()
                                .map(|component| &component.executable)
                                .collect();
                            let batch = executor.execute_epistemic_gpu_execution_batch_with_trace(
                                &executables,
                                capacities_for_epistemic_split(split)?,
                            )?;
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
                                Self::materialize_and_surface_epistemic_stratum_result(
                                    &mut executor,
                                    primary_head,
                                    result,
                                    is_last,
                                    &queried_predicates,
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
                                if is_last || queried_predicates.contains(head.as_str()) {
                                    let buffer =
                                        executor.store().get(head.as_str()).ok_or_else(|| {
                                            XlogError::Execution(format!(
                                                "missing stratified ordinary stratum output \
                                                 relation {head}"
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
            }
            LogicExecutionPlan::EpistemicG91Compatibility(_)
            | LogicExecutionPlan::EpistemicWfsGpu(_) => {
                unreachable!("iterative GPU epistemic plans are handled earlier")
            }
            LogicExecutionPlan::Ordinary(_) => {
                unreachable!("ordinary plans are handled earlier")
            }
        }

        let total_output_rows: u64 = queries.iter().map(|q| q.buffer.num_rows()).sum();
        let stats = if profiling {
            Some(executor.execution_stats(total_output_rows))
        } else {
            None
        };
        Ok(LogicEvalResult { queries, stats })
    }

    /// Materialize one epistemic stratum result's GATED head(s) into the store and
    /// surface them as query results when appropriate.
    ///
    /// Every gated head (primary `final_output` plus joint additional heads) is
    /// written to the store so higher strata can gate against it. A head is added
    /// to `queries` when its stratum is the TOP stratum OR the source program
    /// explicitly queries it.
    fn materialize_and_surface_epistemic_stratum_result(
        executor: &mut Executor,
        primary_head: String,
        result: EpistemicGpuExecutionResult,
        is_last: bool,
        queried_predicates: &BTreeSet<&str>,
        queries: &mut Vec<LogicQueryResult>,
    ) -> Result<()> {
        executor.materialize_epistemic_head_relation(&primary_head, &result.final_output)?;
        for (head, buffer) in &result.additional_head_outputs {
            executor.materialize_epistemic_head_relation(head, buffer)?;
        }

        // Collect the heads to surface: primary + additional, filtered by
        // top-stratum-or-explicitly-queried.
        let surface_primary = is_last || queried_predicates.contains(primary_head.as_str());
        let additional_heads: Vec<String> = result
            .additional_head_outputs
            .iter()
            .map(|(head, _)| head.clone())
            .collect();

        let mut all_results = epistemic_result_to_query_results(primary_head.clone(), result);
        all_results.retain(|query_result| {
            if query_result.relation_name == primary_head {
                surface_primary
            } else {
                is_last
                    || (additional_heads.contains(&query_result.relation_name)
                        && queried_predicates.contains(query_result.relation_name.as_str()))
            }
        });
        queries.extend(all_results);
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
            .and_then(|constraints| constraints.get(constraint_index))
            .or_else(|| self.source_program.constraints.get(constraint_index))
            .unwrap_or(&self.program.constraints[constraint_index]);
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
            } if constraint_index < self.program.constraints.len() => {
                self.constraint_violation_error(constraint_index)
            }
            other => other,
        }
    }

    fn enforce_constraints_in_store(
        &self,
        provider: &CudaKernelProvider,
        store: &RelationStore,
    ) -> Result<()> {
        for i in 0..self.program.constraints.len() {
            let name = format!("__xlog_constraint_{}", i);
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
pub fn normalize_program_for_execution(program: Program) -> Result<Program> {
    let max_recursion = program.directives.max_recursion_depth_or_default();
    let expanded = xlog_logic::expand_program_functions(&program, max_recursion)
        .map_err(|e| XlogError::Compilation(e.to_string()))?;
    let normalized = xlog_logic::normalize_meta_builtins(&expanded)?;
    let listed = xlog_logic::normalize_list_builtins(&normalized)?;
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
    compiler.compile_program(&inference_program)?;
    Ok(compiler.schemas().clone())
}

fn compile_gpu_ordinary_pass(program: &Program) -> Result<GpuOrdinaryPass> {
    let mut compiler = Compiler::new();
    let plan = compiler.compile_program(program)?;
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
    let predicates = epistemic_extensional_multi_arity_predicates(program);
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
        LogicExecutionPlan::EpistemicStratified(strata) => {
            for stratum in strata {
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
/// modal literals are recorded and CPU fallback is zero by construction.
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
    let host_wfs_fallback_allowed = if wfs.is_some() { "false" } else { "null" };
    let body = format!(
        "{{\"plan_kind\":\"{}\",\"reduction\":\"{}\",\
\"epistemic_literals\":[{}],\"units\":[],\"max_iterations\":{},\
\"wfs_fixed_relations\":{},\"wfs_convergence_predicates\":{},\
\"wfs_gpu_passes\":{},\
\"host_wfs_fallback_allowed\":{},\
\"cpu_fallback_total_zero\":true}}",
        json_escape(plan_kind),
        json_escape(prov.reduction),
        literals,
        max_iterations
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        wfs_fixed_relations,
        wfs_convergence_predicates,
        wfs_gpu_passes,
        host_wfs_fallback_allowed
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{:016x}\",\"plan_kind\":\"{}\",\
\"reduction\":\"{}\",\"epistemic_literals\":[{}],\"units\":[],\
\"max_iterations\":{},\"wfs_fixed_relations\":{},\
\"wfs_convergence_predicates\":{},\"wfs_gpu_passes\":{},\
\"host_wfs_fallback_allowed\":{},\
\"cpu_fallback_total_zero\":true}}",
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
        host_wfs_fallback_allowed
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
\"cpu_fallback_total_zero\":true}}",
        json_escape(plan_kind),
        json_escape(provenance.reduction),
        literals,
        plan.max_iterations,
        snapshots,
        convergence
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{plan_id:016x}\",\"plan_kind\":\"{}\",\
\"reduction\":\"{}\",\"epistemic_literals\":[{}],\"units\":[],\
\"max_iterations\":{},\"snapshot_relations\":{{{}}},\
\"convergence_predicates\":[{}],\
\"gpu_passes\":[\"upper_bound\",\"refinement\"],\
\"cpu_fallback_total_zero\":true}}",
        json_escape(plan_kind),
        json_escape(provenance.reduction),
        literals,
        plan.max_iterations,
        snapshots,
        convergence
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
    let f = &plan.cpu_fallbacks;
    format!(
        "{{\"mode\":\"{}\",\"epistemic_literals\":[{}],\"required_phases\":[{}],\
\"required_kernel_phases\":[{}],\"constraints\":[{}],\"reductions\":[{}],\
\"cpu_fallbacks\":{{\"candidate_enumeration\":{},\"world_view_validation\":{},\
\"solver_search\":{},\"probabilistic_recompute\":{}}},\"cpu_fallback_is_zero\":{}}}",
        mode,
        literals,
        phases,
        kernels,
        constraints,
        reductions,
        f.candidate_enumeration,
        f.world_view_validation,
        f.solver_search,
        f.probabilistic_recompute,
        f.is_zero()
    )
}

fn epistemic_plan_summary_json(
    plan_kind: &str,
    gpu_plans: &[(String, &xlog_ir::EpistemicGpuPlan)],
) -> String {
    let units = gpu_plans
        .iter()
        .map(|(label, plan)| {
            format!(
                "{{\"unit\":\"{}\",\"plan\":{}}}",
                json_escape(label),
                epistemic_gpu_plan_json(plan)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let all_zero = gpu_plans
        .iter()
        .all(|(_, plan)| plan.cpu_fallbacks.is_zero());
    // Canonical body (without the id) hashed for the stable plan id.
    let body = format!(
        "{{\"plan_kind\":\"{}\",\"units\":[{}],\"cpu_fallback_total_zero\":{}}}",
        json_escape(plan_kind),
        units,
        all_zero
    );
    let plan_id = fnv1a_64(&body);
    format!(
        "{{\"plan_id\":\"epi-{:016x}\",\"plan_kind\":\"{}\",\"units\":[{}],\"cpu_fallback_total_zero\":{}}}",
        plan_id,
        json_escape(plan_kind),
        units,
        all_zero
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
    use xlog_cuda::{CudaDevice, GpuMemoryManager};

    fn ground_term_encoding_test_provider() -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            let device = Arc::new(CudaDevice::new(0)?);
            let memory = Arc::new(GpuMemoryManager::new(
                device.clone(),
                MemoryBudget::with_limit(256 * 1024 * 1024),
            ));
            Ok(Arc::new(CudaKernelProvider::new(device, memory)?))
        })();

        finish_test_provider_setup(
            provider,
            std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1"),
        )
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
        assert!(summary.contains("\"cpu_fallback_total_zero\":true"));
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
                context: "GPU memory allocation".to_string(),
                estimated_bytes: 64,
                budget_bytes: 32,
            },
        );
        match exhausted {
            XlogError::ResourceExhausted {
                context,
                estimated_bytes,
                budget_bytes,
            } => {
                assert_eq!(context, "cloning relation 'fact': GPU memory allocation");
                assert_eq!(estimated_bytes, 64);
                assert_eq!(budget_bytes, 32);
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
    use xlog_cuda::{CudaDevice, GpuMemoryManager};

    fn test_provider() -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            let device = Arc::new(CudaDevice::new(0)?);
            let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
            let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
            Ok(Arc::new(CudaKernelProvider::new(device, memory)?))
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
    use xlog_cuda::{CudaDevice, GpuMemoryManager};

    fn test_provider_with_budget(limit: u64) -> Option<Arc<CudaKernelProvider>> {
        let provider = (|| -> Result<Arc<CudaKernelProvider>> {
            let device = Arc::new(CudaDevice::new(0)?);
            let budget = MemoryBudget::with_limit(limit);
            let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
            Ok(Arc::new(CudaKernelProvider::new(device, memory)?))
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
        let cache_slot_pointer = &mut cache as *mut Option<RelationStore>;
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
            &independently_recomputed,
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
        assert_eq!(*estimated_bytes, staged_column_bytes);
        assert_eq!(
            context,
            "cloning staged prospective base relation 'alpha_input': GPU memory allocation",
            "the authoritative base clone must complete before the staged overlay exhausts memory"
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
            budget_bytes,
            ..
        } = &error
        else {
            panic!("expected GPU resource exhaustion, got {error}");
        };
        assert_eq!(*budget_bytes, tight_budget);
        assert_eq!(
            context,
            "cloning prospective relation snapshot 'stable_input': GPU memory allocation"
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
