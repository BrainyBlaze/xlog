//! GPU-accelerated evaluation of compiled Datalog programs.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use xlog_core::{symbol, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{EpistemicExecutablePlan, ExecutionPlan};
use xlog_logic::ast::{AggOp, PredColumn, TypeRef};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_split_execution,
    reduce_epistemic_program_to_ordinary,
    reduce_epistemic_program_to_ordinary_for_stratified_schema,
    try_plan_stratified_epistemic_program, try_reduce_case_a_recursive_epistemic_program,
    EpistemicSplitExecutablePlan,
};
use xlog_logic::{
    Atom, BodyLiteral, Compiler, EpistemicLiteral, EpistemicOp, Program, Query, Rule, Term,
};
use xlog_runtime::executor::JoinIndexCacheStats;
use xlog_runtime::{
    DeltaRecomputeStats, EpistemicGpuExecutionResult, EpistemicGpuWorkspaceCapacities,
    ExecutionStats, Executor, RelationDelta, RelationStore,
};

/// Result of evaluating a single query in a Datalog program.
pub struct LogicQueryResult {
    /// Internal relation name (e.g. `__xlog_query_0`).
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

struct CoalescedRelationDeltaBatch {
    deltas: HashMap<String, RelationDelta>,
    input_delta_count: usize,
    changed_relations: usize,
    coalesced_insert_rows: u64,
    coalesced_delete_rows: u64,
    canceled_rows: u64,
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
    Single(EpistemicExecutablePlan),
    Split(EpistemicSplitExecutablePlan),
    /// A higher stratum that RECURSES over a lower stratum's materialized
    /// (now-base) determined head. Once the determined head is a base relation in
    /// the store, its `know`/`possible` modal is over an invariant relation, so the
    /// stratum is admissible Case-A: the modal resolves to an ordinary join (no
    /// second gate) and the recursive semi-naive engine iterates the fixpoint. The
    /// reduced ordinary program drives an ordinary RIR plan whose head IS this
    /// stratum's user-visible output relation.
    Ordinary {
        plan: ExecutionPlan,
        /// User-visible output head predicate(s) this stratum computes.
        head_predicates: Vec<String>,
    },
}

#[derive(Clone)]
enum LogicExecutionPlan {
    Ordinary(ExecutionPlan),
    EpistemicWfsGpu(EpistemicWfsGpuPlan),
    EpistemicSingle(EpistemicExecutablePlan),
    EpistemicSplit(EpistemicSplitExecutablePlan),
    /// Stratified epistemic execution: ordered strata, each materializing its
    /// gated head(s) into the store before the next stratum runs.
    EpistemicStratified(Vec<StratumExecutable>),
}

#[derive(Clone)]
struct EpistemicWfsGpuPlan {
    overapprox: WfsGpuOrdinaryPlan,
    lower: WfsGpuOrdinaryPlan,
    upper: WfsGpuOrdinaryPlan,
    intensional_predicates: Vec<String>,
    upper_fixed_names: HashMap<String, String>,
    lower_fixed_names: HashMap<String, String>,
    max_iterations: usize,
}

#[derive(Clone)]
struct WfsGpuOrdinaryPlan {
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
}

/// A compiled Datalog program ready for GPU evaluation.
#[derive(Clone)]
pub struct LogicProgram {
    program: Program,
    plan: LogicExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
    /// `Some` iff the source program contained epistemic literals (regardless of
    /// whether the executable plan ended up epistemic or ordinary).
    epistemic_provenance: Option<EpistemicProvenance>,
}

impl LogicProgram {
    /// Compile a Datalog source string into a GPU-executable program.
    pub fn compile(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;
        let normalized = normalize_program(program)?;
        Self::compile_normalized_program(normalized)
    }

    fn compile_normalized_program(normalized: Program) -> Result<Self> {
        if program_has_epistemic_literals(&normalized) {
            return Self::compile_epistemic_program(normalized);
        }
        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&normalized)?;
        Ok(Self {
            program: normalized,
            plan: LogicExecutionPlan::Ordinary(plan),
            schemas: compiler.schemas().clone(),
            rel_ids: compiler.rel_ids().clone(),
            epistemic_provenance: None,
        })
    }

    fn compile_epistemic_program(normalized: Program) -> Result<Self> {
        // Capture epistemic provenance up front: the source-EIR modal literals are
        // retained even when a Case-A recursive reduction lowers the program to an
        // Ordinary executable plan, so the epistemic plan dump can still emit a stable id
        // for a recursive epistemic fixpoint.
        let provenance_literals = collect_eir_epistemic_literals(&normalized);
        // Stratified epistemic execution FIRST: a modal literal ranges over an
        // epistemically-DETERMINED derived head (`b :- know a` where `a :- know p`,
        // `p` invariant — possibly with the higher stratum RECURSING over the
        // determined head, e.g. `reach :- reach, know a`). Partition into strata;
        // each is compiled through the existing epistemic OR Case-A ordinary path,
        // and at runtime each lower stratum's GATED head is materialized into the
        // store as a base relation before the higher stratum gates against it (via
        // the existing tuple-key membership filter or — once the head is a materialized
        // base relation — Case-A resolve-into-body; either way NO double-gating
        // against a still-modal relation). Example 18's shared BASE modal `q` (EDB,
        // not a determined derived head) returns `None` here and falls through to
        // the joint path UNCHANGED; plain Case-A recursion over an EDB modal
        // (`know edge`) also returns `None` and falls through to Case-A below.
        if let Some(stratified) = try_plan_stratified_epistemic_program(&normalized)? {
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
            let reduced = reduce_epistemic_program_to_ordinary_for_stratified_schema(&normalized);
            let mut schema_compiler = Compiler::new();
            schema_compiler.compile_program(&reduced)?;
            let mut schemas = schema_compiler.schemas().clone();
            augment_same_name_multi_arity_schemas(&normalized, &mut schemas)?;

            let mut strata = Vec::with_capacity(stratified.strata.len());
            for stratum in &stratified.strata {
                strata.push(StratumExecutable {
                    plan: Self::compile_stratum_plan(&stratum.program)?,
                });
            }
            let plan = LogicExecutionPlan::EpistemicStratified(strata);
            let rel_ids = epistemic_relation_ids(&plan)?;
            return Ok(Self {
                program: normalized,
                plan,
                schemas,
                rel_ids,
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "stratified",
                    literals: provenance_literals,
                }),
            });
        }

        // Case A/B: reduce admissible recursive epistemic programs to ordinary
        // recursion. Stratified reduced programs route through the existing ordinary
        // semi-naive engine; non-monotone reduced SCCs route through the GPU-native
        // WFS alternating-fixpoint plan below. Recursive shapes outside the admissible
        // fragment still fail closed in `try_reduce_case_a_recursive_epistemic_program`.
        if let Some(case_a_reduced) = try_reduce_case_a_recursive_epistemic_program(&normalized)? {
            let strat = xlog_logic::stratify::analyze_stratification(&case_a_reduced);
            if !strat.non_monotone_sccs.is_empty() {
                let wfs_plan = compile_epistemic_wfs_gpu_plan(&case_a_reduced)?;
                let schemas = wfs_plan_combined_schemas(&wfs_plan);
                let rel_ids = wfs_plan_combined_rel_ids(&wfs_plan)?;
                return Ok(Self {
                    program: case_a_reduced,
                    plan: LogicExecutionPlan::EpistemicWfsGpu(wfs_plan),
                    schemas,
                    rel_ids,
                    epistemic_provenance: Some(EpistemicProvenance {
                        reduction: "wfs_gpu_recursive",
                        literals: provenance_literals,
                    }),
                });
            }
            let mut compiler = Compiler::new();
            let plan = compiler.compile_program(&case_a_reduced)?;
            return Ok(Self {
                program: case_a_reduced,
                plan: LogicExecutionPlan::Ordinary(plan),
                schemas: compiler.schemas().clone(),
                rel_ids: compiler.rel_ids().clone(),
                epistemic_provenance: Some(EpistemicProvenance {
                    reduction: "ordinary_recursive_modal_reduction",
                    literals: provenance_literals,
                }),
            });
        }

        let reduced = reduce_epistemic_program_to_ordinary(&normalized);
        let mut schema_compiler = Compiler::new();
        schema_compiler.compile_program(&reduced)?;
        let mut schemas = schema_compiler.schemas().clone();
        augment_same_name_multi_arity_schemas(&normalized, &mut schemas)?;

        let plan = if epistemic_output_head_predicate_count(&normalized) > 1 {
            LogicExecutionPlan::EpistemicSplit(compile_epistemic_gpu_split_execution(&normalized)?)
        } else {
            match compile_epistemic_gpu_execution(&normalized) {
                Ok(executable) => LogicExecutionPlan::EpistemicSingle(executable),
                Err(XlogError::UnsupportedEpistemicConstruct { construct, .. })
                    if construct == "epistemic GPU final output relation" =>
                {
                    LogicExecutionPlan::EpistemicSplit(compile_epistemic_gpu_split_execution(
                        &normalized,
                    )?)
                }
                Err(err) => return Err(err),
            }
        };
        let rel_ids = epistemic_relation_ids(&plan)?;

        Ok(Self {
            program: normalized,
            plan,
            schemas,
            rel_ids,
            epistemic_provenance: Some(EpistemicProvenance {
                reduction: "epistemic_executable",
                literals: provenance_literals,
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
                plan,
                head_predicates,
            });
        }
        if epistemic_output_head_predicate_count(stratum_program) > 1 {
            Ok(StratumPlanKind::Split(
                compile_epistemic_gpu_split_execution(stratum_program)?,
            ))
        } else {
            Ok(StratumPlanKind::Single(compile_epistemic_gpu_execution(
                stratum_program,
            )?))
        }
    }

    /// Compile a program with module resolution.
    ///
    /// This method resolves all imports using the provided resolver and merges
    /// imported predicates, functions, and rules into the main program.
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

        let normalized = normalize_program(merged)?;
        Self::compile_normalized_program(normalized)
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
            // ordinary: this is a Case-A recursive epistemic fixpoint (modal literals
            // resolved into invariant joins). It carries no epistemic GPU plan, but it
            // IS GPU-clean by construction (the recursion runs on the ordinary
            // semi-naive engine with no epistemic CPU fallback). Emit a provenance
            // summary with a stable id so the recursive-fixpoint fixture is auditable.
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

    /// Return stable rule provenance for source-visible rules.
    pub fn rule_provenance(&self) -> Vec<xlog_logic::RuleProvenance> {
        xlog_logic::rule_provenance(&self.program, None)
    }

    /// Return direct proof traces for source queries.
    pub fn proof_traces(&self) -> Vec<xlog_logic::QueryProofTrace> {
        let provenance = self.rule_provenance();
        xlog_logic::query_proof_traces(&self.program, &provenance)
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
    pub fn apply_relation_deltas(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
        let insert_rows = deltas
            .values()
            .filter_map(|d| d.insert.as_ref())
            .map(|b| b.num_rows())
            .sum();
        let delete_rows = deltas
            .values()
            .filter_map(|d| d.delete.as_ref())
            .map(|b| b.num_rows())
            .sum();
        let cache_reused = cached_store.is_some();
        let mut changed_relation_names = deltas.keys().cloned().collect::<Vec<_>>();
        changed_relation_names.sort();

        if cached_store.is_none() {
            let (_, store) = self.evaluate_with_relation_store_and_cache(
                provider.clone(),
                relation_store,
                false,
            )?;
            *cached_store = Some(store);
        }

        let store_before_delta = cached_store.as_ref().ok_or_else(|| {
            XlogError::Execution("Missing cached relation store for delta update".to_string())
        })?;
        let mut executor =
            self.executor_from_relation_store(provider.clone(), store_before_delta, false)?;
        let delta_stats = executor
            .apply_deltas_and_recompute(self.ordinary_plan("relation-delta recompute")?, &deltas)?;
        self.enforce_constraints(&provider, &executor)?;

        for name in deltas.keys() {
            let updated = executor.store().get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Delta relation {} missing after runtime recompute",
                    name
                ))
            })?;
            relation_store.put(name, provider.clone_buffer(updated)?);
        }

        *cached_store = Some(self.clone_relation_store(&provider, executor.store())?);

        let mut report = logic_delta_report(delta_stats, insert_rows, delete_rows);
        report.changed_relation_names = changed_relation_names;
        report.planner_telemetry =
            DeltaPlannerTelemetry::from_delta_report(&report, cache_reused, None);
        report.debug_trace = delta_debug_trace(&report);
        Ok(report)
    }

    /// Apply relation deltas while preserving retained session runtime state.
    pub fn apply_relation_deltas_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        deltas: HashMap<String, RelationDelta>,
    ) -> Result<LogicDeltaReport> {
        let insert_rows = deltas
            .values()
            .filter_map(|d| d.insert.as_ref())
            .map(|b| b.num_rows())
            .sum();
        let delete_rows = deltas
            .values()
            .filter_map(|d| d.delete.as_ref())
            .map(|b| b.num_rows())
            .sum();
        let cache_reused = session_runtime.is_some() || cached_store.is_some();
        let mut changed_relation_names = deltas.keys().cloned().collect::<Vec<_>>();
        changed_relation_names.sort();

        if session_runtime.is_none() {
            let seed_store: &RelationStore = match cached_store.as_ref() {
                Some(store) => store,
                None => &*relation_store,
            };
            *session_runtime =
                Some(self.create_session_runtime(provider.clone(), seed_store, false)?);
        }

        if cached_store.is_none() {
            let runtime = session_runtime.as_mut().ok_or_else(|| {
                XlogError::Execution("Missing session runtime for cached evaluation".to_string())
            })?;
            let (_, store) = self.evaluate_with_session_runtime(provider.clone(), runtime)?;
            *cached_store = Some(store);
        }

        let runtime = session_runtime.as_mut().ok_or_else(|| {
            XlogError::Execution("Missing session runtime for delta update".to_string())
        })?;
        let delta_stats = runtime.executor.apply_deltas_and_recompute(
            self.ordinary_plan("session relation-delta recompute")?,
            &deltas,
        )?;
        self.enforce_constraints(&provider, &runtime.executor)?;

        for name in deltas.keys() {
            let updated = runtime.executor.store().get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Delta relation {} missing after runtime recompute",
                    name
                ))
            })?;
            relation_store.put(name, provider.clone_buffer(updated)?);
        }

        *cached_store = Some(self.clone_relation_store(&provider, runtime.executor.store())?);

        let mut report = logic_delta_report(delta_stats, insert_rows, delete_rows);
        report.changed_relation_names = changed_relation_names;
        report.planner_telemetry =
            DeltaPlannerTelemetry::from_delta_report(&report, cache_reused, None);
        report.debug_trace = delta_debug_trace(&report);
        Ok(report)
    }

    /// Apply an ordered batch of relation deltas after device-side coalescing.
    pub fn apply_relation_delta_batch(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
        let coalesced = coalesce_relation_delta_batch(provider.as_ref(), delta_batch)?;
        if coalesced.deltas.is_empty() {
            return Ok(LogicDeltaReport {
                input_delta_count: coalesced.input_delta_count,
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
                canceled_rows: coalesced.canceled_rows,
                planner_telemetry: DeltaPlannerTelemetry {
                    fallback_decision: "no_op".to_string(),
                    ..DeltaPlannerTelemetry::default()
                },
                debug_trace: vec![format!("canceled_rows={}", coalesced.canceled_rows)],
            });
        }

        let mut report =
            self.apply_relation_deltas(provider, relation_store, cached_store, coalesced.deltas)?;
        report.input_delta_count = coalesced.input_delta_count;
        report.changed_relations = coalesced.changed_relations;
        report.coalesced_insert_rows = coalesced.coalesced_insert_rows;
        report.coalesced_delete_rows = coalesced.coalesced_delete_rows;
        report.canceled_rows = coalesced.canceled_rows;
        report.planner_telemetry = DeltaPlannerTelemetry::from_delta_report(&report, true, None);
        report.debug_trace = delta_debug_trace(&report);
        Ok(report)
    }

    /// Apply an ordered batch of relation deltas while preserving session runtime state.
    pub fn apply_relation_delta_batch_with_session_runtime(
        &self,
        provider: Arc<CudaKernelProvider>,
        relation_store: &mut RelationStore,
        cached_store: &mut Option<RelationStore>,
        session_runtime: &mut Option<LogicSessionRuntime>,
        delta_batch: Vec<(String, RelationDelta)>,
    ) -> Result<LogicDeltaReport> {
        let coalesced = coalesce_relation_delta_batch(provider.as_ref(), delta_batch)?;
        if coalesced.deltas.is_empty() {
            return Ok(LogicDeltaReport {
                input_delta_count: coalesced.input_delta_count,
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
                canceled_rows: coalesced.canceled_rows,
                planner_telemetry: DeltaPlannerTelemetry {
                    fallback_decision: "no_op".to_string(),
                    ..DeltaPlannerTelemetry::default()
                },
                debug_trace: vec![format!("canceled_rows={}", coalesced.canceled_rows)],
            });
        }

        let mut report = self.apply_relation_deltas_with_session_runtime(
            provider,
            relation_store,
            cached_store,
            session_runtime,
            coalesced.deltas,
        )?;
        report.input_delta_count = coalesced.input_delta_count;
        report.changed_relations = coalesced.changed_relations;
        report.coalesced_insert_rows = coalesced.coalesced_insert_rows;
        report.coalesced_delete_rows = coalesced.coalesced_delete_rows;
        report.canceled_rows = coalesced.canceled_rows;
        report.planner_telemetry = DeltaPlannerTelemetry::from_delta_report(&report, true, None);
        report.debug_trace = delta_debug_trace(&report);
        Ok(report)
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

        self.load_facts(&provider, &mut executor)?;

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
            let relation_name = format!("__xlog_query_{}", i);
            let buffer = executor.store_mut().remove(&relation_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing query result relation {} (compiler bug?)",
                    relation_name
                ))
            })?;

            let columns = query_output_vars(query);
            queries.push(LogicQueryResult {
                relation_name,
                sort_labels: columns.clone(),
                columns,
                buffer,
            });
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

            let columns = query_output_vars(query);
            queries.push(LogicQueryResult {
                relation_name,
                sort_labels: columns.clone(),
                columns,
                buffer: provider.clone_buffer(buffer)?,
            });
        }

        Ok(LogicEvalResult { queries, stats })
    }

    fn load_facts(&self, provider: &CudaKernelProvider, executor: &mut Executor) -> Result<()> {
        self.load_facts_into_store(provider, executor.store_mut())
    }

    fn load_facts_into_store(
        &self,
        provider: &CudaKernelProvider,
        store: &mut RelationStore,
    ) -> Result<()> {
        let arities = predicate_arities(&self.program);
        let mut rows_by_pred: HashMap<String, Vec<&[Term]>> = HashMap::new();
        for fact in self.program.facts() {
            let pred = fact.head.predicate.as_str();
            let arity = fact.head.terms.len();
            let key = arity_qualified_name_if_needed(pred, arity, &arities);
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
                    push_term_bytes(&mut columns[col_idx], term, typ)?;
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
        let upper_executor =
            self.run_wfs_gpu_pass(&provider, &wfs.overapprox, &base_store, &[], profiling)?;
        let mut upper_store = self.clone_relation_store(&provider, upper_executor.store())?;
        let mut lower_store = self.clone_relation_store(&provider, &base_store)?;

        for _ in 0..wfs.max_iterations {
            let upper_fixed: Vec<_> = wfs
                .upper_fixed_names
                .iter()
                .map(|(source, fixed)| (source.as_str(), fixed.as_str(), &upper_store))
                .collect();
            let lower_executor =
                self.run_wfs_gpu_pass(&provider, &wfs.lower, &base_store, &upper_fixed, profiling)?;
            let next_lower = self.clone_relation_store(&provider, lower_executor.store())?;

            let lower_fixed: Vec<_> = wfs
                .lower_fixed_names
                .iter()
                .map(|(source, fixed)| (source.as_str(), fixed.as_str(), &next_lower))
                .collect();
            let next_upper_executor =
                self.run_wfs_gpu_pass(&provider, &wfs.upper, &base_store, &lower_fixed, profiling)?;
            let next_upper = self.clone_relation_store(&provider, next_upper_executor.store())?;

            let lower_converged =
                self.wfs_gpu_stores_equivalent(&provider, wfs, &lower_store, &next_lower)?;
            let upper_converged =
                self.wfs_gpu_stores_equivalent(&provider, wfs, &upper_store, &next_upper)?;
            lower_store = next_lower;
            upper_store = next_upper;
            if lower_converged && upper_converged {
                return self.logic_result_from_store(provider.as_ref(), &lower_store, None);
            }
        }

        Err(XlogError::ResourceExhausted {
            context: "GPU-backed WFS alternating fixpoint iterations".to_string(),
            estimated_bytes: wfs.max_iterations as u64,
            budget_bytes: wfs.max_iterations as u64,
        })
    }

    fn run_wfs_gpu_pass(
        &self,
        provider: &Arc<CudaKernelProvider>,
        pass: &WfsGpuOrdinaryPlan,
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
                self.wfs_gpu_clone_or_empty(provider, &pass.schemas, source, source_store)?;
            executor.store_mut().put(fixed, buffer);
        }
        executor.execute_plan(&pass.plan)?;
        Ok(executor)
    }

    fn wfs_gpu_clone_or_empty(
        &self,
        provider: &Arc<CudaKernelProvider>,
        schemas: &HashMap<String, Schema>,
        name: &str,
        store: &RelationStore,
    ) -> Result<CudaBuffer> {
        if let Some(buffer) = store.get(name) {
            return provider.clone_buffer(buffer);
        }
        let schema = schemas
            .get(name)
            .or_else(|| self.schemas.get(name))
            .ok_or_else(|| XlogError::Execution(format!("missing WFS GPU schema for {name}")))?;
        provider.create_empty_buffer(schema.clone())
    }

    fn wfs_gpu_stores_equivalent(
        &self,
        provider: &Arc<CudaKernelProvider>,
        wfs: &EpistemicWfsGpuPlan,
        left: &RelationStore,
        right: &RelationStore,
    ) -> Result<bool> {
        for pred in &wfs.intensional_predicates {
            let left_buf = self.wfs_gpu_clone_or_empty(provider, &wfs.lower.schemas, pred, left)?;
            let right_buf =
                self.wfs_gpu_clone_or_empty(provider, &wfs.lower.schemas, pred, right)?;
            if !buffers_gpu_set_equivalent(provider.as_ref(), &left_buf, &right_buf)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn ordinary_plan(&self, context: &str) -> Result<&ExecutionPlan> {
        match &self.plan {
            LogicExecutionPlan::Ordinary(plan) => Ok(plan),
            LogicExecutionPlan::EpistemicWfsGpu(_)
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
                let result = executor.execute_epistemic_gpu_execution(
                    executable,
                    capacities_for_epistemic_executable(executable)?,
                )?;
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
            LogicExecutionPlan::EpistemicWfsGpu(_) => {
                unreachable!("GPU WFS plans are handled earlier")
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
        for i in 0..self.program.constraints.len() {
            let name = format!("__xlog_constraint_{}", i);
            let buf = executor.store().get(&name).ok_or_else(|| {
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

            return Err(XlogError::Execution(format!(
                "Constraint {} violated: {}",
                i,
                format_constraint(&self.program.constraints[i].body)
            )));
        }

        Ok(())
    }
}

const DEFAULT_EPISTEMIC_MAX_MODELS_PER_REDUCTION: usize = 1024;

fn normalize_program(program: Program) -> Result<Program> {
    let max_recursion = program.directives.max_recursion_depth.unwrap_or(100);
    let expanded = xlog_logic::expand_program_functions(&program, max_recursion)
        .map_err(|e| XlogError::Compilation(e.to_string()))?;
    let normalized = xlog_logic::normalize_meta_builtins(&expanded)?;
    let listed = xlog_logic::normalize_list_builtins(&normalized)?;
    Ok(desugar_shared_variable_epistemic_constraints(listed))
}

enum WfsNegationTransform<'a> {
    Drop,
    Rename(&'a HashMap<String, String>),
}

fn compile_epistemic_wfs_gpu_plan(program: &Program) -> Result<EpistemicWfsGpuPlan> {
    if !program.constraints.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "GPU WFS integrity constraints".to_string(),
            context: "cyclic WFS execution currently supports reduced normal rules only"
                .to_string(),
        });
    }

    let negated = wfs_negated_predicates(program);
    let upper_fixed_names = wfs_fixed_names(program, &negated, "__wfs_upper");
    let lower_fixed_names = wfs_fixed_names(program, &negated, "__wfs_lower");

    let overapprox_program = wfs_transform_program(program, WfsNegationTransform::Drop)?;
    let lower_program =
        wfs_transform_program(program, WfsNegationTransform::Rename(&upper_fixed_names))?;
    let upper_program =
        wfs_transform_program(program, WfsNegationTransform::Rename(&lower_fixed_names))?;

    Ok(EpistemicWfsGpuPlan {
        overapprox: compile_wfs_gpu_ordinary_plan(&overapprox_program)?,
        lower: compile_wfs_gpu_ordinary_plan(&lower_program)?,
        upper: compile_wfs_gpu_ordinary_plan(&upper_program)?,
        intensional_predicates: wfs_intensional_predicates(program),
        upper_fixed_names,
        lower_fixed_names,
        max_iterations: (program.directives.max_recursion_depth_or_default() as usize).max(1),
    })
}

fn compile_wfs_gpu_ordinary_plan(program: &Program) -> Result<WfsGpuOrdinaryPlan> {
    let mut compiler = Compiler::new();
    let plan = compiler.compile_program(program)?;
    Ok(WfsGpuOrdinaryPlan {
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
            let mut body = Vec::with_capacity(rule.body.len());
            for lit in &rule.body {
                match (lit, &negation) {
                    (BodyLiteral::Negated(_), WfsNegationTransform::Drop) => {}
                    (BodyLiteral::Negated(atom), WfsNegationTransform::Rename(names)) => {
                        let mut atom = atom.clone();
                        atom.predicate = names.get(&atom.predicate).cloned().ok_or_else(|| {
                            XlogError::Execution(format!(
                                "missing WFS fixed relation name for {}",
                                atom.predicate
                            ))
                        })?;
                        body.push(BodyLiteral::Negated(atom));
                    }
                    _ => body.push(lit.clone()),
                }
            }
            rule.body = body;
            Ok(rule)
        })
        .collect::<Result<Vec<_>>>()?;
    if let WfsNegationTransform::Rename(names) = negation {
        add_wfs_fixed_predicates(&mut out, names)?;
    }
    Ok(out)
}

fn add_wfs_fixed_predicates(program: &mut Program, names: &HashMap<String, String>) -> Result<()> {
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
        let Some(decl) = program.predicates.iter().find(|decl| decl.name == *source) else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "GPU WFS fixed relation schema".to_string(),
                context: format!(
                    "negated predicate {source} has no declaration to type fixed relation {fixed}"
                ),
            });
        };
        let mut fixed_decl = decl.clone();
        fixed_decl.name = fixed.clone();
        program.predicates.push(fixed_decl);
    }
    Ok(())
}

fn wfs_negated_predicates(program: &Program) -> BTreeSet<String> {
    program
        .rules
        .iter()
        .flat_map(|rule| &rule.body)
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

fn wfs_plan_combined_rel_ids(plan: &EpistemicWfsGpuPlan) -> Result<HashMap<String, RelId>> {
    let mut rel_ids = HashMap::new();
    for ordinary in [&plan.overapprox, &plan.lower, &plan.upper] {
        for (name, rel_id) in &ordinary.rel_ids {
            rel_ids.insert(name.clone(), *rel_id);
        }
    }
    Ok(rel_ids)
}

fn schema_from_pred_decl(
    decl: &xlog_logic::ast::PredDecl,
    domains: &HashMap<String, ScalarType>,
) -> Result<Schema> {
    let columns = pred_columns_for_decl(decl);
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

fn pred_columns_for_decl(decl: &xlog_logic::ast::PredDecl) -> Vec<PredColumn> {
    if decl.columns.is_empty() {
        decl.types
            .iter()
            .cloned()
            .map(|typ| PredColumn { name: None, typ })
            .collect()
    } else {
        decl.columns.clone()
    }
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
        .map(|(idx, term)| (format!("c{idx}"), infer_term_type(term)))
        .collect();
    Schema::new(columns)
}

fn infer_term_type(term: &Term) -> ScalarType {
    match term {
        Term::Variable(_) | Term::Anonymous => ScalarType::U64,
        Term::Integer(value) => {
            if *value >= 0 && *value <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Term::Float(_) => ScalarType::F64,
        Term::String(_) | Term::Symbol(_) => ScalarType::Symbol,
        Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
            ScalarType::U64
        }
        Term::Aggregate(agg) => match agg.op {
            AggOp::Count => ScalarType::U32,
            AggOp::Sum => ScalarType::U64,
            AggOp::Min | AggOp::Max => ScalarType::U32,
            AggOp::LogSumExp => ScalarType::F64,
        },
    }
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
    let arities = predicate_arities(program);
    let domains: HashMap<String, ScalarType> = program
        .domains
        .iter()
        .map(|domain| (domain.name.clone(), domain.typ))
        .collect();

    for decl in &program.predicates {
        let Some(pred_arities) = arities.get(&decl.name) else {
            continue;
        };
        if pred_arities.len() <= 1 {
            continue;
        }
        let key = arity_qualified_name(&decl.name, pred_decl_arity(decl));
        schemas.insert(key, schema_from_pred_decl(decl, &domains)?);
    }

    for fact in program.facts() {
        let pred = fact.head.predicate.as_str();
        let arity = fact.head.terms.len();
        let Some(pred_arities) = arities.get(pred) else {
            continue;
        };
        if pred_arities.len() <= 1 {
            continue;
        }
        let key = arity_qualified_name(pred, arity);
        schemas
            .entry(key)
            .or_insert_with(|| schema_from_terms(&fact.head.terms));
    }

    for rule in &program.rules {
        augment_atom_schema_if_needed(&rule.head, &arities, schemas);
        for literal in &rule.body {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    augment_atom_schema_if_needed(atom, &arities, schemas);
                }
                BodyLiteral::Epistemic(epistemic) => {
                    augment_atom_schema_if_needed(&epistemic.atom, &arities, schemas);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }
    }

    for query in &program.queries {
        augment_atom_schema_if_needed(&query.atom, &arities, schemas);
    }

    Ok(())
}

fn augment_atom_schema_if_needed(
    atom: &Atom,
    arities: &HashMap<String, BTreeSet<usize>>,
    schemas: &mut HashMap<String, Schema>,
) {
    let Some(pred_arities) = arities.get(&atom.predicate) else {
        return;
    };
    if pred_arities.len() <= 1 {
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
        add_predicate_arity(&mut arities, &decl.name, pred_decl_arity(decl));
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

fn arity_qualified_name_if_needed(
    predicate: &str,
    arity: usize,
    arities: &HashMap<String, BTreeSet<usize>>,
) -> String {
    if arities.get(predicate).is_some_and(|items| items.len() > 1) {
        arity_qualified_name(predicate, arity)
    } else {
        predicate.to_string()
    }
}

fn arity_qualified_name(predicate: &str, arity: usize) -> String {
    format!("{predicate}/{arity}")
}

fn pred_decl_arity(decl: &xlog_logic::ast::PredDecl) -> usize {
    if decl.columns.is_empty() {
        decl.types.len()
    } else {
        decl.columns.len()
    }
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

fn coalesce_relation_delta_batch(
    provider: &CudaKernelProvider,
    delta_batch: Vec<(String, RelationDelta)>,
) -> Result<CoalescedRelationDeltaBatch> {
    let input_delta_count = delta_batch.len();
    let mut pending_by_relation: HashMap<String, PendingRelationDelta> = HashMap::new();
    let mut canceled_rows = 0u64;

    for (name, delta) in delta_batch {
        let pending = pending_by_relation.entry(name).or_default();
        if let Some(insert) = delta.insert {
            merge_insert_delta(provider, pending, insert, &mut canceled_rows)?;
        }
        if let Some(delete) = delta.delete {
            merge_delete_delta(provider, pending, delete, &mut canceled_rows)?;
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
    Ok(CoalescedRelationDeltaBatch {
        deltas,
        input_delta_count,
        changed_relations,
        coalesced_insert_rows,
        coalesced_delete_rows,
        canceled_rows,
    })
}

fn merge_insert_delta(
    provider: &CudaKernelProvider,
    pending: &mut PendingRelationDelta,
    insert: CudaBuffer,
    canceled_rows: &mut u64,
) -> Result<()> {
    let mut incoming = provider.dedup_full_row(&insert)?;
    if let Some(delete) = pending.delete.take().and_then(non_empty_buffer) {
        let delete_before = buffer_rows(&delete);
        let delete_after = provider.diff_full_row(&delete, &incoming)?;
        let insert_after = provider.diff_full_row(&incoming, &delete)?;
        *canceled_rows += delete_before.saturating_sub(buffer_rows(&delete_after));
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
) -> Result<()> {
    let mut incoming = provider.dedup_full_row(&delete)?;
    if let Some(insert) = pending.insert.take().and_then(non_empty_buffer) {
        let insert_before = buffer_rows(&insert);
        let insert_after = provider.diff_full_row(&insert, &incoming)?;
        let delete_after = provider.diff_full_row(&incoming, &insert)?;
        *canceled_rows += insert_before.saturating_sub(buffer_rows(&insert_after));
        pending.insert = non_empty_buffer(insert_after);
        incoming = delete_after;
    }
    pending.delete = merge_optional_buffer(provider, pending.delete.take(), incoming)?;
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
    if buffer.is_empty() {
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

fn push_term_bytes(out: &mut Vec<u8>, term: &Term, typ: xlog_core::ScalarType) -> Result<()> {
    use xlog_core::symbol;
    use xlog_core::ScalarType;

    match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => {
            let v = u32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::U64, Term::Integer(v)) => {
            let v = u64::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u64 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::I32, Term::Integer(v)) => {
            let v = i32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("i32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::I64, Term::Integer(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Float(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Float(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f64).to_le_bytes());
        }
        (ScalarType::Bool, Term::Integer(v)) => {
            let b = match *v {
                0 => 0u8,
                1 => 1u8,
                other => {
                    return Err(XlogError::Execution(format!(
                        "bool expects 0/1, got {}",
                        other
                    )));
                }
            };
            out.push(b);
        }
        (ScalarType::Bool, Term::Symbol(id)) => {
            let s = symbol::resolve(*id);
            if s == "true" || s == "false" {
                out.push(if s == "true" { 1u8 } else { 0u8 });
            } else {
                return Err(XlogError::Execution(format!(
                    "Expected boolean symbol 'true' or 'false', got '{}'",
                    s
                )));
            }
        }
        (ScalarType::Symbol, Term::String(s)) => {
            out.extend_from_slice(&symbol::intern(s).to_le_bytes());
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
            // Symbol is already interned, just use the ID directly
            out.extend_from_slice(&id.to_le_bytes());
        }
        (_, Term::Variable(v)) => {
            return Err(XlogError::Execution(format!(
                "Fact cannot contain variable {}",
                v
            )));
        }
        (_, Term::Anonymous) => {
            return Err(XlogError::Execution(
                "Fact cannot contain anonymous wildcard '_'".to_string(),
            ));
        }
        (_, Term::Aggregate(_)) => {
            return Err(XlogError::Execution(
                "Fact cannot contain aggregate".to_string(),
            ));
        }
        (expected, got) => {
            return Err(XlogError::Execution(format!(
                "Type mismatch in fact: expected {:?}, got {:?}",
                expected, got
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

fn format_term(term: &Term) -> String {
    match term {
        Term::Variable(v) => v.clone(),
        Term::Anonymous => "_".to_string(),
        Term::Integer(i) => i.to_string(),
        Term::Float(f) => f.to_string(),
        Term::String(s) => format!("{:?}", s),
        Term::Symbol(id) => symbol::resolve(*id),
        Term::List(items) => format!(
            "[{}]",
            items.iter().map(format_term).collect::<Vec<_>>().join(", ")
        ),
        Term::Cons { head, tail } => format!("[{} | {}]", format_term(head), format_term(tail)),
        Term::Compound { functor, args } => format!(
            "{}({})",
            functor,
            args.iter().map(format_term).collect::<Vec<_>>().join(", ")
        ),
        Term::PredRef(name) => format!("predref({})", name),
        Term::Aggregate(a) => format!("{:?}({})", a.op, a.variable),
    }
}

fn format_constraint(body: &[BodyLiteral]) -> String {
    let lits = body
        .iter()
        .map(|lit| match lit {
            BodyLiteral::Positive(a) => {
                let args = a
                    .terms
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", a.predicate, args)
            }
            BodyLiteral::Negated(a) => {
                let args = a
                    .terms
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("not {}({})", a.predicate, args)
            }
            BodyLiteral::Epistemic(lit) => {
                let args = lit
                    .atom
                    .terms
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                let op = match lit.op {
                    xlog_logic::EpistemicOp::Know => "know",
                    xlog_logic::EpistemicOp::Possible => "possible",
                };
                let prefix = if lit.negated { "not " } else { "" };
                format!("{prefix}{op} {}({})", lit.atom.predicate, args)
            }
            BodyLiteral::Comparison(c) => format!("{:?} {:?} {:?}", c.left, c.op, c.right),
            BodyLiteral::IsExpr(is) => format!("{} is {:?}", is.target, is.expr),
            BodyLiteral::Univ(univ) => {
                format!(
                    "{} =.. {}",
                    format_term(&univ.term),
                    format_term(&univ.parts)
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(":- {}.", lits)
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
mod v086_delta_coalesce_tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType};
    use xlog_cuda::{CudaDevice, GpuMemoryManager};

    fn test_provider() -> Option<Arc<CudaKernelProvider>> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
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
                "external_consumer_commit".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[7, 8])), None),
            ),
            (
                "external_consumer_commit".to_string(),
                RelationDelta::new(None, Some(test_buffer(&provider, &[8]))),
            ),
            (
                "external_consumer_commit".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[9])), None),
            ),
        ];

        let report = coalesce_relation_delta_batch(provider.as_ref(), batch)
            .expect("coalesce relation delta batch");
        let delta = report
            .deltas
            .get("external_consumer_commit")
            .expect("coalesced relation");
        let insert = delta.insert.as_ref().expect("coalesced insert");
        assert_eq!(read_u32(&provider, insert), vec![7, 9]);
        assert!(delta.delete.as_ref().map(|b| b.is_empty()).unwrap_or(true));
        assert_eq!(report.input_delta_count, 3);
        assert_eq!(report.changed_relations, 1);
        assert_eq!(report.coalesced_insert_rows, 2);
        assert_eq!(report.coalesced_delete_rows, 0);
        assert_eq!(report.canceled_rows, 1);
    }

    #[test]
    fn relation_delta_batch_updates_runtime_store_and_reports_coalesced_counts() -> Result<()> {
        let Some(provider) = test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return Ok(());
        };

        let source = r#"
            pred external_consumer_commit(u32).
            pred out(u32).

            out(X) :- external_consumer_commit(X).

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
                    "external_consumer_commit".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1, 2, 3])), None),
                ),
                (
                    "external_consumer_commit".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
                ),
                (
                    "external_consumer_commit".to_string(),
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
                HashMap::from([("external_consumer_commit".to_string(), delta)]),
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
        replacement_store.put("external_consumer_commit", test_buffer(&provider, &[1, 3, 4]));
        let replacement =
            program.evaluate_with_relation_store(provider.clone(), &replacement_store, false)?;
        let replacement_rows = sorted_query_rows(&provider, &replacement);

        assert_eq!(coalesced_rows, vec![1, 3, 4]);
        assert_eq!(coalesced_rows, sequential_rows);
        assert_eq!(coalesced_rows, replacement_rows);
        Ok(())
    }
}
