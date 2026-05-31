//! GPU-accelerated evaluation of compiled Datalog programs.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use xlog_core::{symbol, RelId, Result, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{EpistemicExecutablePlan, ExecutionPlan};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_split_execution,
    reduce_epistemic_program_to_ordinary, try_reduce_case_a_recursive_epistemic_program,
    EpistemicSplitExecutablePlan,
};
use xlog_logic::{BodyLiteral, Compiler, Program, Query, Term};
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

#[derive(Clone)]
enum LogicExecutionPlan {
    Ordinary(ExecutionPlan),
    EpistemicSingle(EpistemicExecutablePlan),
    EpistemicSplit(EpistemicSplitExecutablePlan),
}

/// A compiled Datalog program ready for GPU evaluation.
#[derive(Clone)]
pub struct LogicProgram {
    program: Program,
    plan: LogicExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
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
        })
    }

    fn compile_epistemic_program(normalized: Program) -> Result<Self> {
        // Case A: ordinary recursion gated by modal literals over invariant relations.
        // Resolve each modal literal to its (invariant) gated relation and route the
        // resulting ordinary recursive program through the EXISTING recursive/
        // semi-naive engine via an Ordinary plan. Validation still flows through the
        // EIR boundary + FAEEL foundedness guard inside
        // `try_reduce_case_a_recursive_epistemic_program`, so modal self-support
        // (Case B) and every non-Case-A recursive shape still fail closed.
        if let Some(case_a_reduced) = try_reduce_case_a_recursive_epistemic_program(&normalized)? {
            let mut compiler = Compiler::new();
            let plan = compiler.compile_program(&case_a_reduced)?;
            return Ok(Self {
                program: case_a_reduced,
                plan: LogicExecutionPlan::Ordinary(plan),
                schemas: compiler.schemas().clone(),
                rel_ids: compiler.rel_ids().clone(),
            });
        }

        let reduced = reduce_epistemic_program_to_ordinary(&normalized);
        let mut schema_compiler = Compiler::new();
        schema_compiler.compile_program(&reduced)?;
        let schemas = schema_compiler.schemas().clone();

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
        })
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
        let mut rows_by_pred: HashMap<&str, Vec<&[Term]>> = HashMap::new();
        for fact in self.program.facts() {
            rows_by_pred
                .entry(fact.head.predicate.as_str())
                .or_default()
                .push(&fact.head.terms);
        }

        for (pred, rows) in rows_by_pred {
            let schema = self.schemas.get(pred).ok_or_else(|| {
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

            let existing = store.get(pred).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing base relation {} while loading facts",
                    pred
                ))
            })?;

            let merged = provider.union(existing, &fact_buf)?;
            store.put(pred, merged);
        }

        Ok(())
    }

    fn ordinary_plan(&self, context: &str) -> Result<&ExecutionPlan> {
        match &self.plan {
            LogicExecutionPlan::Ordinary(plan) => Ok(plan),
            LogicExecutionPlan::EpistemicSingle(_) | LogicExecutionPlan::EpistemicSplit(_) => {
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
                queries.push(epistemic_result_to_query_result(
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
                    queries.push(epistemic_result_to_query_result(
                        epistemic_output_relation_name(&component.executable)?,
                        result,
                    ));
                }
            }
            LogicExecutionPlan::Ordinary(_) => unreachable!("ordinary plans are handled earlier"),
        }

        let total_output_rows: u64 = queries.iter().map(|q| q.buffer.num_rows()).sum();
        let stats = if profiling {
            Some(executor.execution_stats(total_output_rows))
        } else {
            None
        };
        Ok(LogicEvalResult { queries, stats })
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
    let normalized = xlog_logic::normalize_v085_meta(&expanded)?;
    xlog_logic::normalize_v085_lists(&normalized)
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

fn epistemic_result_to_query_result(
    relation_name: String,
    result: EpistemicGpuExecutionResult,
) -> LogicQueryResult {
    let schema = result.final_output.schema();
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
        buffer: result.final_output,
    }
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
                "wmir_committed".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[7, 8])), None),
            ),
            (
                "wmir_committed".to_string(),
                RelationDelta::new(None, Some(test_buffer(&provider, &[8]))),
            ),
            (
                "wmir_committed".to_string(),
                RelationDelta::new(Some(test_buffer(&provider, &[9])), None),
            ),
        ];

        let report = coalesce_relation_delta_batch(provider.as_ref(), batch)
            .expect("coalesce relation delta batch");
        let delta = report
            .deltas
            .get("wmir_committed")
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
            pred wmir_committed(u32).
            pred out(u32).

            out(X) :- wmir_committed(X).

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
                    "wmir_committed".to_string(),
                    RelationDelta::new(Some(test_buffer(&provider, &[1, 2, 3])), None),
                ),
                (
                    "wmir_committed".to_string(),
                    RelationDelta::new(None, Some(test_buffer(&provider, &[2]))),
                ),
                (
                    "wmir_committed".to_string(),
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
                HashMap::from([("wmir_committed".to_string(), delta)]),
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
        replacement_store.put("wmir_committed", test_buffer(&provider, &[1, 3, 4]));
        let replacement =
            program.evaluate_with_relation_store(provider.clone(), &replacement_store, false)?;
        let replacement_rows = sorted_query_rows(&provider, &replacement);

        assert_eq!(coalesced_rows, vec![1, 3, 4]);
        assert_eq!(coalesced_rows, sequential_rows);
        assert_eq!(coalesced_rows, replacement_rows);
        Ok(())
    }
}
