//! GPU-accelerated evaluation of compiled Datalog programs.

use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{symbol, Result, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::{BodyLiteral, Compiler, Program, Query, Term};
use xlog_runtime::{DeltaRecomputeStats, ExecutionStats, Executor, RelationDelta, RelationStore};

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

/// Summary for a persistent-session relation delta update.
pub struct LogicDeltaReport {
    /// Number of changed relation names in the delta batch.
    pub changed_relations: usize,
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
}

/// A compiled Datalog program ready for GPU evaluation.
#[derive(Clone)]
pub struct LogicProgram {
    program: Program,
    plan: xlog_ir::ExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, xlog_core::RelId>,
}

impl LogicProgram {
    /// Compile a Datalog source string into a GPU-executable program.
    pub fn compile(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;

        // Expand user-defined function calls before compilation
        let max_recursion = program.directives.max_recursion_depth.unwrap_or(100);
        let expanded = xlog_logic::expand_program_functions(&program, max_recursion)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;
        let normalized = xlog_logic::normalize_v085_lists(&expanded)?;

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&normalized)?;
        Ok(Self {
            program: normalized,
            plan,
            schemas: compiler.schemas().clone(),
            rel_ids: compiler.rel_ids().clone(),
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

        // Expand user-defined function calls before compilation
        let max_recursion = merged.directives.max_recursion_depth.unwrap_or(100);
        let expanded = xlog_logic::expand_program_functions(&merged, max_recursion)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;
        let normalized = xlog_logic::normalize_v085_lists(&expanded)?;

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&normalized)?;
        Ok(Self {
            program: normalized,
            plan,
            schemas: compiler.schemas().clone(),
            rel_ids: compiler.rel_ids().clone(),
        })
    }

    /// Look up the schema for a named relation.
    pub fn schema(&self, relation: &str) -> Option<&Schema> {
        self.schemas.get(relation)
    }

    /// Return the full schema map (relation name to schema).
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
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
        executor.execute_plan(&self.plan)?;
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
        let delta_stats = executor.apply_deltas_and_recompute(&self.plan, &deltas)?;
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

        Ok(logic_delta_report(delta_stats, insert_rows, delete_rows))
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

        executor.execute_plan(&self.plan)?;

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

            let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
            let fact_buf = provider.create_buffer_from_slices(&slices, schema.clone())?;

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
        changed_relations: stats.changed_relations,
        insert_rows,
        delete_rows,
        has_deletes: stats.has_deletes,
        affected_sccs: stats.affected_sccs,
        recomputed_sccs: stats.recomputed_sccs,
        incremental_sccs: stats.incremental_sccs,
    }
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
            BodyLiteral::Comparison(c) => format!("{:?} {:?} {:?}", c.left, c.op, c.right),
            BodyLiteral::IsExpr(is) => format!("{} is {:?}", is.target, is.expr),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(":- {}.", lits)
}
