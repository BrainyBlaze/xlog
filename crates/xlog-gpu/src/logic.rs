use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{symbol, Result, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::{BodyLiteral, Compiler, Program, Query, Term};
use xlog_runtime::{ExecutionStats, Executor};

pub struct LogicQueryResult {
    pub relation_name: String,
    pub columns: Vec<String>,
    pub buffer: CudaBuffer,
}

pub struct LogicEvalResult {
    pub queries: Vec<LogicQueryResult>,
    /// Execution statistics (populated when profiling is enabled)
    pub stats: Option<ExecutionStats>,
}

pub struct LogicProgram {
    program: Program,
    plan: xlog_ir::ExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, xlog_core::RelId>,
}

impl LogicProgram {
    pub fn compile(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;

        // Expand user-defined function calls before compilation
        let max_recursion = program.directives.max_recursion_depth.unwrap_or(100);
        let expanded = xlog_logic::expand_program_functions(&program, max_recursion)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&expanded)?;
        Ok(Self {
            program: expanded,
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

        let mut compiler = Compiler::new();
        let plan = compiler.compile_program(&expanded)?;
        Ok(Self {
            program: expanded,
            plan,
            schemas: compiler.schemas().clone(),
            rel_ids: compiler.rel_ids().clone(),
        })
    }

    pub fn schema(&self, relation: &str) -> Option<&Schema> {
        self.schemas.get(relation)
    }

    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

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

            queries.push(LogicQueryResult {
                relation_name,
                columns: query_output_vars(query),
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

    fn load_facts(&self, provider: &CudaKernelProvider, executor: &mut Executor) -> Result<()> {
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

            let existing = executor.store().get(pred).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Missing base relation {} while loading facts",
                    pred
                ))
            })?;

            let merged = provider.union(existing, &fact_buf)?;
            executor.store_mut().put(pred, merged);
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
        if let Term::Variable(name) = term {
            if seen.insert(name.as_str()) {
                out.push(name.clone());
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
