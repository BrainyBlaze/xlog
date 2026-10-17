//! Compilation pipeline for XLOG programs
//!
//! This module compiles the core XLOG AST into execution plans. The compilation
//! process consists of:
//!
//! 1. **Parsing**: Convert source text to AST (`parser::parse_program`)
//! 2. **Stratification**: Analyze negation/aggregation dependencies (`stratify::stratify`)
//! 3. **Lowering**: Transform AST to Relational IR (`lower::Lowerer::lower_program`)
//!
//! The `Compiler` struct orchestrates these phases and provides a single entry
//! point via the `compile` method. User-defined functions are a source-level
//! normalization step: callers using this low-level compiler must expand them
//! before compilation. Execution-facing `LogicProgram` APIs perform that
//! normalization for parsed and source inputs.

use std::path::{Path, PathBuf};

use xlog_core::{Result, XlogError};
use xlog_ir::{ExecutionPlan, GeneratedQueryRuleProvenance};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::compiler_config::CompilerConfig;
use crate::list_normalize::normalize_list_builtins_owned;
use crate::lower::Lowerer;
use crate::magic_sets::rewrite_magic_sets_owned;
use crate::meta_normalize::normalize_meta_builtins_owned;
use crate::module::ModuleError;
use crate::optimizer::Optimizer;
use crate::parser::parse_program;
use crate::resolver::ModuleResolver;
use crate::stratify::stratify;
use crate::{BodyLiteral, Program, Query, Rule as AstRule, Term};

/// The XLOG compiler orchestrates the full compilation pipeline.
///
/// This is the core AST-to-RIR compiler. It does not resolve imports or expand
/// user-defined functions; callers must perform those source-level steps first,
/// or use the execution-facing `LogicProgram` compilation APIs.
///
/// # Example
///
/// ```ignore
/// use xlog_logic::compile::Compiler;
///
/// let mut compiler = Compiler::new();
/// let plan = compiler.compile(r#"
///     edge(1, 2).
///     edge(2, 3).
///     reach(X, Y) :- edge(X, Y).
///     reach(X, Z) :- reach(X, Y), edge(Y, Z).
/// "#)?;
/// ```
pub struct Compiler {
    lowerer: Lowerer,
}

pub(crate) struct CompilationOutput<T, O = Arc<dyn Send + Sync>> {
    pub(crate) value: T,
    pub(crate) allocation_owner: O,
}

enum CompilationValue<T, O = Arc<dyn Send + Sync>> {
    Unowned(T),
    Owned(CompilationOutput<T, O>),
}

type CompilationProgram =
    CompilationValue<Program, crate::program_custody::ProgramAllocationCustody>;

impl<T, O> std::ops::Deref for CompilationValue<T, O> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Unowned(value) => value,
            Self::Owned(output) => &output.value,
        }
    }
}

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use xlog_core::{RelId, Schema};

pub use xlog_core::{
    btree_insert_allocation_bytes, hash_allocation_bytes, push_vector_allocation_bytes,
    CompilationAllocationScope,
};

pub fn formatted_text_allocation_bytes(arguments: std::fmt::Arguments<'_>) -> Result<u64> {
    xlog_core::error::formatted_text_allocation_bytes(arguments)
}

/// Capacity of the original compiler's positive scalar subset, before owned
/// normalization, graph, schema, and physical-plan storage is allocated.
/// The caller must first use its canonical positive-rule and ground-query
/// validators; this helper is capacity metadata, not a source admission.
pub fn positive_task_compiler_allocation_bound(program: &Program) -> Result<u64> {
    use crate::ast::{Atom, PredColumn, PredDecl, TypeRef};
    use crate::stratify::{DepEdge, DependencyGraph, Stratum};
    use std::mem::size_of;
    let overflow = || XlogError::allocation_extent_overflow("positive task compiler");
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let vector = |count: usize, width: usize| mul(add(mul(count, 2)?, 4)?, width);
    let table = |count: usize, width: usize| -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        let buckets = add(mul(count, 8)?, 6)?
            .checked_div(7)
            .and_then(|n| n.max(8).checked_next_power_of_two())
            .ok_or_else(overflow)?;
        add(mul(buckets, add(width, 1)?)?, 16)
    };
    let mut ast = add(
        size_of::<Program>(),
        add(
            vector(program.predicates.len(), size_of::<PredDecl>())?,
            add(
                vector(program.rules.len(), size_of::<AstRule>())?,
                vector(program.queries.len(), size_of::<Query>())?,
            )?,
        )?,
    )?;
    let mut max_name = "__xlog_query_".len() + 20;
    // Generated ground-query heads have one constant column. Original
    // relations keep their declared one- or two-coordinate schema.
    let mut max_arity = usize::from(!program.queries.is_empty());
    let mut names = 0usize;
    for declaration in &program.predicates {
        max_arity = max_arity.max(declaration.types.len());
        ast = add(ast, size_of::<PredDecl>())?;
        ast = add(ast, declaration.name.len())?;
        names = add(names, declaration.name.len())?;
        max_name = max_name.max(declaration.name.len());
        ast = add(ast, vector(declaration.types.len(), size_of::<TypeRef>())?)?;
        {
            ast = add(
                ast,
                vector(declaration.columns.len(), size_of::<PredColumn>())?,
            )?;
            for column in &declaration.columns {
                if let Some(name) = &column.name {
                    ast = add(ast, name.len())?;
                    max_name = max_name.max(name.len());
                }
            }
        }
    }
    let atom_bytes = |atom: &Atom| -> Result<usize> {
        let mut bytes = add(size_of::<Atom>(), atom.predicate.len())?;
        bytes = add(bytes, vector(atom.terms.len(), size_of::<Term>())?)?;
        for term in &atom.terms {
            match term {
                Term::Variable(name) => bytes = add(bytes, name.len())?,
                Term::Integer(_) | Term::Anonymous => {}
                _ => {
                    return Err(XlogError::Compilation(
                        "positive task compiler capacity requires the original scalar subset"
                            .into(),
                    ));
                }
            }
        }
        Ok(bytes)
    };
    for rule in &program.rules {
        ast = add(ast, add(size_of::<AstRule>(), atom_bytes(&rule.head)?)?)?;
        names = add(names, rule.head.predicate.len())?;
        max_name = max_name.max(rule.head.predicate.len());
        if rule.body.len() > 3 {
            return Err(XlogError::Compilation(
                "positive task compiler capacity requires at most three original body atoms".into(),
            ));
        }
        ast = add(ast, vector(rule.body.len(), size_of::<BodyLiteral>())?)?;
        for literal in &rule.body {
            let BodyLiteral::Positive(atom) = literal else {
                return Err(XlogError::Compilation(
                    "positive task compiler capacity requires positive original body atoms".into(),
                ));
            };
            ast = add(ast, atom_bytes(atom)?)?;
            names = add(names, atom.predicate.len())?;
            max_name = max_name.max(atom.predicate.len());
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    max_name = max_name.max(name.len());
                }
            }
        }
    }
    for query in &program.queries {
        ast = add(ast, add(size_of::<Query>(), atom_bytes(&query.atom)?)?)?;
    }
    let queries = program.queries.len();
    let rules = add(program.rules.len(), queries)?;
    let relations = add(program.predicates.len(), queries)?;
    let edges = program
        .rules
        .iter()
        .try_fold(queries, |edges, rule| add(edges, rule.body.len()))?;
    let max_coordinates = program.rules.iter().try_fold(0usize, |maximum, rule| {
        let coordinates = rule.body.iter().try_fold(0usize, |count, literal| {
            let BodyLiteral::Positive(atom) = literal else {
                return Err(XlogError::Compilation(
                    "positive task compiler capacity requires positive original body atoms".into(),
                ));
            };
            add(count, atom.terms.len())
        })?;
        Ok::<_, XlogError>(maximum.max(coordinates).max(rule.head.terms.len()))
    })?;
    let max_coordinates = program
        .queries
        .iter()
        .fold(max_coordinates, |maximum, query| {
            maximum.max(query.atom.terms.len())
        });
    // Source snapshot, outer meta/list normalizers, prepared frontend clone,
    // query desugaring, inner meta/list normalizers, and error/reference copies.
    // These passes retain no generated finite lists or compound terms in the
    // canonically validated positive scalar subset.
    let mut bytes = mul(ast, 8)?;
    for query in &program.queries {
        let generated_ast = add(
            size_of::<AstRule>(),
            add(
                size_of::<Atom>(),
                add(
                    size_of::<BodyLiteral>(),
                    mul(add(query.atom.terms.len(), 1)?, size_of::<Term>())?,
                )?,
            )?,
        )?;
        bytes = add(bytes, mul(add(generated_ast, mul(max_name, 3)?)?, 8)?)?;
    }
    // Stratification and lowering each own a dependency graph. Tarjan's
    // indices/lowlinks, stack/on-stack sets, successor arrays and SCC/stratum
    // snapshots are included separately, not as a caller-supplied graph count.
    let edge = add(size_of::<DepEdge>(), mul(max_name, 2)?)?;
    let graph = add(
        size_of::<DependencyGraph>(),
        add(
            add(vector(edges, edge)?, vector(edges, size_of::<usize>())?)?,
            add(
                table(relations, add(size_of::<String>(), max_name)?)?,
                table(relations, add(size_of::<(String, Vec<usize>)>(), max_name)?)?,
            )?,
        )?,
    )?;
    let tarjan = add(
        mul(
            table(relations, add(size_of::<(String, usize)>(), max_name)?)?,
            2,
        )?,
        add(
            table(relations, add(size_of::<String>(), max_name)?)?,
            add(
                vector(relations, add(size_of::<String>(), max_name)?)?,
                vector(edges, add(size_of::<String>(), max_name)?)?,
            )?,
        )?,
    )?;
    let strata = add(
        vector(relations, size_of::<Stratum>())?,
        add(
            vector(relations, size_of::<Vec<String>>())?,
            vector(relations, add(size_of::<String>(), max_name)?)?,
        )?,
    )?;
    bytes = add(bytes, mul(add(add(graph, tarjan)?, strata)?, 4)?)?;
    // Meta/list/safety environments, schema inference, relation ids and fact
    // counters all use the original relation/coordinate rosters. Three
    // optimizer schema snapshots plus outer returned schemas remain distinct.
    let schema = add(
        size_of::<Schema>(),
        add(
            vector(max_arity, size_of::<(String, xlog_core::ScalarType)>())?,
            add(
                vector(max_arity, size_of::<String>())?,
                add(
                    mul(mul(max_arity, 2)?, max_name)?,
                    vector(max_arity, size_of::<usize>())?,
                )?,
            )?,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(relations, add(size_of::<String>(), add(max_name, schema)?)?)?,
            8,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(
                relations,
                add(size_of::<(String, xlog_core::RelId)>(), max_name)?,
            )?,
            4,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(rules, add(size_of::<(String, Vec<usize>)>(), max_name)?)?,
            4,
        )?,
    )?;
    let coordinate = add(size_of::<(String, xlog_core::ScalarType)>(), max_name)?;
    bytes = add(
        bytes,
        mul(table(max_coordinates, coordinate)?, add(rules, 4)?)?,
    )?;
    bytes = add(
        bytes,
        table(
            relations,
            size_of::<(xlog_core::RelId, xlog_stats::RelationStats)>(),
        )?,
    )?;
    bytes = add(
        bytes,
        crate::lower::positive_task_lowering_allocation_bound(program, max_name)?,
    )?;
    // Generated-head provenance and sorting/reference arrays are live alongside
    // the plan; a failed semantic check may format every original identifier.
    bytes = add(
        bytes,
        vector(queries, size_of::<xlog_ir::GeneratedQueryRuleProvenance>())?,
    )?;
    bytes = add(bytes, vector(rules, size_of::<(usize, usize)>())?)?;
    bytes = add(bytes, mul(names, 2)?)?;
    bytes = add(
        bytes,
        add(
            size_of::<Compiler>(),
            add(
                size_of::<Optimizer>(),
                add(size_of::<StatsManager>(), 6 * size_of::<usize>())?,
            )?,
        )?,
    )?;
    u64::try_from(bytes).map_err(|_| overflow())
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    /// Create a new compiler instance.
    pub fn new() -> Self {
        Self {
            lowerer: Lowerer::new(),
        }
    }

    /// Set the maximum active rules for TensorMaskedJoin (16..=128).
    pub fn set_max_active_rules(&mut self, max: usize) {
        self.lowerer.set_max_active_rules(max);
    }

    /// Compile XLOG source code into an execution plan.
    ///
    /// This is the main entry point for core compilation. It chains together:
    /// 1. Parsing (source → AST)
    /// 2. Stratification (analyze dependencies, check for cycles)
    /// 3. Lowering (AST → Relational IR execution plan)
    ///
    /// Function definitions and calls must already be expanded into the core
    /// language. Use an execution-facing `LogicProgram` API when source-level
    /// import resolution and function normalization are required.
    ///
    /// # Arguments
    ///
    /// * `source` - The XLOG source code as a string
    ///
    /// # Returns
    ///
    /// * `Ok(ExecutionPlan)` - The compiled execution plan ready for execution
    /// * `Err(XlogError)` - If any compilation phase fails:
    ///   - `XlogError::Parse` - Syntax errors in the source
    ///   - `XlogError::StratificationCycle` - Unstratifiable negation/aggregation
    ///   - `XlogError::Compilation` - Other semantic errors
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut compiler = Compiler::new();
    ///
    /// // Compile a simple transitive closure program
    /// let plan = compiler.compile(r#"
    ///     edge(1, 2).
    ///     edge(2, 3).
    ///     reach(X, Y) :- edge(X, Y).
    ///     reach(X, Z) :- reach(X, Y), edge(Y, Z).
    /// "#)?;
    ///
    /// // The plan can now be executed by xlog-runtime
    /// ```
    pub fn compile(&mut self, source: &str) -> Result<ExecutionPlan> {
        self.compile_with_stats_snapshot(source, None)
    }

    /// Compile XLOG source code into an execution plan, optionally seeding the optimizer
    /// with a runtime statistics snapshot.
    ///
    /// This entry point delegates through the composable config-aware API
    /// with `CompilerConfig::default()`, which preserves existing triangle,
    /// 4-cycle, recursive, and selectivity-aware dispatch behavior
    /// bit-identically.
    pub fn compile_with_stats_snapshot(
        &mut self,
        source: &str,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_with_config_and_stats_snapshot(
            source,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Composable entry point that accepts a `CompilerConfig`.
    ///
    /// Default-config callers should keep using `compile()` /
    /// `compile_with_stats_snapshot()`. This entry point exists so callers can
    /// enable the variable-ordering cost model per call without an environment
    /// override.
    pub fn compile_with_config_and_stats_snapshot(
        &mut self,
        source: &str,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        let program = parse_program(source)?;
        self.compile_owned_program_with_config_and_stats_snapshot(program, config, stats_snapshot)
    }

    /// Compile a parsed XLOG program into an execution plan.
    ///
    /// This is useful for callers that want to inspect the AST (facts, queries,
    /// constraints) while compiling without reparsing. The parsed program must
    /// already be normalized to the core language, including function expansion.
    pub fn compile_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.compile_program_with_stats_snapshot(program, None)
    }

    /// Compile a program whose authored constraint identities were prepared at
    /// the outer full-program boundary.
    pub fn compile_prepared_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.compile_prepared_program_with_stats_snapshot(program, None)
    }

    /// Compile a parsed XLOG program into an execution plan, optionally seeding the optimizer.
    ///
    /// Delegates to [`Self::compile_program_with_config_and_stats_snapshot`]
    /// with `CompilerConfig::default()`.
    pub fn compile_program_with_stats_snapshot(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_program_with_config_and_stats_snapshot(
            program,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Compile an already prepared program with an optional statistics snapshot.
    pub fn compile_prepared_program_with_stats_snapshot(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_program_with_config_and_stats_snapshot(
            program,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Validate a parsed program through the production lowering contracts without
    /// requiring the dependency graph to be stratifiable.
    ///
    /// This is used by epistemic preparation, where a supported negated modal cycle
    /// must reach well-founded execution rather than fail ordinary stratification.
    /// The normal preprocessing, negation-safety, schema/type, constant, arithmetic,
    /// and projection checks still run unchanged.
    pub(crate) fn validate_program_without_stratification(
        &mut self,
        program: &Program,
    ) -> Result<()> {
        program.validate_prepared_authored_constraint_identity()?;
        let program =
            run_frontend_passes_with_owner(CompilationValue::Unowned(program.clone()), &mut None)?;
        self.lowerer.validate_program_without_plan(&program)
    }

    pub(crate) fn validate_program_without_stratification_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        let mut stage = scope.child_scope();
        let result = (|| {
            program.validate_prepared_authored_constraint_identity_with_allocation_owner(
                &mut |bytes| stage.reserve_allocation(bytes),
            )?;
            let input = clone_compilation_program(program, &mut stage)?;
            let program = run_frontend_passes_with_owner(
                CompilationValue::Owned(input),
                &mut Some(&mut stage),
            )?;
            self.lowerer
                .validate_program_without_plan_with_allocation_owner(&program, &mut stage)
        })();
        finish_compiler_stage(result, stage)
    }

    /// Composable program-level entry point.
    ///
    /// `config` is currently consumed only by the promoter when it wires the
    /// variable-ordering cost model. With `CompilerConfig::default()`, the
    /// promoter keeps the default variable order.
    pub fn compile_program_with_config_and_stats_snapshot(
        &mut self,
        program: &Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_owned_program_with_config_and_stats_snapshot(
            program.clone(),
            config,
            stats_snapshot,
        )
    }

    /// [`Self::compile_program_with_config_and_stats_snapshot`] for a program
    /// the caller hands over by value (no clone of the AST).
    pub fn compile_owned_program_with_config_and_stats_snapshot(
        &mut self,
        mut program: Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        if program.authored_constraint_source_bound.is_some() {
            program.validate_prepared_authored_constraint_identity()?;
        } else {
            program.prepare_authored_constraint_identity_at_root()?;
        }
        self.compile_prepared_owned_program_with_config_and_stats_snapshot(
            program,
            config,
            stats_snapshot,
        )
    }

    /// Config-aware compilation for an already prepared program.
    pub fn compile_prepared_program_with_config_and_stats_snapshot(
        &mut self,
        program: &Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_owned_program_with_config_and_stats_snapshot(
            program.clone(),
            config,
            stats_snapshot,
        )
    }

    /// Compile through the original complete pipeline. Persistent lowerer state
    /// owns its receipts; discarded compiler data retires with its local stage.
    ///
    /// The protective input clone acquires its physical heap allocations and
    /// typed custody metadata through the caller's original scope before birth.
    /// Buffer and payload receipts move with their corresponding data through
    /// every frontend pass. This does not transfer or divide the external
    /// caller's original receipts.
    pub fn compile_prepared_program_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        self.compile_with_allocation_stage(scope, |compiler, stage| {
            let input = clone_compilation_program(program, stage)?;
            let output = compiler.compile_prepared_owned_program_core(
                CompilationValue::Owned(input),
                &CompilerConfig::default(),
                None,
                &mut Some(stage),
            )?;
            match output {
                CompilationValue::Owned(output) => Ok(output),
                CompilationValue::Unowned(_) => unreachable!("original owned compiler output"),
            }
        })
    }

    /// Statistics-aware form using the same original compilation owner.
    pub fn compile_prepared_program_with_stats_snapshot_and_allocation_owner(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        self.compile_with_allocation_stage(scope, |compiler, stage| {
            let input = clone_compilation_program(program, stage)?;
            let output = compiler.compile_prepared_owned_program_core(
                CompilationValue::Owned(input),
                &CompilerConfig::default(),
                stats_snapshot,
                &mut Some(stage),
            )?;
            match output {
                CompilationValue::Owned(output) => Ok(output),
                CompilationValue::Unowned(_) => unreachable!("original owned compiler output"),
            }
        })
    }

    /// Owned-AST form of the same allocation-aware compilation pipeline.
    pub fn compile_prepared_owned_program_with_allocation_owner(
        &mut self,
        program: Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        self.compile_with_allocation_stage(scope, |compiler, stage| {
            // Incoming external custody belongs to the caller. A guarded
            // local copy makes only this compiler's AST lifetime separable.
            let input = clone_compilation_program(&program, stage)?;
            drop(program);
            let output = compiler.compile_prepared_owned_program_core(
                CompilationValue::Owned(input),
                &CompilerConfig::default(),
                None,
                &mut Some(stage),
            )?;
            match output {
                CompilationValue::Owned(output) => Ok(output),
                CompilationValue::Unowned(_) => unreachable!("original owned compiler output"),
            }
        })
    }

    fn compile_with_allocation_stage<F>(
        &mut self,
        scope: &mut CompilationAllocationScope<'_>,
        compile: F,
    ) -> Result<ExecutionPlan>
    where
        F: FnOnce(
            &mut Self,
            &mut CompilationAllocationScope<'_>,
        ) -> Result<CompilationOutput<ExecutionPlan>>,
    {
        let output = {
            let mut stage = scope.child_scope();
            let result = compile(self, &mut stage);
            finish_compiler_stage(result, stage)?
        };
        // Keep the temporary carrier alive if receipt-storage acquisition
        // declines: its plan drops before its original allocation owner.
        scope.retain(Arc::clone(&output.allocation_owner))?;
        Ok(output.value)
    }

    /// [`Self::compile_prepared_program_with_config_and_stats_snapshot`] for a
    /// program handed over by value.
    ///
    /// The frontend passes (desugar → meta → list → magic sets) each take and
    /// return the `Program` by value, so a fact-heavy AST is never cloned
    /// between passes; the `&Program` entry points clone exactly once.
    pub fn compile_prepared_owned_program_with_config_and_stats_snapshot(
        &mut self,
        program: Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        let output = self.compile_prepared_owned_program_core(
            CompilationValue::Unowned(program),
            config,
            stats_snapshot,
            &mut None,
        )?;
        match output {
            CompilationValue::Unowned(plan) => Ok(plan),
            CompilationValue::Owned(_) => unreachable!("ordinary unowned compiler output"),
        }
    }

    fn compile_prepared_owned_program_core(
        &mut self,
        program: CompilationProgram,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<CompilationValue<ExecutionPlan>> {
        match scope.as_deref_mut() {
            Some(scope) => program
                .validate_prepared_authored_constraint_identity_with_allocation_owner(
                    &mut |bytes| scope.reserve_allocation(bytes),
                ),
            None => program.validate_prepared_authored_constraint_identity(),
        }?;
        let generated_query_heads = generated_query_heads_output_with_owner(&program, scope)?;
        let program = run_frontend_passes_with_owner(program, scope)?;

        // Analyze dependencies and retain only the original lowerer state.
        self.prepare_strata_with_owner(&program, scope)?;

        set_compilation_cardinality_hints(&mut self.lowerer, stats_snapshot, scope)?;

        let mut plan = match scope.as_deref_mut() {
            Some(scope) => CompilationValue::Owned(
                self.lowerer
                    .lower_program_guarded_with_allocation_owner(&program, scope)?,
            ),
            None => CompilationValue::Unowned(self.lowerer.lower_program(&program)?),
        };

        let stats_arc =
            compilation_statistics_with_owner(&self.lowerer, &program, stats_snapshot, scope)?;
        drop(program);

        // Build schemas by RelId for the optimizer
        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        plan = match (plan, scope.as_deref_mut()) {
            (CompilationValue::Owned(input), Some(scope)) => {
                CompilationValue::Owned(crate::optimizer::helper_split_pass::run_guarded_plan(
                    input,
                    &schemas_by_rel_id,
                    &stats_arc,
                    scope,
                    |schema, scope| {
                        self.lowerer
                            .create_helper_relation_with_allocation_owner(schema, scope)
                    },
                )?)
            }
            (CompilationValue::Unowned(mut plan), None) => {
                crate::optimizer::helper_split_pass::run(
                    &mut plan,
                    &schemas_by_rel_id,
                    &stats_arc,
                    |schema| self.lowerer.create_helper_relation(schema),
                );
                CompilationValue::Unowned(plan)
            }
            _ => unreachable!("canonical plan allocation custody"),
        };

        drop(schemas_by_rel_id);
        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        let optimizer = match schemas_by_rel_id {
            CompilationValue::Owned(schemas) => {
                let mut optimizer = Optimizer::new(Arc::clone(&stats_arc));
                optimizer.set_schemas(schemas.value);
                CompilationValue::Owned(CompilationOutput {
                    value: optimizer,
                    allocation_owner: schemas.allocation_owner,
                })
            }
            CompilationValue::Unowned(schemas) => {
                let mut optimizer = Optimizer::new(Arc::clone(&stats_arc));
                optimizer.set_schemas(schemas);
                CompilationValue::Unowned(optimizer)
            }
        };
        plan = match (plan, scope.as_deref_mut()) {
            (CompilationValue::Owned(input), Some(scope)) => {
                CompilationValue::Owned(optimizer.optimize_guarded_plan(input, scope)?)
            }
            (CompilationValue::Unowned(mut plan), None) => {
                for rules in &mut plan.rules_by_scc {
                    for rule in rules {
                        let body = std::mem::replace(&mut rule.body, xlog_ir::RirNode::Unit);
                        rule.body = optimizer.optimize(body);
                    }
                }
                CompilationValue::Unowned(plan)
            }
            _ => unreachable!("canonical plan allocation custody"),
        };

        drop(optimizer);

        // Selectivity-aware reordering pass. Runs BETWEEN the optimizer loop
        // and promote_multiway.
        // Locked compile-pipeline ordering:
        //   lower → helper_split_pass → optimizer → selectivity_pass → promote_multiway
        //
        // Takes `rel_ids` so per-body Scans can be resolved against
        // `StatsManager`. Behavior on empty stats / unseeded relations is
        // no-op (safety floor).
        plan = match (plan, scope.as_deref_mut()) {
            (CompilationValue::Owned(input), Some(scope)) => {
                CompilationValue::Owned(crate::optimizer::selectivity_pass::run_guarded_plan(
                    input,
                    &stats_arc,
                    self.lowerer.rel_ids(),
                    scope,
                )?)
            }
            (CompilationValue::Unowned(mut plan), None) => {
                crate::optimizer::selectivity_pass::run(
                    &mut plan,
                    &stats_arc,
                    self.lowerer.rel_ids(),
                );
                CompilationValue::Unowned(plan)
            }
            _ => unreachable!("canonical plan allocation custody"),
        };

        // Promote eligible triangle subtrees to RirNode::MultiWayJoin. Runs
        // *after* the optimizer so the optimizer never has to learn the new
        // variant. Fallback identity preserves binary-join semantics on
        // dispatch decline.
        //
        // Pass the lowerer's predicate→RelId map so the promoter can gate
        // recursive-SCC bodies on the count of in-SCC Scans (≤ 1 = promote,
        // ≥ 2 = skip).
        //
        // Also pass `&stats_arc` and the caller-provided `&CompilerConfig`.
        // With `CompilerConfig::default()` (`Disabled`), the promoter never
        // sets `var_order` and default dispatch is bit-identical.
        plan = match (plan, scope.as_deref_mut()) {
            (CompilationValue::Owned(input), Some(scope)) => {
                CompilationValue::Owned(crate::promote::promote_multiway_guarded_plan(
                    input,
                    self.lowerer.rel_ids(),
                    &stats_arc,
                    config,
                    scope,
                )?)
            }
            (CompilationValue::Unowned(mut plan), None) => {
                crate::promote::promote_multiway(
                    &mut plan,
                    self.lowerer.rel_ids(),
                    &stats_arc,
                    config,
                );
                CompilationValue::Unowned(plan)
            }
            _ => unreachable!("canonical plan allocation custody"),
        };

        drop(stats_arc);
        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        plan = match (plan, scope.as_deref_mut()) {
            (CompilationValue::Owned(input), Some(scope)) => CompilationValue::Owned(
                crate::optimizer::helper_split_pass::run_kclique_specs_guarded_plan(
                    input,
                    &schemas_by_rel_id,
                    scope,
                    |schema, scope| {
                        self.lowerer
                            .create_helper_relation_with_allocation_owner(schema, scope)
                    },
                )?,
            ),
            (CompilationValue::Unowned(mut plan), None) => {
                crate::optimizer::helper_split_pass::run_kclique_specs(
                    &mut plan,
                    &schemas_by_rel_id,
                    |schema| self.lowerer.create_helper_relation(schema),
                );
                CompilationValue::Unowned(plan)
            }
            _ => unreachable!("canonical plan allocation custody"),
        };

        drop(schemas_by_rel_id);

        plan = attach_generated_query_provenance(plan, &generated_query_heads, scope)?;
        drop(generated_query_heads);

        Ok(plan)
    }

    fn prepare_strata_with_owner(
        &mut self,
        program: &Program,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        match scope.as_deref_mut() {
            Some(scope) => {
                let mut stage = scope.child_scope();
                let result = self.prepare_strata_core(program, &mut Some(&mut stage));
                finish_compiler_stage(result, stage)
            }
            None => self.prepare_strata_core(program, scope),
        }
    }

    fn prepare_strata_core(
        &mut self,
        program: &Program,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        let strata = match scope.as_deref_mut() {
            Some(scope) => crate::stratify::stratify_with_allocation_owner(program, scope),
            None => stratify(program),
        }
        .map_err(|error| map_stratification_to_naf_error_with_owner(error, scope))?;
        acquire_frontend_vector::<Vec<String>>(scope, strata.len())?;
        let strata_preds = strata
            .into_iter()
            .map(|stratum| stratum.predicates)
            .collect();
        match scope.as_deref_mut() {
            Some(scope) => self
                .lowerer
                .set_strata_with_allocation_owner(strata_preds, scope),
            None => {
                self.lowerer.set_strata(strata_preds);
                Ok(())
            }
        }
    }

    /// Reset the compiler state for a fresh compilation.
    ///
    /// This creates a new lowerer, clearing any cached schemas or relation IDs
    /// from previous compilations.
    pub fn reset(&mut self) {
        self.lowerer = Lowerer::new();
    }

    /// Get the mapping from predicate names to relation IDs after compilation.
    ///
    /// This mapping is needed to register relations in the executor with
    /// the correct RelIds.
    pub fn rel_ids(&self) -> &HashMap<String, RelId> {
        self.lowerer.rel_ids()
    }

    /// Get the inferred schemas for predicates after compilation.
    ///
    /// These schemas are needed to create GPU buffers with correct column types.
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        self.lowerer.schemas()
    }
}

fn generated_query_heads_for_program(program: &Program) -> Result<Vec<String>> {
    generated_query_heads_for_program_with_owner(program, &mut None)
}

// Every reached caller returns either inline validation state or the guarded
// plan carrier. Canonical error producers own their diagnostic payloads; the
// stratification cycle is mapped before this boundary. No discarded scratch
// is transferred into either successful output or a diagnostic.
pub(crate) fn finish_compiler_stage<T>(
    result: Result<T>,
    stage: CompilationAllocationScope<'_>,
) -> Result<T> {
    drop(stage);
    result
}

fn schemas_snapshot_with_owner(
    lowerer: &Lowerer,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<CompilationValue<HashMap<RelId, Schema>>> {
    let count = lowerer
        .rel_ids()
        .keys()
        .filter(|name| lowerer.schemas().contains_key(*name))
        .count();
    let allocation_owner = match scope.as_deref_mut() {
        Some(scope) => {
            let mut bytes = hash_allocation_bytes::<RelId, Schema>(count)?;
            for name in lowerer.rel_ids().keys() {
                if let Some(schema) = lowerer.schemas().get(name) {
                    bytes = bytes
                        .checked_add(schema.owned_allocation_bytes()?)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("schema snapshot"))?;
                }
            }
            Some(scope.reserve_allocation(bytes)?)
        }
        None => None,
    };
    // The original roster is known before insertion. Reserve its one actual
    // table so incremental collect growth creates no discarded hash buffers.
    let mut schemas = HashMap::with_capacity(count);
    for (name, id) in lowerer.rel_ids() {
        if let Some(schema) = lowerer.schemas().get(name) {
            schemas.insert(*id, schema.clone());
        }
    }
    Ok(match allocation_owner {
        Some(allocation_owner) => CompilationValue::Owned(CompilationOutput {
            value: schemas,
            allocation_owner,
        }),
        None => CompilationValue::Unowned(schemas),
    })
}

fn set_compilation_cardinality_hints(
    lowerer: &mut Lowerer,
    stats_snapshot: Option<&StatsSnapshot>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    match scope.as_deref_mut() {
        Some(scope) => {
            let mut stage = scope.child_scope();
            let result = set_compilation_cardinality_hints_core(
                lowerer,
                stats_snapshot,
                &mut Some(&mut stage),
            );
            finish_compiler_stage(result, stage)
        }
        None => set_compilation_cardinality_hints_core(lowerer, stats_snapshot, scope),
    }
}

fn set_compilation_cardinality_hints_core(
    lowerer: &mut Lowerer,
    stats_snapshot: Option<&StatsSnapshot>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    let mut cardinality_hints: HashMap<String, u64> = HashMap::new();
    if let Some(snapshot) = stats_snapshot {
        if !snapshot.rel_names.is_empty() {
            let _lookup_owner = match scope.as_deref_mut() {
                Some(scope) => Some(scope.reserve_allocation(hash_allocation_bytes::<
                    RelId,
                    &str,
                >(
                    snapshot.rel_names.len()
                )?)?),
                None => None,
            };
            let rel_name_by_id: HashMap<RelId, &str> = snapshot
                .rel_names
                .iter()
                .map(|(id, name)| (*id, name.as_str()))
                .collect();
            for rel in &snapshot.relations {
                if let Some(name) = rel_name_by_id.get(&rel.rel_id) {
                    if scope.is_some() {
                        if !cardinality_hints.contains_key(*name)
                            && cardinality_hints.len() == cardinality_hints.capacity()
                        {
                            acquire_frontend(
                                scope,
                                hash_allocation_bytes::<String, u64>(cardinality_hints.len() + 1)?,
                            )?;
                        }
                        acquire_frontend(
                            scope,
                            u64::try_from(name.len()).map_err(|_| {
                                XlogError::allocation_extent_overflow("cardinality name")
                            })?,
                        )?;
                    }
                    cardinality_hints.insert((*name).to_string(), rel.cardinality);
                }
            }
        }
    }
    match scope.as_deref_mut() {
        Some(scope) => {
            lowerer.set_cardinality_hints_with_allocation_owner(cardinality_hints, scope)?
        }
        None => lowerer.set_cardinality_hints(cardinality_hints),
    }
    Ok(())
}

fn compilation_statistics_with_owner(
    lowerer: &Lowerer,
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<CompilationValue<Arc<StatsManager>>> {
    match scope.as_deref_mut() {
        Some(scope) => {
            let mut stage = scope.child_scope();
            let manager = compilation_statistics_core(
                lowerer,
                program,
                stats_snapshot,
                &mut Some(&mut stage),
            )?;
            let bytes = manager
                .clone_allocation_bytes()?
                .checked_add(
                    u64::try_from(
                        std::mem::size_of::<StatsManager>() + 2 * std::mem::size_of::<usize>(),
                    )
                    .map_err(|_| XlogError::allocation_extent_overflow("statistics manager Arc"))?,
                )
                .ok_or_else(|| XlogError::allocation_extent_overflow("statistics manager clone"))?;
            let allocation_owner = stage.reserve_allocation(bytes)?;
            let value = Arc::new(manager.clone());
            drop(manager);
            drop(stage);
            Ok(CompilationValue::Owned(CompilationOutput {
                value,
                allocation_owner,
            }))
        }
        None => Ok(CompilationValue::Unowned(Arc::new(
            compilation_statistics_core(lowerer, program, stats_snapshot, scope)?,
        ))),
    }
}

fn compilation_statistics_core(
    lowerer: &Lowerer,
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<StatsManager> {
    if scope.is_some() {
        acquire_frontend(
            scope,
            hash_allocation_bytes::<RelId, xlog_stats::RelationStats>(lowerer.rel_ids().len())?,
        )?;
    }
    let mut mgr = StatsManager::new();
    {
        let fact_count = program.facts().count();
        let _counts_owner = match scope.as_deref_mut() {
            Some(scope) => {
                Some(scope.reserve_allocation(hash_allocation_bytes::<&str, u64>(fact_count)?)?)
            }
            None => None,
        };
        let mut fact_counts: HashMap<&str, u64> = HashMap::with_capacity(fact_count);
        for fact in program.facts() {
            *fact_counts.entry(fact.head.predicate.as_str()).or_insert(0) += 1;
        }

        for (pred, rel_id) in lowerer.rel_ids() {
            mgr.register_relation(*rel_id);
            let rows = fact_counts.get(pred.as_str()).copied().unwrap_or(0);
            if rows > 0 {
                mgr.update_cardinality(*rel_id, rows);
                if let Some(schema) = lowerer.schemas().get(pred) {
                    mgr.update_byte_size(*rel_id, rows * schema.row_size_bytes() as u64);
                }
            }
        }
    }

    if let Some(snapshot) = stats_snapshot {
        reserve_stats_snapshot_merge(snapshot, lowerer.rel_ids().len(), scope)?;
        if snapshot.rel_names.is_empty() {
            mgr.merge_snapshot(snapshot);
        } else {
            let _lookup_owner = match scope.as_deref_mut() {
                Some(scope) => Some(scope.reserve_allocation(hash_allocation_bytes::<
                    RelId,
                    &str,
                >(
                    snapshot.rel_names.len()
                )?)?),
                None => None,
            };
            let rel_name_by_id: HashMap<RelId, &str> = snapshot
                .rel_names
                .iter()
                .map(|(id, name)| (*id, name.as_str()))
                .collect();

            for rel in &snapshot.relations {
                let Some(pred) = rel_name_by_id.get(&rel.rel_id) else {
                    continue;
                };
                let Some(rel_id) = lowerer.rel_ids().get(*pred) else {
                    continue;
                };

                let mut remapped = rel.clone();
                remapped.rel_id = *rel_id;

                if let Some(schema) = lowerer.schemas().get(*pred) {
                    remapped.column_stats.retain(|col| {
                        col.col_idx < schema.arity()
                            && schema.column_type(col.col_idx) == Some(col.dtype)
                    });
                } else {
                    remapped.column_stats.clear();
                }

                mgr.register_relation(*rel_id);
                if let Some(stats) = mgr.get_relation_stats_mut(*rel_id) {
                    *stats = remapped;
                }
            }

            for js in &snapshot.join_selectivities {
                if js.left_keys.len() != js.right_keys.len() {
                    continue;
                }

                let Some(left_pred) = rel_name_by_id.get(&js.left_rel) else {
                    continue;
                };
                let Some(right_pred) = rel_name_by_id.get(&js.right_rel) else {
                    continue;
                };
                let Some(&left_id) = lowerer.rel_ids().get(*left_pred) else {
                    continue;
                };
                let Some(&right_id) = lowerer.rel_ids().get(*right_pred) else {
                    continue;
                };

                let Some(left_schema) = lowerer.schemas().get(*left_pred) else {
                    continue;
                };
                let Some(right_schema) = lowerer.schemas().get(*right_pred) else {
                    continue;
                };
                if js.left_keys.iter().any(|&k| k >= left_schema.arity())
                    || js.right_keys.iter().any(|&k| k >= right_schema.arity())
                {
                    continue;
                }

                mgr.set_join_selectivity(
                    left_id,
                    right_id,
                    js.left_keys.clone(),
                    js.right_keys.clone(),
                    js.selectivity,
                );
            }
        }
    }
    Ok(mgr)
}

fn reserve_stats_snapshot_merge(
    snapshot: &StatsSnapshot,
    original_relations: usize,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    if scope.is_none() {
        return Ok(());
    }
    acquire_frontend(
        scope,
        hash_allocation_bytes::<RelId, xlog_stats::RelationStats>(
            original_relations
                .checked_add(snapshot.relations.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("statistics registry"))?,
        )?,
    )?;
    acquire_frontend(
        scope,
        hash_allocation_bytes::<(RelId, RelId), xlog_stats::JoinSelectivity>(
            snapshot.join_selectivities.len(),
        )?,
    )?;
    for relation in &snapshot.relations {
        acquire_frontend_vector::<xlog_stats::ColumnStats>(scope, relation.column_stats.len())?;
        acquire_frontend_vector::<xlog_stats::PrefixDegreeStats>(
            scope,
            relation.prefix_degrees.len(),
        )?;
        acquire_frontend_vector::<xlog_stats::KeyHeatStats>(scope, relation.key_heats.len())?;
    }
    for selectivity in &snapshot.join_selectivities {
        // Caller key copies and the genuine cache entry's original key copies.
        for _ in 0..2 {
            acquire_frontend_vector::<usize>(scope, selectivity.left_keys.len())?;
            acquire_frontend_vector::<usize>(scope, selectivity.right_keys.len())?;
        }
    }
    Ok(())
}

struct QueryProvenanceAllocations {
    _plan: Arc<dyn Send + Sync>,
    _provenance: Arc<dyn Send + Sync>,
    _metadata: Arc<dyn Send + Sync>,
}

fn attach_generated_query_provenance(
    input: CompilationValue<ExecutionPlan>,
    heads: &[String],
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<CompilationValue<ExecutionPlan>> {
    match (input, scope.as_deref_mut()) {
        (CompilationValue::Owned(mut input), Some(scope)) => {
            // An empty query roster preserves the original empty plan vector.
            if heads.is_empty() {
                input.value.generated_query_rules =
                    generated_query_provenance_core(&input.value, heads, &mut Some(scope))?;
                return Ok(CompilationValue::Owned(input));
            }
            let provenance_owner = scope.reserve_allocation(push_vector_allocation_bytes::<
                GeneratedQueryRuleProvenance,
            >(heads.len())?)?;
            let metadata_owner = scope.reserve_allocation(
                u64::try_from(
                    std::mem::size_of::<QueryProvenanceAllocations>()
                        + 2 * std::mem::size_of::<usize>(),
                )
                .map_err(|_| XlogError::allocation_extent_overflow("query provenance owner"))?,
            )?;
            let provenance =
                generated_query_provenance_core(&input.value, heads, &mut Some(scope))?;
            // All fallible work is complete. The original plan and the new
            // payload remain charged through this genuine owner transfer.
            let allocation_owner = Arc::new(QueryProvenanceAllocations {
                _plan: input.allocation_owner,
                _provenance: provenance_owner,
                _metadata: metadata_owner,
            });
            input.value.generated_query_rules = provenance;
            Ok(CompilationValue::Owned(CompilationOutput {
                value: input.value,
                allocation_owner,
            }))
        }
        (CompilationValue::Unowned(mut plan), None) => {
            plan.generated_query_rules = generated_query_provenance_core(&plan, heads, &mut None)?;
            Ok(CompilationValue::Unowned(plan))
        }
        _ => unreachable!("canonical query provenance allocation custody"),
    }
}

fn generated_query_provenance_core(
    plan: &ExecutionPlan,
    heads: &[String],
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Vec<GeneratedQueryRuleProvenance>> {
    heads
        .iter()
        .enumerate()
        .map(|(query_index, head)| {
            let count = plan
                .rules_by_scc
                .iter()
                .try_fold(0usize, |n, rules| n.checked_add(rules.len()))
                .ok_or_else(|| XlogError::allocation_extent_overflow("query provenance"))?;
            let _positions_receipt =
                match scope.as_deref_mut() {
                    Some(scope) => Some(scope.reserve_allocation(
                        push_vector_allocation_bytes::<(usize, usize)>(count)?,
                    )?),
                    None => None,
                };
            let positions = plan
                .rules_by_scc
                .iter()
                .enumerate()
                .flat_map(|(scc_index, rules)| {
                    rules
                        .iter()
                        .enumerate()
                        .filter_map(move |(rule_index, rule)| {
                            (rule.head == *head).then_some((scc_index, rule_index))
                        })
                })
                .collect::<Vec<_>>();
            let [(scc_index, rule_index)] = positions.as_slice() else {
                return Err(frontend_compilation_error(
                    format_args!(
                        "generated query head {head} must have exactly one compiled rule, found {}",
                        positions.len()
                    ),
                    scope,
                ));
            };
            Ok(GeneratedQueryRuleProvenance {
                query_index,
                scc_index: *scc_index,
                rule_index: *rule_index,
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn generated_query_heads_output_with_owner(
    program: &Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<CompilationValue<Vec<String>>> {
    match scope.as_deref_mut() {
        Some(scope) => {
            let mut output_scope = scope.child_scope();
            output_scope.prepare_output_owner()?;
            let value = generated_query_heads_for_program_with_owner(
                program,
                &mut Some(&mut output_scope),
            )?;
            let allocation_owner = output_scope.into_owner()?;
            Ok(CompilationValue::Owned(CompilationOutput {
                value,
                allocation_owner,
            }))
        }
        None => Ok(CompilationValue::Unowned(
            generated_query_heads_for_program_with_owner(program, scope)?,
        )),
    }
}

fn generated_query_heads_for_program_with_owner(
    program: &Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Vec<String>> {
    acquire_frontend_vector::<String>(scope, program.queries.len())?;
    (0..program.queries.len())
        .map(|query_index| {
            acquire_frontend_format(scope, format_args!("__xlog_query_{query_index}"))?;
            let generated_head = format!("__xlog_query_{query_index}");
            let authored_collision = program
                .predicates
                .iter()
                .any(|declaration| declaration.name == generated_head)
                || program
                    .rules
                    .iter()
                    .any(|rule| rule.head.predicate == generated_head)
                || program
                    .learnable_rules
                    .iter()
                    .any(|rule| rule.head.predicate == generated_head);
            if authored_collision {
                return Err(frontend_compilation_error(
                    format_args!(
                        "authored relation {generated_head} collides with generated query head"
                    ),
                    scope,
                ));
            }
            Ok(generated_head)
        })
        .collect()
}

#[derive(Clone, Copy)]
enum FrontendPass {
    Desugar,
    Meta,
    List,
    Magic,
}

impl FrontendPass {
    fn run(
        self,
        program: Program,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
        custody: Option<&mut crate::program_custody::ProgramAllocationCustody>,
    ) -> Result<Program> {
        match (scope.as_deref_mut(), custody) {
            (Some(scope), Some(custody)) => {
                let mut input = crate::program_custody::RetainedProgram {
                    value: program,
                    custody: std::mem::take(custody),
                };
                let output = match self {
                    Self::Desugar => {
                        input.value = desugar_queries_and_constraints_with_owner(
                            input.value,
                            &mut Some(scope),
                            Some(&mut input.custody),
                        )?;
                        input
                    }
                    Self::Meta => {
                        crate::meta_normalize::normalize_meta_builtins_retained(input, scope)?
                    }
                    Self::List => {
                        crate::list_normalize::normalize_list_builtins_retained(input, scope)?
                    }
                    Self::Magic => {
                        let rewritten =
                            crate::magic_sets::rewrite_magic_sets_retained(input, scope)?;
                        drop(rewritten.report);
                        rewritten.program
                    }
                };
                *custody = output.custody;
                Ok(output.value)
            }
            (None, None) => match self {
                Self::Desugar => {
                    desugar_queries_and_constraints_with_owner(program, &mut None, None)
                }
                Self::Meta => normalize_meta_builtins_owned(program),
                Self::List => normalize_list_builtins_owned(program),
                Self::Magic => {
                    let rewritten = rewrite_magic_sets_owned(program)?;
                    drop(rewritten.report);
                    Ok(rewritten.program)
                }
            },
            _ => unreachable!("canonical frontend allocation custody"),
        }
    }
}

fn clone_compilation_program(
    program: &Program,
    scope: &mut CompilationAllocationScope<'_>,
) -> Result<CompilationOutput<Program, crate::program_custody::ProgramAllocationCustody>> {
    let retained = program.clone_with_allocation_custody(scope)?;
    Ok(CompilationOutput {
        value: retained.value,
        allocation_owner: retained.custody,
    })
}

fn run_frontend_passes_with_owner(
    mut input: CompilationProgram,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<CompilationProgram> {
    for pass in [
        FrontendPass::Desugar,
        FrontendPass::Meta,
        FrontendPass::List,
        FrontendPass::Magic,
    ] {
        input = match (input, scope.as_deref_mut()) {
            (CompilationValue::Unowned(program), None) => {
                let program = pass.run(program, &mut None, None)?;
                CompilationValue::Unowned(program)
            }
            (CompilationValue::Owned(mut input), Some(scope)) => {
                let mut stage = scope.child_scope();
                let program = pass.run(
                    input.value,
                    &mut Some(&mut stage),
                    Some(&mut input.allocation_owner),
                )?;
                // Every pass returns real data with its original typed custody;
                // only the pass's independent scratch scope retires here.
                drop(stage);
                CompilationValue::Owned(CompilationOutput {
                    value: program,
                    allocation_owner: input.allocation_owner,
                })
            }
            _ => unreachable!("canonical frontend allocation custody"),
        };
    }
    match scope.as_deref_mut() {
        Some(scope) => {
            let mut stage = scope.child_scope();
            let result = validate_negation_safety_with_owner(&input, &mut Some(&mut stage));
            finish_compiler_stage(result, stage)?;
        }
        None => validate_negation_safety_with_owner(&input, scope)?,
    }
    Ok(input)
}

fn acquire_frontend(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    bytes: u64,
) -> Result<()> {
    if bytes != 0 {
        if let Some(scope) = scope.as_deref_mut() {
            scope.acquire(bytes)?;
        }
    }
    Ok(())
}

fn acquire_frontend_vector<T>(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    count: usize,
) -> Result<()> {
    if scope.is_some() {
        acquire_frontend(scope, push_vector_allocation_bytes::<T>(count)?)?;
    }
    Ok(())
}

fn acquire_frontend_format(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    arguments: std::fmt::Arguments<'_>,
) -> Result<()> {
    if scope.is_some() {
        acquire_frontend(scope, formatted_text_allocation_bytes(arguments)?)?;
    }
    Ok(())
}

fn acquire_frontend_variables(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    literal: &BodyLiteral,
) -> Result<()> {
    if scope.is_none() {
        return Ok(());
    }
    let mut count = 0usize;
    let mut text = 0u64;
    let mut overflow = false;
    literal.visit_variables(&mut |name| {
        if let Some(next) = count.checked_add(1) {
            count = next;
        } else {
            overflow = true;
        }
        if let Some(next) = text.checked_add(name.len() as u64) {
            text = next;
        } else {
            overflow = true;
        }
    });
    if overflow {
        return Err(XlogError::allocation_extent_overflow("frontend variable"));
    }
    acquire_frontend_vector::<&str>(scope, count)?;
    acquire_frontend(scope, text)?;
    Ok(())
}

fn desugar_queries_and_constraints_with_owner(
    program: Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    mut custody: Option<&mut crate::program_custody::ProgramAllocationCustody>,
) -> Result<Program> {
    let mut out = program;
    let final_rules = out
        .rules
        .len()
        .checked_add(out.constraints.len())
        .and_then(|n| n.checked_add(out.queries.len()))
        .ok_or_else(|| XlogError::allocation_extent_overflow("frontend rule"))?;
    if let Some(custody) = custody.as_deref_mut() {
        let scope = scope
            .as_deref_mut()
            .expect("original Program custody scope");
        custody
            .rules
            .prepare_elements(final_rules - out.rules.len(), scope)?;
        if final_rules > out.rules.capacity() {
            let bytes = final_rules
                .checked_mul(std::mem::size_of::<AstRule>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("frontend rule"))?;
            let owner = scope.reserve_allocation(bytes)?;
            out.rules.reserve_exact(final_rules - out.rules.len());
            custody.rules.backing_owner = Some(owner);
        }
    } else if final_rules > out.rules.capacity() {
        acquire_frontend_vector::<AstRule>(scope, final_rules)?;
    }
    for constraint in &out.constraints {
        let authored_index = match scope.as_deref_mut() {
            Some(scope) => constraint.require_authored_index_with_allocation_owner(&mut |bytes| {
                scope.reserve_allocation(bytes)
            }),
            None => constraint.require_authored_index(),
        }?;
        let mut payload = custody
            .as_ref()
            .map(|_| crate::program_custody::PayloadCustody::default());
        if let Some(payload) = payload.as_mut() {
            let scope = scope
                .as_deref_mut()
                .expect("original Program custody scope");
            payload.reserve_unit(
                formatted_text_allocation_bytes(format_args!(
                    "__xlog_constraint_{authored_index}"
                ))?,
                scope,
            )?;
            payload.reserve_unit(std::mem::size_of::<Term>() as u64, scope)?;
            let mut body =
                crate::program_custody::PayloadCustody::for_body_clone(&constraint.body, scope)?;
            payload.append_from(&mut body, scope)?;
        } else {
            acquire_frontend_format(scope, format_args!("__xlog_constraint_{authored_index}"))?;
            acquire_frontend_vector::<Term>(scope, 1)?;
        }
        if payload.is_none() && scope.is_some() {
            acquire_frontend_vector::<BodyLiteral>(scope, constraint.body.len())?;
            for literal in &constraint.body {
                acquire_frontend(scope, literal.owned_allocation_bytes()?)?;
            }
        }
        let pred = format!("__xlog_constraint_{authored_index}");
        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: vec![Term::Integer(1)],
            },
            body: constraint.body.clone(),
        });
        if let Some(payload) = payload {
            custody
                .as_deref_mut()
                .expect("original Program custody")
                .rules
                .elements
                .push(payload);
        }
    }
    for (index, Query { atom }) in out.queries.iter().enumerate() {
        let mut payload = custody
            .as_ref()
            .map(|_| crate::program_custody::PayloadCustody::default());
        if let Some(payload) = payload.as_mut() {
            payload.reserve_unit(
                formatted_text_allocation_bytes(format_args!("__xlog_query_{index}"))?,
                scope
                    .as_deref_mut()
                    .expect("original Program custody scope"),
            )?;
        } else {
            acquire_frontend_format(scope, format_args!("__xlog_query_{index}"))?;
        }
        let pred = format!("__xlog_query_{index}");
        let mut head_terms_owner = None;
        let mut head_terms = Vec::new();
        let mut seen_owner = None;
        let mut seen = HashSet::new();
        for term in &atom.terms {
            let _variables_owner = if let Some(scope) = scope.as_deref_mut() {
                let mut count = Some(0usize);
                term.visit_variables(&mut |_| count = count.and_then(|n| n.checked_add(1)));
                let count =
                    count.ok_or_else(|| XlogError::allocation_extent_overflow("query variable"))?;
                if count == 0 {
                    None
                } else {
                    Some(scope.reserve_allocation(push_vector_allocation_bytes::<&str>(count)?)?)
                }
            } else {
                None
            };
            for name in term.variables() {
                if seen.contains(name) {
                    continue;
                }
                let next_seen_owner =
                    if seen.len() == seen.capacity() {
                        match scope.as_deref_mut() {
                            Some(scope) => Some(scope.reserve_allocation(
                                hash_allocation_bytes::<&str, ()>(seen.len() + 1)?,
                            )?),
                            None => None,
                        }
                    } else {
                        None
                    };
                let next_head_terms_owner = if head_terms.len() == head_terms.capacity() {
                    if payload.is_some() {
                        Some(
                            scope
                                .as_deref_mut()
                                .expect("original Program custody scope")
                                .reserve_allocation(
                                    xlog_core::push_vector_storage_bytes::<Term>(
                                        head_terms.len() + 1,
                                    )?,
                                )?,
                        )
                    } else {
                        acquire_frontend_vector::<Term>(scope, head_terms.len() + 1)?;
                        None
                    }
                } else {
                    None
                };
                if let Some(payload) = payload.as_mut() {
                    payload.reserve_unit(
                        name.len() as u64,
                        scope
                            .as_deref_mut()
                            .expect("original Program custody scope"),
                    )?;
                } else {
                    acquire_frontend(scope, name.len() as u64)?;
                }
                seen.insert(name);
                if let Some(owner) = next_seen_owner {
                    seen_owner = Some(owner);
                }
                head_terms.push(Term::Variable(name.to_string()));
                if let Some(owner) = next_head_terms_owner {
                    head_terms_owner = Some(owner);
                }
            }
        }
        drop(seen);
        drop(seen_owner);
        if head_terms.is_empty() {
            if payload.is_some() {
                head_terms_owner = Some(
                    scope
                        .as_deref_mut()
                        .expect("original Program custody scope")
                        .reserve_allocation(xlog_core::push_vector_storage_bytes::<Term>(1)?)?,
                );
            } else {
                acquire_frontend_vector::<Term>(scope, 1)?;
            }
            head_terms.push(Term::Integer(1));
        }
        if let Some(payload) = payload.as_mut() {
            let scope = scope
                .as_deref_mut()
                .expect("original Program custody scope");
            if let Some(owner) = head_terms_owner.as_ref() {
                payload.retain_original(owner, scope)?;
            }
            payload.reserve_unit(std::mem::size_of::<BodyLiteral>() as u64, scope)?;
            let mut atom = crate::program_custody::PayloadCustody::for_atom_clone(atom, scope)?;
            payload.append_from(&mut atom, scope)?;
        } else {
            acquire_frontend_vector::<BodyLiteral>(scope, 1)?;
        }
        if payload.is_none() && scope.is_some() {
            acquire_frontend(scope, atom.owned_allocation_bytes()?)?;
        }
        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: head_terms,
            },
            body: vec![BodyLiteral::Positive(atom.clone())],
        });
        if let Some(payload) = payload {
            custody
                .as_deref_mut()
                .expect("original Program custody")
                .rules
                .elements
                .push(payload);
        }
    }
    Ok(out)
}

fn validate_negation_safety_with_owner(
    program: &Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    for rule in &program.rules {
        acquire_frontend_format(scope, format_args!("rule {}", rule.head.predicate))?;
        validate_body_naf_safety_with_owner(
            &rule.body,
            &format!("rule {}", rule.head.predicate),
            scope,
        )?;
    }
    for constraint in &program.constraints {
        let index = match scope.as_deref_mut() {
            Some(scope) => constraint.require_authored_index_with_allocation_owner(&mut |bytes| {
                scope.reserve_allocation(bytes)
            }),
            None => constraint.require_authored_index(),
        }?;
        acquire_frontend_format(scope, format_args!("constraint {index}"))?;
        validate_body_naf_safety_with_owner(
            &constraint.body,
            &format!("constraint {index}"),
            scope,
        )?;
    }
    for (index, rule) in program.learnable_rules.iter().enumerate() {
        acquire_frontend_format(scope, format_args!("learnable rule {index}"))?;
        validate_body_naf_safety_with_owner(&rule.body, &format!("learnable rule {index}"), scope)?;
    }
    Ok(())
}

fn validate_body_naf_safety_with_owner(
    body: &[BodyLiteral],
    context: &str,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    match scope.as_deref_mut() {
        Some(scope) => {
            let mut stage = scope.child_scope();
            let result = validate_body_naf_safety_prepared(body, context, &mut Some(&mut stage));
            finish_compiler_stage(result, stage)
        }
        None => validate_body_naf_safety(body, context),
    }
}

fn validate_body_naf_safety_prepared(
    body: &[BodyLiteral],
    context: &str,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    if scope.is_some() {
        let mut count = Some(0usize);
        for literal in body {
            acquire_frontend_variables(scope, literal)?;
            match literal {
                BodyLiteral::Positive(atom) => {
                    atom.visit_variables(&mut |_| count = count.and_then(|n| n.checked_add(1)))
                }
                BodyLiteral::IsExpr(_) => count = count.and_then(|n| n.checked_add(1)),
                _ => {}
            }
        }
        let count =
            count.ok_or_else(|| XlogError::allocation_extent_overflow("negation variable"))?;
        acquire_frontend(scope, hash_allocation_bytes::<String, ()>(count)?)?;
    }
    validate_body_naf_safety_core(body, context, scope)
}

fn validate_body_naf_safety(body: &[BodyLiteral], context: &str) -> Result<()> {
    validate_body_naf_safety_core(body, context, &mut None)
}

fn validate_body_naf_safety_core(
    body: &[BodyLiteral],
    context: &str,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    let mut bound: HashSet<String> = HashSet::new();
    for lit in body {
        match lit {
            BodyLiteral::Positive(atom) => {
                for name in atom.variables() {
                    bound.insert(name.to_string());
                }
            }
            BodyLiteral::Negated(atom) => {
                for name in atom.variables() {
                    if !bound.contains(name) {
                        return Err(naf_error_with_owner(format_args!(
                            "unbound variable {} in negated atom {}/{} in {}; bind it before not with a positive atom or deterministic is expression, or use '_' for existential positions",
                            name,
                            atom.predicate,
                            atom.arity(),
                            context
                        ), scope));
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                bound.insert(is_expr.target.clone());
            }
            BodyLiteral::Epistemic(_) => {}
            BodyLiteral::Comparison(_) | BodyLiteral::Univ(_) => {}
        }
    }
    Ok(())
}

fn map_stratification_to_naf_error(err: XlogError) -> XlogError {
    match err {
        XlogError::StratificationCycle(cycle) => naf_error(format!(
            "deterministic not atom must be stratified; cycle through negation or aggregation: {}",
            cycle.join(" -> ")
        )),
        other => other,
    }
}

fn map_stratification_to_naf_error_with_owner(
    error: XlogError,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> XlogError {
    if scope.is_none() {
        return map_stratification_to_naf_error(error);
    }
    let XlogError::StratificationCycle(cycle) = error.unowned() else {
        return error;
    };
    let bytes = cycle
        .iter()
        .try_fold(0u64, |bytes, name| bytes.checked_add(name.len() as u64))
        .and_then(|bytes| {
            bytes.checked_add((cycle.len().saturating_sub(1) as u64).checked_mul(4)?)
        });
    let receipt = match bytes
        .ok_or_else(|| XlogError::allocation_extent_overflow("negation cycle"))
        .and_then(|bytes| {
            scope
                .as_deref_mut()
                .expect("original cycle scope")
                .reserve_allocation(bytes)
        }) {
        Ok(receipt) => receipt,
        Err(error) => return error,
    };
    let joined = cycle.join(" -> ");
    let mapped = naf_error_with_owner(
        format_args!(
            "deterministic not atom must be stratified; cycle through negation or aggregation: {}",
            joined
        ),
        scope,
    );
    drop(joined);
    drop(receipt);
    mapped
}

fn naf_error_with_owner(
    message: std::fmt::Arguments<'_>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> XlogError {
    let _message_receipt = match scope.as_deref_mut() {
        Some(scope) => match formatted_text_allocation_bytes(message)
            .and_then(|bytes| scope.reserve_allocation(bytes))
        {
            Ok(receipt) => Some(receipt),
            Err(error) => return error,
        },
        None => None,
    };
    let bytes = formatted_text_allocation_bytes(format_args!("negation safety error: {}", message));
    let producer = match scope.as_deref_mut() {
        Some(scope) => CompilationAllocationScope::prepare_diagnostic(
            bytes,
            Some(&mut |bytes| scope.reserve_allocation(bytes)),
        ),
        None => CompilationAllocationScope::prepare_diagnostic(bytes, None),
    };
    match producer {
        Ok(producer) => producer.finish(|| naf_error(std::fmt::format(message))),
        Err(error) => error,
    }
}

fn frontend_compilation_error(
    arguments: std::fmt::Arguments<'_>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> XlogError {
    let bytes = formatted_text_allocation_bytes(arguments);
    let producer = match scope.as_deref_mut() {
        Some(scope) => CompilationAllocationScope::prepare_diagnostic(
            bytes,
            Some(&mut |bytes| scope.reserve_allocation(bytes)),
        ),
        None => CompilationAllocationScope::prepare_diagnostic(bytes, None),
    };
    match producer {
        Ok(producer) => producer.finish(|| XlogError::Compilation(std::fmt::format(arguments))),
        Err(error) => error,
    }
}

fn naf_error(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(format!("negation safety error: {}", message.into()))
}

/// Convenience function to compile source in one call.
///
/// This creates a short-lived compiler and compiles the source.
/// For multiple compilations, prefer creating a `Compiler` instance directly.
/// Function definitions and calls must already be expanded; execution-facing
/// `LogicProgram` APIs own source normalization and import resolution.
///
/// # Example
///
/// ```ignore
/// use xlog_logic::compile::compile;
///
/// let plan = compile("edge(1, 2). reach(X, Y) :- edge(X, Y).")?;
/// ```
pub fn compile(source: &str) -> Result<ExecutionPlan> {
    let mut compiler = Compiler::new();
    compiler.compile(source)
}

/// Load modules for an entry source file.
///
/// This function:
/// 1. Loads the entry module from the exact supplied path
/// 2. Loads all direct and transitive `.xlog` module dependencies
///
/// # Arguments
///
/// * `entry_file` - Path to the entry source file
/// * `search_paths` - Additional directories to search for modules
///
/// # Returns
///
/// The loaded module resolver with all dependencies resolved, or an error
/// if module resolution fails.
pub fn load_modules(
    entry_file: &Path,
    search_paths: Vec<PathBuf>,
) -> std::result::Result<ModuleResolver, ModuleError> {
    let mut resolver = ModuleResolver::new(search_paths);
    resolver.load_entry_file(entry_file)?;

    Ok(resolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;
    use xlog_ir::RirNode;
    use xlog_stats::ColumnStats;
    use xlog_stats::RelationStats;
    use xlog_stats::StatsManager;

    #[test]
    fn test_compiler_new() {
        let compiler = Compiler::new();
        // Just verify it can be created
        drop(compiler);
    }

    #[test]
    fn test_compile_fact() {
        let mut compiler = Compiler::new();
        let result = compiler.compile("edge(1, 2).");
        assert!(result.is_ok(), "Failed to compile fact: {:?}", result.err());
    }

    #[test]
    fn test_compile_simple_rule() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile simple rule: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty(), "Expected at least one SCC");
    }

    #[test]
    fn compiler_records_each_program_query_at_its_compiled_rule_position() {
        let plan = Compiler::new()
            .compile(
                r#"
                    source(a).
                    ?- source(X).
                    ?- source(a).
                "#,
            )
            .expect("queries should compile");

        assert_eq!(plan.generated_query_rules.len(), 2);
        for (query_index, provenance) in plan.generated_query_rules.iter().enumerate() {
            assert_eq!(provenance.query_index, query_index);
            let rule = &plan.rules_by_scc[provenance.scc_index][provenance.rule_index];
            assert_eq!(rule.head, format!("__xlog_query_{query_index}"));
        }
    }

    #[test]
    fn compiler_does_not_infer_query_provenance_from_an_authored_prefix() {
        let mut program = parse_program(
            r#"
                source(a).
                authored(X) :- source(X).
            "#,
        )
        .expect("authored program should parse");
        program
            .rules
            .iter_mut()
            .find(|rule| rule.head.predicate == "authored")
            .expect("authored rule should exist")
            .head
            .predicate = "__xlog_query_authored".to_string();

        let plan = Compiler::new()
            .compile_program(&program)
            .expect("authored prefix should remain an ordinary rule");
        assert!(plan.generated_query_rules.is_empty());
    }

    #[test]
    fn compiler_rejects_an_authored_exact_generated_query_head_collision() {
        let mut program = parse_program(
            r#"
                source(a).
                authored(X) :- source(X).
                ?- source(X).
            "#,
        )
        .expect("collision witness should parse");
        program
            .rules
            .iter_mut()
            .find(|rule| rule.head.predicate == "authored")
            .expect("authored rule should exist")
            .head
            .predicate = "__xlog_query_0".to_string();

        let error = Compiler::new()
            .compile_program(&program)
            .expect_err("authored exact query-head collisions must fail compilation");
        assert!(
            error
                .to_string()
                .contains("authored relation __xlog_query_0 collides with generated query head"),
            "unexpected collision error: {error}"
        );
    }

    #[test]
    fn compiler_reports_bound_is_target_before_arithmetic_type_mismatch() {
        let error = Compiler::new()
            .compile(
                r#"
                val(10).
                bad(Z) :- val(Z), Z is Z + 1.
                ?- bad(Z).
                "#,
            )
            .expect_err("an already-bound is-expression target must be rejected");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable Z already bound; 'is' requires fresh variable"
        );
    }

    #[test]
    fn compiler_rejects_is_target_bound_by_an_earlier_is_expression() {
        let error = Compiler::new()
            .compile(
                r#"
                val(10).
                bad(Z) :- val(X), Z is X + X, Z is X + X.
                ?- bad(Z).
                "#,
            )
            .expect_err("a prior is-expression target must stay bound");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable Z already bound; 'is' requires fresh variable"
        );
    }

    #[test]
    fn compiler_accepts_fresh_chained_is_targets() {
        Compiler::new()
            .compile(
                r#"
                val(10).
                good(Z) :- val(X), Y is X + X, Z is Y + Y.
                ?- good(Z).
                "#,
            )
            .expect("distinct source-ordered is-expression targets must remain valid");
    }

    #[test]
    fn compiler_reports_unbound_is_input_before_a_later_duplicate_target() {
        let error = Compiler::new()
            .compile(
                r#"
                seed(1).
                bad(Z) :- seed(X), Z is U + 1, Z is 1 + 1.
                ?- bad(Z).
                "#,
            )
            .expect_err("an unreachable later binding must not preempt the first is error");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable U used in arithmetic but not bound"
        );
    }

    #[test]
    fn test_compile_transitive_closure() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
        "#,
        );
        assert!(result.is_ok(), "Failed to compile TC: {:?}", result.err());

        let plan = result.unwrap();
        // Should have SCCs for edge and reach
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn test_compile_with_negation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            node(1).
            node(2).
            node(3).
            edge(1, 2).
            isolated(X) :- node(X), not edge(X, _).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with negation: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compile_with_comparison() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            value(1).
            value(5).
            value(10).
            value(15).
            small(X) :- value(X), X < 10.
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with comparison: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_schema_infers_from_rule_body_types() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            path(X, Y) :- reach(X, Y).
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile rule for schema inference: {:?}",
            result.err()
        );

        let schema = compiler
            .schemas()
            .get("reach")
            .expect("missing reach schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::U32),
            "reach column 0 should match edge column type"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U32),
            "reach column 1 should match edge column type"
        );

        let schema = compiler.schemas().get("path").expect("missing path schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::U32),
            "path column 0 should inherit the transitive body type"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U32),
            "path column 1 should inherit the transitive body type"
        );
    }

    #[test]
    fn test_schema_propagates_known_columns_when_a_sibling_column_is_unresolved() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            path(X) :- intermediate(X, Z).
            intermediate(X, Z) :- source(X), unresolved(Z).
            unresolved(Z) :- unresolved(Z).
            source(value).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile partial rule-head schema inference: {:?}",
            result.err()
        );

        let schema = compiler.schemas().get("path").expect("missing path schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "path should inherit the known intermediate column independently of its sibling"
        );
    }

    #[test]
    fn test_schema_rejects_conflicting_rule_head_evidence_in_either_order() {
        let symbol_first = r#"
            pred symbols(symbol).
            pred integers(i64).
            symbols(value).
            integers(5000000000).
            mixed(X) :- symbols(X).
            mixed(X) :- integers(X).
        "#;
        let integer_first = r#"
            pred symbols(symbol).
            pred integers(i64).
            symbols(value).
            integers(5000000000).
            mixed(X) :- integers(X).
            mixed(X) :- symbols(X).
        "#;

        for source in [symbol_first, integer_first] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting inferred rule-head types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("column 1"), "{message}");
        }
    }

    #[test]
    fn test_schema_rejects_conflicting_body_evidence_in_either_order() {
        let first_body_order = r#"
            source_number(1).
            source_symbol(value).
            mixed(X) :- source_number(X), source_symbol(X).
            mixed(1) :- source_number(1).
        "#;
        let reversed_body_order = r#"
            source_number(1).
            source_symbol(value).
            mixed(X) :- source_symbol(X), source_number(X).
            mixed(1) :- source_number(1).
        "#;

        for source in [first_body_order, reversed_body_order] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting body-column types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("variable X"), "{message}");
        }
    }

    #[test]
    fn test_schema_infers_arithmetic_binding_result_type() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                computed(X) :- X is cast(1, u64).
            "#,
            )
            .expect("an arithmetic binding should determine its head-column type");

        let schema = compiler
            .schemas()
            .get("computed")
            .expect("missing computed schema");
        assert_eq!(schema.column_type(0), Some(ScalarType::U64));
    }

    #[test]
    fn test_schema_rejects_arithmetic_and_atom_evidence_in_either_rule_order() {
        let arithmetic_first = r#"
            pred integers(i64).
            integers(5000000000).
            mixed(X) :- X is cast(1, u64).
            mixed(X) :- integers(X).
        "#;
        let atom_first = r#"
            pred integers(i64).
            integers(5000000000).
            mixed(X) :- integers(X).
            mixed(X) :- X is cast(1, u64).
        "#;

        for source in [arithmetic_first, atom_first] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting arithmetic and atom types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("column 1"), "{message}");
        }
    }

    #[test]
    fn test_schema_rejects_incompatible_constant_heads_before_runtime() {
        let sources = [
            r#"
                pred p(u32).
                seed().
                p(value) :- seed().
            "#,
            r#"
                p(1).
                p(value).
            "#,
            r#"
                p(value).
                p(1).
            "#,
            r#"
                p(1).
                p(5000000000).
            "#,
            r#"
                p(5000000000).
                p(1).
            "#,
        ];

        for source in sources {
            let error = Compiler::new()
                .compile(source)
                .expect_err("incompatible constant head types must fail during compilation");
            let message = error.to_string();
            assert!(message.contains("p"), "{message}");
            assert!(message.contains("head term at position 0"), "{message}");
        }
    }

    #[test]
    fn test_declared_scalar_schemas_accept_supported_literal_conversions() {
        Compiler::new()
            .compile(
                r#"
                pred signed(i64).
                pred wide(u64).
                pred real(f64).
                pred flag(bool).
                seed().
                signed(1).
                wide(1).
                real(1).
                real(2) :- seed().
                flag(true).
                flag(false) :- seed().
            "#,
            )
            .expect("representable literals should use declared scalar storage types");
    }

    #[test]
    fn test_compile_unstratifiable_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            p :- not q.
            q :- not p.
        "#,
        );
        assert!(result.is_err(), "Should fail with stratification cycle");
    }

    #[test]
    fn test_compile_syntax_error_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile("edge(1, 2"); // Missing closing paren and period
        assert!(result.is_err(), "Should fail with syntax error");
    }

    #[test]
    fn test_compile_convenience_function() {
        let result = compile("edge(1, 2).");
        assert!(
            result.is_ok(),
            "Convenience compile failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compiler_reset() {
        let mut compiler = Compiler::new();

        // First compilation
        let result1 = compiler.compile("edge(1, 2).");
        assert!(result1.is_ok());

        // Reset and compile again
        compiler.reset();
        let result2 = compiler.compile("node(1). node(2).");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_compile_with_pred_decl() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            pred edge(u32, u32).
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with pred decl: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compile_multi_stratum() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            // Base facts
            edge(1, 2).
            edge(2, 3).
            edge(3, 1).

            // Stratum 0: edge (base)
            // Stratum 1: reach (depends on edge, recursive)
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).

            // Stratum 2: non_reach (negates reach)
            all_pairs(X, Y) :- edge(X, Z), edge(Y, W).
            non_reach(X, Y) :- all_pairs(X, Y), not reach(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile multi-stratum: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // Should have multiple strata
        assert!(!plan.strata.is_empty(), "Expected multiple strata");
    }

    #[test]
    fn test_compile_aggregation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            edge(1, 3).
            edge(2, 3).
            out_degree(X, count(Y)) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with aggregation: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        let out_degree_rules: Vec<_> = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|r| r.head == "out_degree")
            .collect();
        assert_eq!(out_degree_rules.len(), 1, "Expected one out_degree rule");

        // Aggregation lowering should produce a GroupBy node (wrapped in a Project to match head order).
        let body = &out_degree_rules[0].body;
        match body {
            RirNode::Project { input, .. } => {
                assert!(
                    matches!(input.as_ref(), RirNode::GroupBy { .. }),
                    "Expected Project(GroupBy(..)), got {:?}",
                    input
                );
            }
            other => panic!("Expected Project(GroupBy(..)), got {:?}", other),
        }
    }

    #[test]
    fn test_aggregate_schemas_match_runtime_result_types() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                pred edge(u32, u64).
                pred score(u32, f64).
                pred counted(u32, u64).
                pred summed(u32, u64).
                pred minimum(u32, u64).
                pred maximum(u32, u64).
                pred combined(u32, f64).
                edge(1, 5000000000).
                score(1, 2.5).
                counted(X, count(Y)) :- edge(X, Y).
                summed(X, sum(Y)) :- edge(X, Y).
                minimum(X, min(Y)) :- edge(X, Y).
                maximum(X, max(Y)) :- edge(X, Y).
                combined(X, logsumexp(Y)) :- score(X, Y).
            "#,
            )
            .expect("aggregate declarations should match provider result schemas");

        for predicate in ["counted", "summed", "minimum", "maximum"] {
            assert_eq!(
                compiler
                    .schemas()
                    .get(predicate)
                    .expect("missing aggregate schema")
                    .column_type(1),
                Some(ScalarType::U64),
                "unexpected result type for {predicate}"
            );
        }
        assert_eq!(
            compiler
                .schemas()
                .get("combined")
                .expect("missing log-sum-exp schema")
                .column_type(1),
            Some(ScalarType::F64)
        );
    }

    #[test]
    fn test_undeclared_count_schema_is_u64() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                edge(1, 5000000000).
                degree(X, count(Y)) :- edge(X, Y).
            "#,
            )
            .expect("undeclared count result should compile with its runtime type");
        assert_eq!(
            compiler
                .schemas()
                .get("degree")
                .expect("missing inferred count schema")
                .column_type(1),
            Some(ScalarType::U64)
        );
    }

    #[test]
    fn test_unsupported_aggregate_inputs_are_rejected_before_runtime() {
        let fixtures = [
            r#"
                pred source(u32, i32).
                pred result(u32, u64).
                source(1, 2).
                result(X, sum(Y)) :- source(X, Y).
            "#,
            r#"
                pred source(u32, u32).
                pred result(u32, f64).
                source(1, 2).
                result(X, logsumexp(Y)) :- source(X, Y).
            "#,
        ];

        for source in fixtures {
            let error = Compiler::new()
                .compile(source)
                .expect_err("unsupported aggregate input must fail during compilation");
            let message = error.to_string();
            assert!(message.contains("Unsupported aggregate input"), "{message}");
            assert!(message.contains("execution provider requires"), "{message}");
        }
    }

    #[test]
    fn test_compile_with_stats_snapshot() {
        let mut compiler = Compiler::new();
        let source = r#"
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#;

        let _ = compiler.compile(source).expect("Initial compile failed");
        let edge_id = *compiler.rel_ids().get("edge").expect("edge rel_id missing");

        let mut mgr = StatsManager::new();
        mgr.register_relation(edge_id);
        mgr.update_cardinality(edge_id, 42);
        let snapshot = mgr.snapshot();

        let plan = compiler
            .compile_with_stats_snapshot(source, Some(&snapshot))
            .expect("Compile with snapshot failed");
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn test_compile_with_named_stats_snapshot_reorders_joins() {
        let mut compiler = Compiler::new();
        let source = r#"
            foo(1).
            edge(1).
            out(X) :- edge(X), foo(X).
        "#;

        // Snapshot uses different RelIds than the compiler will assign for this program.
        // Map: RelId(0) -> edge (small), RelId(1) -> foo (big)
        let mut edge_stats = RelationStats::new(RelId(0));
        edge_stats.update_cardinality(10);
        let mut foo_stats = RelationStats::new(RelId(1));
        foo_stats.update_cardinality(10_000);

        let snapshot = StatsSnapshot {
            relations: vec![edge_stats, foo_stats],
            join_selectivities: Vec::new(),
            rel_names: vec![
                (RelId(0), "edge".to_string()),
                (RelId(1), "foo".to_string()),
            ],
        };

        let plan = compiler
            .compile_with_stats_snapshot(source, Some(&snapshot))
            .expect("Compile with named snapshot failed");

        let foo_id = *compiler.rel_ids().get("foo").expect("foo rel_id missing");
        let edge_id = *compiler.rel_ids().get("edge").expect("edge rel_id missing");

        let out_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|r| r.head == "out")
            .expect("out rule missing");

        // Peel projections to reach the join.
        let mut node = &out_rule.body;
        while let RirNode::Project { input, .. } = node {
            node = input;
        }

        match node {
            RirNode::ChainJoin {
                left,
                right,
                fallback,
                ..
            } => {
                // ChainJoin promotion wraps eligible two-atom joins after
                // stats-aware ordering. The chain node and its captured
                // fallback must agree on the build-side choice.
                assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));

                let mut fallback_node = fallback.as_ref();
                while let RirNode::Project { input, .. } = fallback_node {
                    fallback_node = input;
                }
                match fallback_node {
                    RirNode::Join { left, right, .. } => {
                        assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                        assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));
                    }
                    other => panic!("Expected ChainJoin fallback Join node, got {:?}", other),
                }
            }
            RirNode::Join { left, right, .. } => {
                // Prefer building on the smaller relation (right/build side).
                assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));
            }
            other => panic!("Expected Join node, got {:?}", other),
        }
    }

    fn helper_split_source() -> &'static str {
        r#"
            ab(0, 0). bc(0, 0). cd(0, 0). de(0, 0). ef(0, 0). af(0, 0).
            out(A, B, C, D, F) :-
                ab(A, B),
                bc(B, C),
                cd(C, D),
                de(D, E),
                ef(E, F),
                af(A, F).
        "#
    }

    fn helper_split_snapshot(distinct_d: u64) -> StatsSnapshot {
        let mut snapshot_relations = Vec::new();
        for (idx, name) in ["ab", "bc", "cd", "de", "ef", "af"].iter().enumerate() {
            let mut rel_stats = RelationStats::new(RelId(idx as u32));
            rel_stats.update_cardinality(8192);
            if *name == "de" {
                let mut d_col = ColumnStats::new(0, ScalarType::U32);
                d_col.update_distinct(distinct_d);
                rel_stats.add_column(d_col);
            }
            snapshot_relations.push(rel_stats);
        }
        StatsSnapshot {
            relations: snapshot_relations,
            join_selectivities: Vec::new(),
            rel_names: ["ab", "bc", "cd", "de", "ef", "af"]
                .iter()
                .enumerate()
                .map(|(idx, name)| (RelId(idx as u32), (*name).to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_compile_with_named_stats_snapshot_creates_helper_relation() {
        let mut compiler = Compiler::new();
        let snapshot = helper_split_snapshot(1);
        let plan = compiler
            .compile_with_stats_snapshot(helper_split_source(), Some(&snapshot))
            .expect("compile with helper stats");
        let helper = compiler
            .rel_ids()
            .iter()
            .find_map(|(name, rel)| {
                name.starts_with("__kclique_helper_")
                    .then_some((name.clone(), *rel))
            })
            .expect("helper relation allocated");

        let helper_rule_count = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|rule| rule.head == helper.0)
            .count();
        assert_eq!(helper_rule_count, 1);

        let helper_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == helper.0)
            .expect("helper rule");
        assert!(
            matches!(helper_rule.body, RirNode::ChainJoin { .. }),
            "helper split output should be eligible for ChainJoin promotion"
        );

        let out_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == "out")
            .expect("out rule");
        assert!(contains_scan(&out_rule.body, helper.1));
    }

    #[test]
    fn test_compile_with_flat_named_stats_keeps_original_rule() {
        let mut compiler = Compiler::new();
        let snapshot = helper_split_snapshot(8192);
        let plan = compiler
            .compile_with_stats_snapshot(helper_split_source(), Some(&snapshot))
            .expect("compile with flat stats");

        assert!(!compiler
            .rel_ids()
            .keys()
            .any(|name| name.starts_with("__kclique_helper_")));
        let out_rules = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|rule| rule.head == "out")
            .count();
        assert_eq!(out_rules, 1);
    }

    fn contains_scan(node: &RirNode, rel: RelId) -> bool {
        match node {
            RirNode::Scan { rel: scan_rel } => *scan_rel == rel,
            RirNode::Join { left, right, .. } | RirNode::ChainJoin { left, right, .. } => {
                contains_scan(left, rel) || contains_scan(right, rel)
            }
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => contains_scan(input, rel),
            RirNode::Union { inputs } => inputs.iter().any(|input| contains_scan(input, rel)),
            RirNode::Diff { left, right } => contains_scan(left, rel) || contains_scan(right, rel),
            RirNode::Fixpoint {
                base, recursive, ..
            } => contains_scan(base, rel) || contains_scan(recursive, rel),
            RirNode::MultiWayJoin { inputs, .. } => {
                inputs.iter().any(|input| contains_scan(input, rel))
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                rel_index.iter().any(|(input_rel, _)| *input_rel == rel)
            }
            RirNode::Unit => false,
        }
    }
}
