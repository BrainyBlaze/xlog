//! Compilation pipeline for XLOG programs
//!
//! This module provides the main entry point for compiling XLOG source code
//! into execution plans. The compilation process consists of:
//!
//! 1. **Parsing**: Convert source text to AST (`parser::parse_program`)
//! 2. **Stratification**: Analyze negation/aggregation dependencies (`stratify::stratify`)
//! 3. **Lowering**: Transform AST to Relational IR (`lower::Lowerer::lower_program`)
//!
//! The `Compiler` struct orchestrates these phases and provides a single
//! entry point via the `compile` method.

use std::path::{Path, PathBuf};

use xlog_core::Result;
use xlog_ir::ExecutionPlan;
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::compiler_config::CompilerConfig;
use crate::lower::Lowerer;
use crate::module::ModuleError;
use crate::optimizer::Optimizer;
use crate::parser::parse_program;
use crate::resolver::ModuleResolver;
use crate::stratify::stratify;
use crate::{BodyLiteral, Program, Query, Rule as AstRule, Term};

/// The XLOG compiler orchestrates the full compilation pipeline.
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

use std::collections::HashMap;
use std::sync::Arc;
use xlog_core::{RelId, Schema};

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
    /// This is the main entry point for compilation. It chains together:
    /// 1. Parsing (source → AST)
    /// 2. Stratification (analyze dependencies, check for cycles)
    /// 3. Lowering (AST → Relational IR execution plan)
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
    /// W2.1: this entry point delegates through the new composable API
    /// with `CompilerConfig::default()`, which preserves slice
    /// 1/2/4/W2.2 behavior bit-identically.
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

    /// W2.1: composable entry point that accepts a `CompilerConfig`.
    ///
    /// Default-config callers should keep using `compile()` /
    /// `compile_with_stats_snapshot()`. This entry point exists so
    /// W2.1 can flip the variable-ordering cost model on per-call
    /// without an env override.
    pub fn compile_with_config_and_stats_snapshot(
        &mut self,
        source: &str,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        let program = parse_program(source)?;
        self.compile_program_with_config_and_stats_snapshot(&program, config, stats_snapshot)
    }

    /// Compile a parsed XLOG program into an execution plan.
    ///
    /// This is useful for callers that want to inspect the AST (facts, queries,
    /// constraints) while compiling without reparsing.
    pub fn compile_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.compile_program_with_stats_snapshot(program, None)
    }

    /// Compile a parsed XLOG program into an execution plan, optionally seeding the optimizer.
    ///
    /// W2.1: delegates to
    /// [`Self::compile_program_with_config_and_stats_snapshot`] with
    /// `CompilerConfig::default()`.
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

    /// W2.1: composable program-level entry point.
    ///
    /// `config` is currently consumed only by the promoter
    /// (W2.1 step 5) when it wires the variable-ordering cost
    /// model. With `CompilerConfig::default()`, the promoter
    /// behaves identically to pre-W2.1.
    pub fn compile_program_with_config_and_stats_snapshot(
        &mut self,
        program: &Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        let program = desugar_queries_and_constraints(program);

        // Phase 2: Stratify (analyze dependencies, detect cycles)
        let strata = stratify(&program)?;

        // Convert strata to the format expected by the lowerer
        let strata_preds: Vec<Vec<String>> = strata.into_iter().map(|s| s.predicates).collect();

        // Phase 3: Lower AST to execution plan
        self.lowerer.set_strata(strata_preds);

        // If we have predicate names for the snapshot, use them to seed lowering-time
        // join ordering with better cardinality estimates.
        let mut cardinality_hints: HashMap<String, u64> = HashMap::new();
        if let Some(snapshot) = stats_snapshot {
            if !snapshot.rel_names.is_empty() {
                let rel_name_by_id: HashMap<RelId, &str> = snapshot
                    .rel_names
                    .iter()
                    .map(|(id, name)| (*id, name.as_str()))
                    .collect();
                for rel in &snapshot.relations {
                    if let Some(name) = rel_name_by_id.get(&rel.rel_id) {
                        cardinality_hints.insert((*name).to_string(), rel.cardinality);
                    }
                }
            }
        }
        self.lowerer.set_cardinality_hints(cardinality_hints);

        let mut plan = self.lowerer.lower_program(&program)?;

        // Phase 4: Optimize (predicate pushdown + cost-aware rewrites)
        //
        // Seed statistics with any known fact cardinalities so cost estimation has
        // at least a baseline for EDB relations.
        let mut mgr = StatsManager::new();
        let mut fact_counts: HashMap<String, u64> = HashMap::new();
        for fact in program.facts() {
            *fact_counts.entry(fact.head.predicate.clone()).or_insert(0) += 1;
        }

        for (pred, rel_id) in self.lowerer.rel_ids() {
            mgr.register_relation(*rel_id);
            let rows = fact_counts.get(pred).copied().unwrap_or(0);
            if rows > 0 {
                mgr.update_cardinality(*rel_id, rows);
                if let Some(schema) = self.lowerer.schemas().get(pred) {
                    mgr.update_byte_size(*rel_id, rows * schema.row_size_bytes() as u64);
                }
            }
        }

        if let Some(snapshot) = stats_snapshot {
            if snapshot.rel_names.is_empty() {
                mgr.merge_snapshot(snapshot);
            } else {
                let rel_name_by_id: HashMap<RelId, &str> = snapshot
                    .rel_names
                    .iter()
                    .map(|(id, name)| (*id, name.as_str()))
                    .collect();

                for rel in &snapshot.relations {
                    let Some(pred) = rel_name_by_id.get(&rel.rel_id) else {
                        continue;
                    };
                    let Some(rel_id) = self.lowerer.rel_ids().get(*pred) else {
                        continue;
                    };

                    let mut remapped = rel.clone();
                    remapped.rel_id = *rel_id;

                    if let Some(schema) = self.lowerer.schemas().get(*pred) {
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
                    let Some(&left_id) = self.lowerer.rel_ids().get(*left_pred) else {
                        continue;
                    };
                    let Some(&right_id) = self.lowerer.rel_ids().get(*right_pred) else {
                        continue;
                    };

                    let Some(left_schema) = self.lowerer.schemas().get(*left_pred) else {
                        continue;
                    };
                    let Some(right_schema) = self.lowerer.schemas().get(*right_pred) else {
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

        // Build schemas by RelId for the optimizer
        let schemas_by_rel_id: HashMap<RelId, Schema> = self
            .lowerer
            .rel_ids()
            .iter()
            .filter_map(|(pred, rel_id)| {
                self.lowerer
                    .schemas()
                    .get(pred)
                    .map(|schema| (*rel_id, schema.clone()))
            })
            .collect();

        let stats_arc = Arc::new(mgr);

        crate::optimizer::helper_split_pass::run(
            &mut plan,
            &schemas_by_rel_id,
            &stats_arc,
            |schema| self.lowerer.create_helper_relation(schema),
        );

        let schemas_by_rel_id: HashMap<RelId, Schema> = self
            .lowerer
            .rel_ids()
            .iter()
            .filter_map(|(pred, rel_id)| {
                self.lowerer
                    .schemas()
                    .get(pred)
                    .map(|schema| (*rel_id, schema.clone()))
            })
            .collect();

        let mut optimizer = Optimizer::new(Arc::clone(&stats_arc));
        optimizer.set_schemas(schemas_by_rel_id);
        for rules in &mut plan.rules_by_scc {
            for rule in rules {
                rule.body = optimizer.optimize(rule.body.clone());
            }
        }

        // v0.6.5 slice 3: selectivity-aware reordering pass. Runs
        // BETWEEN the optimizer loop and promote_multiway.
        // Locked compile-pipeline ordering:
        //   lower → helper_split_pass → optimizer → selectivity_pass → promote_multiway
        //
        // v0.6.5 W2.2: takes `rel_ids` so per-body Scans can be
        // resolved against `StatsManager`. Behavior on empty
        // stats / unseeded relations is no-op (safety floor).
        crate::optimizer::selectivity_pass::run(&mut plan, &stats_arc, self.lowerer.rel_ids());

        // v0.6.5 slice 1: promote eligible triangle subtrees to
        // RirNode::MultiWayJoin. Runs *after* the optimizer so the
        // optimizer never has to learn the new variant. Fallback
        // identity preserves v0.6.2 binary-join semantics on
        // dispatch decline.
        //
        // v0.6.5 slice 4: pass the lowerer's predicate→RelId map
        // so the promoter can gate recursive-SCC bodies on the
        // count of in-SCC Scans (≤ 1 = promote, ≥ 2 = skip).
        //
        // W2.1: also pass `&stats_arc` and the caller-provided
        // `&CompilerConfig`. With `CompilerConfig::default()`
        // (`Disabled`), the promoter never sets `var_order` and
        // slice 1/2/4/W2.2 dispatch is bit-identical.
        crate::promote::promote_multiway(&mut plan, self.lowerer.rel_ids(), &stats_arc, config);

        let schemas_by_rel_id: HashMap<RelId, Schema> = self
            .lowerer
            .rel_ids()
            .iter()
            .filter_map(|(pred, rel_id)| {
                self.lowerer
                    .schemas()
                    .get(pred)
                    .map(|schema| (*rel_id, schema.clone()))
            })
            .collect();

        crate::optimizer::helper_split_pass::run_kclique_specs(
            &mut plan,
            &schemas_by_rel_id,
            |schema| self.lowerer.create_helper_relation(schema),
        );

        Ok(plan)
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

fn desugar_queries_and_constraints(program: &Program) -> Program {
    let mut out = program.clone();

    // Constraints: `:- body.` becomes `__xlog_constraint_i(1) :- body.`
    for (i, constraint) in program.constraints.iter().enumerate() {
        let pred = format!("__xlog_constraint_{}", i);
        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: vec![Term::Integer(1)],
            },
            body: constraint.body.clone(),
        });
    }

    // Queries: `?- atom.` becomes `__xlog_query_i(Vars...) :- atom.`
    for (i, Query { atom }) in program.queries.iter().enumerate() {
        let pred = format!("__xlog_query_{}", i);

        let mut head_terms: Vec<Term> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for term in &atom.terms {
            if let Term::Variable(name) = term {
                if seen.insert(name.as_str()) {
                    head_terms.push(Term::Variable(name.clone()));
                }
            }
        }

        if head_terms.is_empty() {
            head_terms.push(Term::Integer(1));
        }

        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: head_terms,
            },
            body: vec![BodyLiteral::Positive(atom.clone())],
        });
    }

    out
}

/// Convenience function to compile source in one call.
///
/// This creates a short-lived compiler and compiles the source.
/// For multiple compilations, prefer creating a `Compiler` instance directly.
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

/// Load and validate modules for a source file.
///
/// This function:
/// 1. Determines the module path from the entry file name
/// 2. Loads the entry module and all its dependencies
/// 3. Validates imports (checks for conflicts, private predicates, etc.)
///
/// # Arguments
///
/// * `entry_file` - Path to the main .xlog file
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

    // Determine base directory and module path
    let base_dir = entry_file.parent().unwrap_or(Path::new("."));
    let module_name = entry_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");

    // Load entry module (recursively loads dependencies)
    resolver.load_module(base_dir, &[module_name.to_string()])?;

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
            isolated(X) :- node(X), not edge(X, Y).
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
                // W63 wraps eligible two-atom joins after stats-aware
                // ordering. The chain node and its captured fallback must
                // agree on the build-side choice.
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
                name.starts_with("__w37_helper_")
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
            "helper split output should be eligible for W63 ChainJoin promotion"
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
            .any(|name| name.starts_with("__w37_helper_")));
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
