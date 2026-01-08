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

use xlog_core::Result;
use xlog_ir::ExecutionPlan;

use crate::lower::Lowerer;
use crate::parser::parse_program;
use crate::stratify::stratify;

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
        // Phase 1: Parse source into AST
        let program = parse_program(source)?;

        // Phase 2: Stratify (analyze dependencies, detect cycles)
        let strata = stratify(&program)?;

        // Convert strata to the format expected by the lowerer
        let strata_preds: Vec<Vec<String>> = strata
            .into_iter()
            .map(|s| s.predicates)
            .collect();

        // Phase 3: Lower AST to execution plan
        self.lowerer.set_strata(strata_preds);
        self.lowerer.lower_program(&program)
    }

    /// Reset the compiler state for a fresh compilation.
    ///
    /// This creates a new lowerer, clearing any cached schemas or relation IDs
    /// from previous compilations.
    pub fn reset(&mut self) {
        self.lowerer = Lowerer::new();
    }
}

/// Convenience function to compile source in one call.
///
/// This creates a temporary compiler and compiles the source.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let result = compiler.compile(r#"
            edge(1, 2).
            reach(X, Y) :- edge(X, Y).
        "#);
        assert!(result.is_ok(), "Failed to compile simple rule: {:?}", result.err());

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty(), "Expected at least one SCC");
    }

    #[test]
    fn test_compile_transitive_closure() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
        "#);
        assert!(result.is_ok(), "Failed to compile TC: {:?}", result.err());

        let plan = result.unwrap();
        // Should have SCCs for edge and reach
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn test_compile_with_negation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
            node(1).
            node(2).
            node(3).
            edge(1, 2).
            isolated(X) :- node(X), not edge(X, Y).
        "#);
        assert!(result.is_ok(), "Failed to compile with negation: {:?}", result.err());
    }

    #[test]
    fn test_compile_with_comparison() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
            value(1).
            value(5).
            value(10).
            value(15).
            small(X) :- value(X), X < 10.
        "#);
        assert!(result.is_ok(), "Failed to compile with comparison: {:?}", result.err());
    }

    #[test]
    fn test_compile_unstratifiable_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
            p :- not q.
            q :- not p.
        "#);
        assert!(result.is_err(), "Should fail with stratification cycle");
    }

    #[test]
    fn test_compile_syntax_error_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile("edge(1, 2");  // Missing closing paren and period
        assert!(result.is_err(), "Should fail with syntax error");
    }

    #[test]
    fn test_compile_convenience_function() {
        let result = compile("edge(1, 2).");
        assert!(result.is_ok(), "Convenience compile failed: {:?}", result.err());
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
        let result = compiler.compile(r#"
            pred edge(u32, u32).
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#);
        assert!(result.is_ok(), "Failed to compile with pred decl: {:?}", result.err());
    }

    #[test]
    fn test_compile_multi_stratum() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
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
        "#);
        assert!(result.is_ok(), "Failed to compile multi-stratum: {:?}", result.err());

        let plan = result.unwrap();
        // Should have multiple strata
        assert!(!plan.strata.is_empty(), "Expected multiple strata");
    }

    #[test]
    fn test_compile_aggregation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(r#"
            edge(1, 2).
            edge(1, 3).
            edge(2, 3).
            out_degree(X, count(Y)) :- edge(X, Y).
        "#);
        assert!(result.is_ok(), "Failed to compile with aggregation: {:?}", result.err());
    }
}
