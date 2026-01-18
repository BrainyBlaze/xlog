//! Integration tests for xlog-logic
//!
//! Tests the full compilation pipeline from source to execution plan.

use xlog_logic::{compile, parse_program, stratify, Compiler};

// =============================================================================
// Parsing Tests
// =============================================================================

#[test]
fn test_parse_tc_program() {
    let input = include_str!("logic/tc.xlog");
    let result = parse_program(input);
    assert!(
        result.is_ok(),
        "Failed to parse TC program: {:?}",
        result.err()
    );

    let program = result.unwrap();
    assert_eq!(program.rules.len(), 5); // 3 facts + 2 rules
    assert_eq!(program.queries.len(), 1);
}

#[test]
fn test_parse_stratified_program() {
    let input = include_str!("logic/stratified.xlog");
    let result = parse_program(input);
    assert!(
        result.is_ok(),
        "Failed to parse stratified program: {:?}",
        result.err()
    );

    let program = result.unwrap();
    assert!(program.rules.iter().any(|r| r.has_negation()));
}

#[test]
fn test_parse_aggregate_program() {
    let input = include_str!("logic/aggregates.xlog");
    let result = parse_program(input);
    assert!(
        result.is_ok(),
        "Failed to parse aggregate program: {:?}",
        result.err()
    );

    let program = result.unwrap();
    assert!(program.rules.iter().any(|r| r.has_aggregation()));
}

// =============================================================================
// Stratification Tests
// =============================================================================

#[test]
fn test_stratify_tc_program() {
    let input = include_str!("logic/tc.xlog");
    let program = parse_program(input).expect("Parse failed");
    let result = stratify(&program);
    assert!(
        result.is_ok(),
        "Failed to stratify TC program: {:?}",
        result.err()
    );
}

#[test]
fn test_stratify_negation_program() {
    let input = include_str!("logic/stratified.xlog");
    let program = parse_program(input).expect("Parse failed");
    let result = stratify(&program);
    assert!(
        result.is_ok(),
        "Failed to stratify negation program: {:?}",
        result.err()
    );

    let strata = result.unwrap();
    // Should have at least 2 strata (base predicates + predicates with negation)
    assert!(
        strata.len() >= 2,
        "Expected at least 2 strata for negation program"
    );
}

// =============================================================================
// Full Compilation Tests
// =============================================================================

#[test]
fn test_compile_tc_program() {
    let input = include_str!("logic/tc.xlog");
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(
        result.is_ok(),
        "Failed to compile TC program: {:?}",
        result.err()
    );

    let plan = result.unwrap();
    assert!(!plan.sccs.is_empty(), "Expected SCCs in execution plan");
}

#[test]
fn test_compile_stratified_program() {
    let input = include_str!("logic/stratified.xlog");
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(
        result.is_ok(),
        "Failed to compile stratified program: {:?}",
        result.err()
    );

    let plan = result.unwrap();
    assert!(!plan.strata.is_empty(), "Expected strata in execution plan");
}

#[test]
fn test_compile_aggregate_program() {
    let input = include_str!("logic/aggregates.xlog");
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(
        result.is_ok(),
        "Failed to compile aggregate program: {:?}",
        result.err()
    );
}

#[test]
fn test_compile_desugars_query_to_rule() {
    let input = include_str!("logic/tc.xlog");
    let mut compiler = Compiler::new();
    let plan = compiler.compile(input).expect("Compile failed");

    assert!(
        compiler.schemas().contains_key("__xlog_query_0"),
        "Expected query predicate schema to be inferred"
    );

    let has_query_rule = plan
        .rules_by_scc
        .iter()
        .flatten()
        .any(|r| r.head == "__xlog_query_0");
    assert!(
        has_query_rule,
        "Expected a compiled rule for __xlog_query_0"
    );
}

#[test]
fn test_compile_desugars_constraint_to_rule() {
    let input = r#"
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        :- reach(X, X).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(input).expect("Compile failed");

    assert!(
        compiler.schemas().contains_key("__xlog_constraint_0"),
        "Expected constraint predicate schema to be inferred"
    );

    let has_constraint_rule = plan
        .rules_by_scc
        .iter()
        .flatten()
        .any(|r| r.head == "__xlog_constraint_0");
    assert!(
        has_constraint_rule,
        "Expected a compiled rule for __xlog_constraint_0"
    );
}

#[test]
fn test_compile_convenience_function() {
    let input = include_str!("logic/tc.xlog");
    let result = compile(input);
    assert!(
        result.is_ok(),
        "Convenience compile failed: {:?}",
        result.err()
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_compile_unstratifiable_program() {
    // p depends negatively on q, q depends negatively on p - cycle through negation
    let input = r#"
        p :- not q.
        q :- not p.
    "#;
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(result.is_err(), "Should fail with stratification cycle");
}

#[test]
fn test_compile_syntax_error() {
    let input = "edge(1, 2"; // Missing closing paren and period
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(result.is_err(), "Should fail with syntax error");
}

// Note: Full execution tests require xlog-cuda and xlog-runtime
// which depend on CUDA hardware. These will be added in later tasks.
