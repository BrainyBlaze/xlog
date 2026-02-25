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

// =============================================================================
// Learnable Rule Tests (ILP)
// =============================================================================

#[test]
fn test_parse_learnable_rule() {
    let input = include_str!("logic/learnable.xlog");
    let result = parse_program(input);
    assert!(
        result.is_ok(),
        "Failed to parse learnable program: {:?}",
        result.err()
    );

    let program = result.unwrap();
    assert_eq!(program.learnable_rules.len(), 1);

    let lr = &program.learnable_rules[0];
    assert_eq!(lr.mask_name, "W_mask");
    assert_eq!(lr.head.predicate, "reach");
    assert_eq!(lr.head.terms.len(), 2);
    assert_eq!(lr.body.len(), 2);
}

#[test]
fn test_parse_learnable_rule_preserves_normal_rules() {
    let input = include_str!("logic/learnable.xlog");
    let program = parse_program(input).unwrap();

    // 3 facts (edge) are in program.rules
    assert_eq!(program.facts().count(), 3);
    // The learnable rule is NOT in program.rules
    assert_eq!(program.proper_rules().count(), 0);
    // It's in learnable_rules
    assert_eq!(program.learnable_rules.len(), 1);
    // Query still parsed
    assert_eq!(program.queries.len(), 1);
}

#[test]
fn test_stratify_with_learnable_rule() {
    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let program = parse_program(input).unwrap();
    let result = stratify(&program);
    assert!(
        result.is_ok(),
        "Stratification failed: {:?}",
        result.err()
    );
    // Verify the learnable rule's head predicate appears in the strata
    let strata = result.unwrap();
    let all_preds: Vec<&str> = strata
        .iter()
        .flat_map(|s| s.predicates.iter().map(|p| p.as_str()))
        .collect();
    assert!(
        all_preds.contains(&"reach"),
        "Learnable rule head 'reach' should appear in strata, got: {:?}",
        all_preds
    );
    assert!(
        all_preds.contains(&"b1"),
        "Learnable rule body 'b1' should appear in strata, got: {:?}",
        all_preds
    );
    assert!(
        all_preds.contains(&"b2"),
        "Learnable rule body 'b2' should appear in strata, got: {:?}",
        all_preds
    );
}

#[test]
fn test_compile_learnable_rule_produces_tmj() {
    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(
        result.is_ok(),
        "Compilation failed: {:?}",
        result.err()
    );

    // Verify we can find a TensorMaskedJoin in the plan
    let plan = result.unwrap();
    let has_tmj = plan.rules_by_scc.iter().flatten().any(|rule| {
        matches!(rule.body, xlog_ir::rir::RirNode::TensorMaskedJoin { .. })
    });
    assert!(has_tmj, "Expected TensorMaskedJoin in compiled plan");
}

#[test]
fn test_learnable_rule_body_validation() {
    // Body must have exactly 2 positive atoms
    let input = r#"
        learnable(W) :: h(X) :- b1(X, Z).
    "#;
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(result.is_err(), "Should reject single-body learnable rule");
}

// =============================================================================
// M1 Gap Tests — Syntax & IR
// =============================================================================

// T1.2: Parse failure on malformed learnable rule
#[test]
fn test_parse_learnable_malformed_fails() {
    // Missing mask name
    let input = "learnable() :: h(X) :- b1(X, Z), b2(Z, Y).";
    assert!(parse_program(input).is_err());

    // Missing :: separator
    let input2 = "learnable(W) h(X,Y) :- b1(X,Z), b2(Z,Y).";
    assert!(parse_program(input2).is_err());
}

// T1.4: referenced_relations() includes all rel_index entries
#[test]
fn test_tmj_referenced_relations_complete() {
    let input = r#"
        edge(1,2).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let plan = compiler.compile_program(&program).unwrap();

    for rule in plan.rules_by_scc.iter().flatten() {
        if let xlog_ir::rir::RirNode::TensorMaskedJoin { ref rel_index, .. } = rule.body {
            let collected = rule.body.referenced_relations();
            for (rel_id, _) in rel_index {
                assert!(
                    collected.contains(rel_id),
                    "referenced_relations missing RelId {:?}",
                    rel_id
                );
            }
            return;
        }
    }
    panic!("No TensorMaskedJoin found in compiled plan");
}

// T1.8: Optimizer handles TensorMaskedJoin without panic
#[test]
fn test_optimizer_handles_tmj() {
    let input = r#"
        edge(1,2).
        edge(2,3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
        ?- reach(1, N).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let result = compiler.compile_program(&program);
    assert!(result.is_ok(), "Optimizer should handle TensorMaskedJoin: {:?}", result.err());
}

// T2: Learnable head validation — unbound variable must fail
#[test]
fn test_learnable_head_unbound_variable_fails() {
    let input = r#"
        edge(1, 2).
        learnable(W) :: reach(X, Q) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(result.is_err(), "Should reject head variable Q not in body");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not found in body"), "Error should mention unbound var: {}", err);
}

// T2: Learnable head validation — constant in head must fail
#[test]
fn test_learnable_head_constant_fails() {
    let input = r#"
        edge(1, 2).
        learnable(W) :: reach(1, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let result = compiler.compile(input);
    assert!(result.is_err(), "Should reject constant in learnable head");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("only variables"), "Error should mention variables-only: {}", err);
}

// Note: Full execution tests require xlog-cuda and xlog-runtime
// which depend on CUDA hardware. These will be added in later tasks.
