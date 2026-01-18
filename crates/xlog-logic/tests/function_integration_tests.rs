//! Integration tests for user-defined functions

use xlog_logic::parse_program as parse;
use xlog_logic::function::{FunctionRegistry, FunctionError};
use xlog_logic::expand::ExpansionContext;
use xlog_logic::ast::{ArithExpr, FuncDef, FuncParam, FuncBody, CondExpr, CompOp};

#[test]
fn test_full_function_pipeline() {
    let src = r#"
        func double(X) = X * 2.
        func quadruple(X) = double(double(X)).

        pred input(f64).
        input(5.0).

        pred output(f64).
        output(Y) :- input(X), Y is quadruple(X).

        ?- output(X).
    "#;

    let program = parse(src).unwrap();

    // Build function registry
    let registry = FunctionRegistry::from_program(&program).unwrap();

    // Verify both functions registered
    assert!(registry.contains("double"));
    assert!(registry.contains("quadruple"));

    // Verify quadruple is not recursive (calls double, not itself)
    assert!(!registry.is_recursive("double"));
    assert!(!registry.is_recursive("quadruple"));
}

#[test]
fn test_recursive_function_with_base_case() {
    let src = r#"
        func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).
    "#;

    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    assert!(registry.is_recursive("factorial"));

    // Should pass validation (has base case)
    assert!(registry.validate().is_ok());
}

#[test]
fn test_recursive_without_base_case_fails() {
    let src = r#"
        func bad(N) = bad(N - 1).
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(result.is_err());
    match result.unwrap_err() {
        FunctionError::RecursionWithoutBaseCase { name } => {
            assert_eq!(name, "bad");
        }
        e => panic!("Expected RecursionWithoutBaseCase, got {:?}", e),
    }
}

#[test]
fn test_function_name_conflict_with_predicate() {
    let src = r#"
        pred foo(u32).
        func foo(X) = X + 1.
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(result.is_err());
    match result.unwrap_err() {
        FunctionError::NameConflict { name } => {
            assert_eq!(name, "foo");
        }
        e => panic!("Expected NameConflict, got {:?}", e),
    }
}

#[test]
fn test_max_recursion_depth_exceeded() {
    let src = r#"
        func countdown(N) = if N <= 0 then 0 else countdown(N - 1).
    "#;

    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    // Expansion with low depth should fail
    let mut ctx = ExpansionContext::new(&registry, 10);
    let result = ctx.expand_call("countdown", &[ArithExpr::Integer(100)]);

    match result {
        Err(FunctionError::MaxRecursionDepth { name, depth }) => {
            assert_eq!(name, "countdown");
            assert_eq!(depth, 10);
        }
        _ => panic!("Expected MaxRecursionDepth error"),
    }
}

#[test]
fn test_simple_function_expansion() {
    let src = "func double(X) = X + X.";
    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    let mut ctx = ExpansionContext::new(&registry, 100);
    let result = ctx.expand_call("double", &[ArithExpr::Integer(5)]).unwrap();

    // Should expand to 5 + 5
    match result {
        ArithExpr::Add(l, r) => {
            assert!(matches!(*l, ArithExpr::Integer(5)));
            assert!(matches!(*r, ArithExpr::Integer(5)));
        }
        _ => panic!("Expected Add expression"),
    }
}

#[test]
fn test_nested_function_expansion() {
    let src = r#"
        func double(X) = X * 2.
        func quadruple(X) = double(double(X)).
    "#;
    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    let mut ctx = ExpansionContext::new(&registry, 100);
    let result = ctx.expand_call("quadruple", &[ArithExpr::Integer(3)]).unwrap();

    // quadruple(3) -> double(double(3)) -> double(3 * 2) -> (3 * 2) * 2
    // Result should be Mul(Mul(3, 2), 2)
    match result {
        ArithExpr::Mul(_, _) => {}  // Nested Mul is expected
        _ => panic!("Expected Mul expression, got {:?}", result),
    }
}

#[test]
fn test_duplicate_function_error() {
    let src = r#"
        func foo(X) = X + 1.
        func foo(Y) = Y * 2.
    "#;
    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(result.is_err());
    match result.unwrap_err() {
        FunctionError::DuplicateDefinition { name } => {
            assert_eq!(name, "foo");
        }
        e => panic!("Expected DuplicateDefinition, got {:?}", e),
    }
}

#[test]
fn test_private_function_not_exported() {
    // Test that private functions work but aren't exported
    let src = r#"
        private func helper(X) = X + 1.
        func public_fn(X) = helper(X) * 2.
    "#;
    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    // Both should be registered
    assert!(registry.contains("helper"));
    assert!(registry.contains("public_fn"));

    // Check visibility
    let helper = registry.get("helper").unwrap();
    let public_fn = registry.get("public_fn").unwrap();
    assert!(helper.is_private);
    assert!(!public_fn.is_private);
}

#[test]
fn test_function_with_type_annotations() {
    let src = "func dist(X: f64, Y: f64) -> f64 = pow(X * X + Y * Y, 0.5).";
    let program = parse(src).unwrap();
    let func = &program.functions[0];

    assert_eq!(func.params.len(), 2);
    assert!(func.params[0].typ.is_some());
    assert!(func.params[1].typ.is_some());
    assert!(func.return_type.is_some());
}

#[test]
fn test_recursion_warning_analysis() {
    let mut registry = FunctionRegistry::new();

    // func risky(N) = if N <= 0 then 1 else risky(N + 1)
    // This should produce a warning because N + 1 moves away from N <= 0
    let risky = FuncDef {
        name: "risky".to_string(),
        params: vec![FuncParam { name: "N".to_string(), typ: None }],
        return_type: None,
        body: FuncBody::Conditional(CondExpr {
            cond_left: ArithExpr::Variable("N".to_string()),
            cond_op: CompOp::Le,
            cond_right: ArithExpr::Integer(0),
            then_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Integer(1))),
            else_branch: Box::new(FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "risky".to_string(),
                args: vec![ArithExpr::Add(
                    Box::new(ArithExpr::Variable("N".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                )],
            })),
        }),
        is_private: false,
    };

    registry.register(risky).unwrap();
    let (result, warnings) = registry.validate_with_warnings();

    assert!(result.is_ok()); // It's valid (has base case)
    assert!(!warnings.is_empty(), "Expected warning for risky recursion");
    assert!(warnings[0].message.contains("increases"));
}
