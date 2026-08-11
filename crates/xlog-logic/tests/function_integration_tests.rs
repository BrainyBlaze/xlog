//! Integration tests for user-defined functions

use xlog_logic::ast::{ArithExpr, CompOp, CondExpr, FuncBody, FuncDef, FuncParam};
use xlog_logic::expand::ExpansionContext;
use xlog_logic::function::{FunctionError, FunctionRegistry};
use xlog_logic::parse_program as parse;
use xlog_logic::{expand_program_functions, Compiler};

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
fn test_predicate_function_expands_and_compiles() {
    let src = r#"
        pred parent(u32, u32).
        pred answer(u32).

        func get_parent(Child) = P :- parent(Child, P).

        parent(1, 2).
        answer(P) :- P is get_parent(1).
        ?- answer(P).
    "#;

    let program = parse(src).unwrap();
    let expanded = expand_program_functions(&program, 100).unwrap();
    let mut compiler = Compiler::new();

    compiler.compile_program(&expanded).unwrap();
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
    let error = result.unwrap_err();
    assert!(error.to_string().starts_with("error[E0502]:"), "{error}");
    match error {
        FunctionError::RecursionWithoutBaseCase { name } => {
            assert_eq!(name, "bad");
        }
        e => panic!("Expected RecursionWithoutBaseCase, got {:?}", e),
    }
}

#[test]
fn wholly_unguarded_recursive_component_reports_first_declaration() {
    let program = parse(
        r#"
        func first(Value) = second(Value).
        func second(Value) = first(Value).
        "#,
    )
    .unwrap();
    let error = FunctionRegistry::from_program(&program)
        .expect_err("unguarded recursive component must fail strict validation");

    assert!(
        matches!(&error, FunctionError::RecursionWithoutBaseCase { name } if name == "first"),
        "unexpected error: {error:?}"
    );
    assert!(error.to_string().starts_with("error[E0502]:"), "{error}");
}

#[test]
fn recursive_component_accepts_a_conditional_member() {
    let program = parse(
        r#"
        func first(Value) = if Value = 0 then 0 else second(Value).
        func second(Value) = first(Value).
        "#,
    )
    .unwrap();

    FunctionRegistry::from_program(&program)
        .expect("a recursive component with conditional base-case syntax is valid");
}

#[test]
fn test_predicate_body_calls_participate_in_recursion_validation() {
    let src = r#"
        func repeat(Value) = Result :- Result is repeat(Value).
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(matches!(
        result,
        Err(FunctionError::RecursionWithoutBaseCase { name }) if name == "repeat"
    ));
}

#[test]
fn test_predicate_body_calls_participate_in_undefined_function_validation() {
    let src = r#"
        func lookup(Value) = Result :- Result is missing(Value).
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    let error = result.expect_err("strict validation must reject an undefined call");
    assert!(matches!(
        &error,
        FunctionError::UndefinedFunction { name } if name == "missing"
    ));
    assert!(error.to_string().starts_with("error[E0503]:"), "{error}");
}

#[test]
fn strict_validation_reports_undefined_calls_in_source_order() {
    let program = parse(
        r#"
        func first(Value) = missing_first(Value) + missing_second(Value).
        func second(Value) = missing_third(Value).
        "#,
    )
    .unwrap();

    for _ in 0..32 {
        let error = FunctionRegistry::from_program(&program)
            .expect_err("strict validation must reject the first undefined callee");
        assert!(
            matches!(&error, FunctionError::UndefinedFunction { name } if name == "missing_first"),
            "unexpected error: {error:?}"
        );
        assert!(error.to_string().starts_with("error[E0503]:"), "{error}");
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
    let error = result.unwrap_err();
    assert!(error.to_string().starts_with("error[E0505]:"), "{error}");
    match error {
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
    let result = ctx
        .expand_call("quadruple", &[ArithExpr::Integer(3)])
        .unwrap();

    // quadruple(3) -> double(double(3)) -> double(3 * 2) -> (3 * 2) * 2
    // Result should be Mul(Mul(3, 2), 2)
    match result {
        ArithExpr::Mul(_, _) => {} // Nested Mul is expected
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
    let error = result.unwrap_err();
    assert!(error.to_string().starts_with("error[E0501]:"), "{error}");
    match error {
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
        params: vec![FuncParam {
            name: "N".to_string(),
            typ: None,
        }],
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
