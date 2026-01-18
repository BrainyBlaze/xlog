use xlog_logic::ast::{ArithExpr, FuncBody};
use xlog_logic::parse_program as parse;

#[test]
fn test_parse_simple_func() {
    let src = "func square(X) = X * X.";
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.functions[0].name, "square");
    assert_eq!(program.functions[0].params.len(), 1);
    assert_eq!(program.functions[0].params[0].name, "X");
    assert!(!program.functions[0].is_private);
}

#[test]
fn test_parse_typed_func() {
    let src = "func square(X: f64) -> f64 = X * X.";
    let program = parse(src).unwrap();
    let func = &program.functions[0];
    assert!(func.params[0].typ.is_some());
    assert!(func.return_type.is_some());
}

#[test]
fn test_parse_multi_param_func() {
    let src = "func add(X, Y) = X + Y.";
    let program = parse(src).unwrap();
    assert_eq!(program.functions[0].params.len(), 2);
}

#[test]
fn test_parse_private_func() {
    let src = "private func helper(X) = X + 1.";
    let program = parse(src).unwrap();
    assert!(program.functions[0].is_private);
}

#[test]
fn test_parse_conditional_func() {
    let src = "func abs_val(X) = if X < 0 then 0 - X else X.";
    let program = parse(src).unwrap();
    match &program.functions[0].body {
        FuncBody::Conditional(cond) => {
            assert!(matches!(&cond.cond_left, ArithExpr::Variable(v) if v == "X"));
        }
        _ => panic!("expected conditional body"),
    }
}

#[test]
fn test_parse_predicate_func() {
    let src = "func get_parent(X) = P :- parent(X, P).";
    let program = parse(src).unwrap();
    match &program.functions[0].body {
        FuncBody::Predicate { result, body } => {
            assert_eq!(result, "P");
            assert!(!body.is_empty());
        }
        _ => panic!("expected predicate body"),
    }
}

#[test]
fn test_parse_recursive_func() {
    let src = "func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).";
    let program = parse(src).unwrap();
    assert_eq!(program.functions[0].name, "fib");
}

#[test]
fn test_parse_func_call_in_rule() {
    let src = r#"
        func square(X) = X * X.
        result(Y) :- input(X), Y is square(X).
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.rules.len(), 1);
}

#[test]
fn test_parse_max_recursion_pragma() {
    let src = r#"
        #pragma max_recursion_depth = 500
        func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.directives.max_recursion_depth, Some(500));
}

#[test]
fn test_parse_nested_func_calls() {
    let src = "func compose(X) = outer(inner(X)).";
    let program = parse(src).unwrap();

    assert_eq!(program.functions.len(), 1);
    let func = &program.functions[0];
    assert_eq!(func.name, "compose");

    // Body should be FuncCall containing another FuncCall
    match &func.body {
        FuncBody::Arithmetic(ArithExpr::FuncCall { name, args }) => {
            assert_eq!(name, "outer");
            assert_eq!(args.len(), 1);
            match &args[0] {
                ArithExpr::FuncCall { name: inner_name, .. } => {
                    assert_eq!(inner_name, "inner");
                }
                _ => panic!("Expected inner FuncCall"),
            }
        }
        _ => panic!("Expected FuncCall body"),
    }
}

#[test]
fn test_parse_deeply_nested() {
    let src = "func deep(X) = a(b(c(d(X)))).";
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
}

#[test]
fn test_parse_both_call_syntaxes() {
    let src = r#"
        func double(X) = X * 2.

        // Via is expression
        result1(Y) :- input(X), Y is double(X).

        // Function call in arithmetic expression via is
        check(X, R) :- input(X), R is double(X), R > 10.
    "#;
    let program = parse(src).unwrap();

    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.rules.len(), 2);
}

#[test]
fn test_parse_complex_conditional() {
    let src = "func clamp(X, Lo, Hi) = if X < Lo then Lo else if X > Hi then Hi else X.";
    let program = parse(src).unwrap();
    assert_eq!(program.functions[0].params.len(), 3);
    match &program.functions[0].body {
        FuncBody::Conditional(_) => {}
        _ => panic!("Expected conditional body"),
    }
}
