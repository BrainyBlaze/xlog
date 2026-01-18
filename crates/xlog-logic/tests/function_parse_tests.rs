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
