//! Tests that UDFs work correctly with the module system.
//!
//! These tests verify that:
//! 1. Functions are correctly parsed within module definitions
//! 2. Imported functions are recognized in use declarations
//! 3. Private functions are marked correctly
//! 4. Recursive functions parse correctly
//! 5. Function syntax including conditionals works properly

use serial_test::serial;
use xlog_logic::ast::{ArithExpr, FuncBody};
use xlog_logic::parser::parse_program;

#[test]
#[serial]
fn test_module_with_function_definitions() {
    // Module with arithmetic and conditional functions
    let math_module = r#"
        func abs(X) = if X < 0 then 0 - X else X.
        func clamp(X, Lo, Hi) = if X < Lo then Lo else if X > Hi then Hi else X.
    "#;

    let prog = parse_program(math_module).unwrap();

    // Verify functions were parsed
    assert_eq!(prog.functions.len(), 2);
    assert!(prog.functions.iter().any(|f| f.name == "abs"));
    assert!(prog.functions.iter().any(|f| f.name == "clamp"));

    // Verify abs has 1 parameter
    let abs_func = prog.functions.iter().find(|f| f.name == "abs").unwrap();
    assert_eq!(abs_func.params.len(), 1);
    assert_eq!(abs_func.params[0].name, "X");

    // Verify clamp has 3 parameters
    let clamp_func = prog.functions.iter().find(|f| f.name == "clamp").unwrap();
    assert_eq!(clamp_func.params.len(), 3);
    assert_eq!(clamp_func.params[0].name, "X");
    assert_eq!(clamp_func.params[1].name, "Lo");
    assert_eq!(clamp_func.params[2].name, "Hi");
}

#[test]
#[serial]
fn test_import_function_from_module() {
    // Main program importing functions from math module
    let main_src = r#"
        use math::{abs, clamp}.
        pred distance(f64, f64).
    "#;

    let prog = parse_program(main_src).unwrap();

    // Verify import is parsed
    assert_eq!(prog.imports.len(), 1);
    let import = &prog.imports[0];
    assert_eq!(import.module_path, vec!["math"]);

    let imports = import.imports.as_ref().expect("Expected specific imports");
    assert!(imports.contains(&"abs".to_string()));
    assert!(imports.contains(&"clamp".to_string()));
}

#[test]
#[serial]
fn test_import_all_from_module() {
    // Import all public members from a module
    let main_src = r#"
        use math.
        pred result(f64).
    "#;

    let prog = parse_program(main_src).unwrap();

    assert_eq!(prog.imports.len(), 1);
    let import = &prog.imports[0];
    assert_eq!(import.module_path, vec!["math"]);
    assert!(import.imports.is_none()); // None means import all
}

#[test]
#[serial]
fn test_private_function_in_module() {
    let util_module = r#"
        private func helper(X) = X * 2.
        func public_func(X) = helper(X) + 1.
    "#;

    let prog = parse_program(util_module).unwrap();

    assert_eq!(prog.functions.len(), 2);

    // Verify private function is marked
    let helper = prog.functions.iter().find(|f| f.name == "helper").unwrap();
    assert!(helper.is_private, "helper should be marked as private");

    // Verify public function is not marked private
    let public_func = prog
        .functions
        .iter()
        .find(|f| f.name == "public_func")
        .unwrap();
    assert!(
        !public_func.is_private,
        "public_func should not be marked as private"
    );
}

#[test]
#[serial]
fn test_recursive_function_in_module() {
    let math_module = r#"
        func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).
    "#;

    let prog = parse_program(math_module).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let factorial = prog
        .functions
        .iter()
        .find(|f| f.name == "factorial")
        .unwrap();
    assert_eq!(factorial.params.len(), 1);
    assert_eq!(factorial.params[0].name, "N");

    // Verify the body is a conditional (base case check)
    match &factorial.body {
        FuncBody::Conditional(cond) => {
            // Condition should be N <= 1
            assert!(matches!(&cond.cond_left, ArithExpr::Variable(v) if v == "N"));
        }
        _ => panic!("Expected conditional body for factorial"),
    }
}

#[test]
#[serial]
fn test_function_syntax_basic() {
    // Test basic function syntax: func name(params) = body.
    let src = "func square(X) = X * X.";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.name, "square");
    assert_eq!(func.params.len(), 1);
    assert!(!func.is_private);

    // Body should be X * X
    match &func.body {
        FuncBody::Arithmetic(ArithExpr::Mul(left, right)) => {
            assert!(matches!(left.as_ref(), ArithExpr::Variable(v) if v == "X"));
            assert!(matches!(right.as_ref(), ArithExpr::Variable(v) if v == "X"));
        }
        _ => panic!("Expected multiplication expression"),
    }
}

#[test]
#[serial]
fn test_function_syntax_multiple_params() {
    let src = "func add3(A, B, C) = A + B + C.";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.params.len(), 3);
    assert_eq!(func.params[0].name, "A");
    assert_eq!(func.params[1].name, "B");
    assert_eq!(func.params[2].name, "C");
}

#[test]
#[serial]
fn test_conditional_function_if_then_else() {
    let src = "func abs_val(X) = if X < 0 then 0 - X else X.";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.name, "abs_val");

    match &func.body {
        FuncBody::Conditional(cond) => {
            // Condition: X < 0
            assert!(matches!(&cond.cond_left, ArithExpr::Variable(v) if v == "X"));
            // The right side should be 0 (either Integer or Float)
            match &cond.cond_right {
                ArithExpr::Integer(0) => {}
                ArithExpr::Float(f) if *f == 0.0 => {}
                other => panic!("Expected 0 in condition right, got {:?}", other),
            }
        }
        _ => panic!("Expected conditional body"),
    }
}

#[test]
#[serial]
fn test_nested_conditional_function() {
    // Test nested if-then-else (clamp function)
    let src = "func clamp(X, Lo, Hi) = if X < Lo then Lo else if X > Hi then Hi else X.";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.name, "clamp");
    assert_eq!(func.params.len(), 3);

    // Should be a conditional with nested conditional in else branch
    match &func.body {
        FuncBody::Conditional(outer) => {
            // Outer condition: X < Lo
            assert!(matches!(&outer.cond_left, ArithExpr::Variable(v) if v == "X"));
            assert!(matches!(&outer.cond_right, ArithExpr::Variable(v) if v == "Lo"));

            // Then branch should be Lo
            match outer.then_branch.as_ref() {
                FuncBody::Arithmetic(ArithExpr::Variable(v)) => {
                    assert_eq!(v, "Lo");
                }
                _ => panic!("Expected Lo in then branch"),
            }

            // Else branch should be another conditional
            match outer.else_branch.as_ref() {
                FuncBody::Conditional(inner) => {
                    // Inner condition: X > Hi
                    assert!(matches!(&inner.cond_left, ArithExpr::Variable(v) if v == "X"));
                    assert!(matches!(&inner.cond_right, ArithExpr::Variable(v) if v == "Hi"));
                }
                _ => panic!("Expected nested conditional in else branch"),
            }
        }
        _ => panic!("Expected conditional body"),
    }
}

#[test]
#[serial]
fn test_function_with_typed_params() {
    let src = "func dist(X: f64, Y: f64) -> f64 = pow(X * X + Y * Y, 0.5).";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.name, "dist");

    // Verify typed parameters
    assert!(func.params[0].typ.is_some());
    assert!(func.params[1].typ.is_some());

    // Verify return type
    assert!(func.return_type.is_some());
}

#[test]
#[serial]
fn test_module_with_mixed_declarations() {
    // Module with both predicates and functions
    let module_src = r#"
        pred input(f64).
        pred output(f64).

        func double(X) = X * 2.
        func triple(X) = X * 3.

        input(5.0).
        output(Y) :- input(X), Y is double(X).
    "#;

    let prog = parse_program(module_src).unwrap();

    // Verify predicates
    assert_eq!(prog.predicates.len(), 2);

    // Verify functions
    assert_eq!(prog.functions.len(), 2);
    assert!(prog.functions.iter().any(|f| f.name == "double"));
    assert!(prog.functions.iter().any(|f| f.name == "triple"));

    // Verify rules (including facts)
    assert_eq!(prog.rules.len(), 2); // 1 fact + 1 rule
}

#[test]
#[serial]
fn test_function_with_nested_calls() {
    let src = "func quadruple(X) = double(double(X)).";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];

    // Body should be FuncCall containing another FuncCall
    match &func.body {
        FuncBody::Arithmetic(ArithExpr::FuncCall { name, args }) => {
            assert_eq!(name, "double");
            assert_eq!(args.len(), 1);

            // Inner call should also be double
            match &args[0] {
                ArithExpr::FuncCall {
                    name: inner_name, ..
                } => {
                    assert_eq!(inner_name, "double");
                }
                _ => panic!("Expected inner FuncCall"),
            }
        }
        _ => panic!("Expected FuncCall body"),
    }
}

#[test]
#[serial]
fn test_import_from_nested_module_path() {
    let src = r#"
        use utils/math::{abs, clamp}.
        use deep/nested/module.
    "#;

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.imports.len(), 2);

    // First import: utils/math with specific items
    let import1 = &prog.imports[0];
    assert_eq!(import1.module_path, vec!["utils", "math"]);
    assert!(import1.imports.is_some());

    // Second import: deep/nested/module (all)
    let import2 = &prog.imports[1];
    assert_eq!(import2.module_path, vec!["deep", "nested", "module"]);
    assert!(import2.imports.is_none());
}

#[test]
#[serial]
fn test_multiple_private_functions() {
    let src = r#"
        private func helper1(X) = X + 1.
        private func helper2(X) = X * 2.
        func exposed(X) = helper1(helper2(X)).
    "#;

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 3);

    let helper1 = prog.functions.iter().find(|f| f.name == "helper1").unwrap();
    let helper2 = prog.functions.iter().find(|f| f.name == "helper2").unwrap();
    let exposed = prog.functions.iter().find(|f| f.name == "exposed").unwrap();

    assert!(helper1.is_private);
    assert!(helper2.is_private);
    assert!(!exposed.is_private);
}

#[test]
#[serial]
fn test_fibonacci_recursive_function() {
    let src = "func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let fib = &prog.functions[0];
    assert_eq!(fib.name, "fib");
    assert_eq!(fib.params.len(), 1);

    // Should be conditional with recursive calls in else branch
    match &fib.body {
        FuncBody::Conditional(cond) => {
            // Base case: N <= 1 then N
            match cond.then_branch.as_ref() {
                FuncBody::Arithmetic(ArithExpr::Variable(v)) => {
                    assert_eq!(v, "N");
                }
                _ => panic!("Expected N in then branch"),
            }

            // Recursive case: fib(N-1) + fib(N-2)
            match cond.else_branch.as_ref() {
                FuncBody::Arithmetic(ArithExpr::Add(_, _)) => {
                    // The recursive structure is correct
                }
                _ => panic!("Expected Add expression in else branch"),
            }
        }
        _ => panic!("Expected conditional body"),
    }
}

#[test]
#[serial]
fn test_function_used_in_rule_body() {
    let src = r#"
        func square(X) = X * X.

        pred input(f64).
        pred output(f64).

        input(3.0).
        output(Y) :- input(X), Y is square(X).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify function is parsed
    assert_eq!(prog.functions.len(), 1);

    // Verify rule uses is expression
    let rules: Vec<_> = prog.proper_rules().collect();
    assert_eq!(rules.len(), 1);

    // The rule should have a body with input(X) and Y is square(X)
    assert_eq!(rules[0].body.len(), 2);
}

#[test]
#[serial]
fn test_predicate_based_function() {
    let src = "func get_parent(X) = P :- parent(X, P).";

    let prog = parse_program(src).unwrap();

    assert_eq!(prog.functions.len(), 1);
    let func = &prog.functions[0];
    assert_eq!(func.name, "get_parent");

    match &func.body {
        FuncBody::Predicate { result, body } => {
            assert_eq!(result, "P");
            assert!(!body.is_empty());
        }
        _ => panic!("Expected predicate body"),
    }
}

#[test]
#[serial]
fn test_max_recursion_pragma_with_function() {
    let src = r#"
        #pragma max_recursion_depth = 500
        func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify pragma is parsed
    assert_eq!(prog.directives.max_recursion_depth, Some(500));

    // Verify function is parsed
    assert_eq!(prog.functions.len(), 1);
    assert_eq!(prog.functions[0].name, "factorial");
}

#[test]
#[serial]
fn test_module_imports_with_functions_and_predicates() {
    let src = r#"
        use graph::{edge, reach}.
        use math::{abs, clamp}.

        pred distance(u32, u32, f64).

        func safe_div(X, Y) = if Y = 0 then 0 else X / Y.
    "#;

    let prog = parse_program(src).unwrap();

    // Verify imports
    assert_eq!(prog.imports.len(), 2);

    // Verify predicate declaration
    assert_eq!(prog.predicates.len(), 1);

    // Verify function
    assert_eq!(prog.functions.len(), 1);
    assert_eq!(prog.functions[0].name, "safe_div");
}
