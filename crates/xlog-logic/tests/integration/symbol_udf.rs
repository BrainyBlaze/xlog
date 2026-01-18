//! Tests that symbols work correctly with UDFs.
//!
//! These tests verify that:
//! 1. Predicates with symbol columns work with functions
//! 2. Symbols are correctly interned in function-related contexts
//! 3. Function output used with symbol predicates
//!
//! All tests use `serial_test::serial` since they manipulate global symbol state.

use serial_test::serial;
use xlog_core::symbol;
use xlog_logic::parser::parse_program;

#[test]
#[serial]
fn test_symbol_predicate_with_function_in_body() {
    symbol::clear();

    let src = r#"
        pred label(u32, symbol).
        label(1, low).
        label(2, medium).
        label(3, high).

        func double(X) = X * 2.

        pred result(u32, symbol).
        result(D, L) :- label(N, L), D is double(N).

        ?- result(X, Y).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols are interned
    let low = symbol::intern("low");
    let medium = symbol::intern("medium");
    let high = symbol::intern("high");

    assert_eq!(symbol::resolve(low), "low");
    assert_eq!(symbol::resolve(medium), "medium");
    assert_eq!(symbol::resolve(high), "high");

    // Verify function exists
    assert!(prog.functions.iter().any(|f| f.name == "double"));

    // Verify predicates exist
    assert!(prog.predicates.iter().any(|p| p.name == "label"));
    assert!(prog.predicates.iter().any(|p| p.name == "result"));

    // Verify there's a proper rule (not just facts)
    assert!(prog.proper_rules().count() > 0);

    // Verify there's a query
    assert!(!prog.queries.is_empty());
}

#[test]
#[serial]
fn test_symbol_with_conditional_function() {
    symbol::clear();

    let src = r#"
        pred task(symbol, f64).
        task(urgent, 9.0).
        task(normal, 5.0).
        task(low, 2.0).

        func prioritize(Priority) = if Priority > 7.0 then 3 else if Priority > 4.0 then 2 else 1.

        pred categorized(symbol, f64, f64).
        categorized(Label, Score, Category) :-
            task(Label, Score),
            Category is prioritize(Score).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("urgent")), "urgent");
    assert_eq!(symbol::resolve(symbol::intern("normal")), "normal");
    assert_eq!(symbol::resolve(symbol::intern("low")), "low");

    // Verify function with nested conditionals exists
    let prioritize = prog
        .functions
        .iter()
        .find(|f| f.name == "prioritize")
        .expect("prioritize function should exist");
    assert_eq!(prioritize.params.len(), 1);

    // Verify the rule exists
    let rules: Vec<_> = prog.proper_rules().collect();
    assert_eq!(rules.len(), 1);
}

#[test]
#[serial]
fn test_symbol_in_function_output_predicate() {
    symbol::clear();

    let src = r#"
        pred data(symbol, f64).
        data(sensor_a, -5.0).
        data(sensor_b, 3.0).

        func abs(X) = if X < 0 then 0 - X else X.

        pred absolute_reading(symbol, f64).
        absolute_reading(Label, AbsVal) :- data(Label, Val), AbsVal is abs(Val).

        ?- absolute_reading(X, Y).
    "#;

    let prog = parse_program(src).unwrap();

    // Symbols interned
    assert_eq!(symbol::resolve(symbol::intern("sensor_a")), "sensor_a");
    assert_eq!(symbol::resolve(symbol::intern("sensor_b")), "sensor_b");

    // Verify function exists
    assert!(prog.functions.iter().any(|f| f.name == "abs"));

    // Verify query exists
    assert!(!prog.queries.is_empty());
}

#[test]
#[serial]
fn test_symbol_with_multiple_functions() {
    symbol::clear();

    let src = r#"
        pred measurement(symbol, f64).
        measurement(temp_sensor, 25.5).
        measurement(humidity_sensor, 65.0).
        measurement(pressure_sensor, 1013.25).

        func normalize(X, Min, Max) = (X - Min) / (Max - Min).
        func clamp(X, Lo, Hi) = if X < Lo then Lo else if X > Hi then Hi else X.

        pred normalized_reading(symbol, f64).
        normalized_reading(Label, Norm) :-
            measurement(Label, Val),
            Raw is normalize(Val, 0, 100),
            Norm is clamp(Raw, 0, 1).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify all symbols are interned
    let temp = symbol::intern("temp_sensor");
    let humidity = symbol::intern("humidity_sensor");
    let pressure = symbol::intern("pressure_sensor");

    assert_eq!(symbol::resolve(temp), "temp_sensor");
    assert_eq!(symbol::resolve(humidity), "humidity_sensor");
    assert_eq!(symbol::resolve(pressure), "pressure_sensor");

    // Verify both functions exist
    assert!(prog.functions.iter().any(|f| f.name == "normalize"));
    assert!(prog.functions.iter().any(|f| f.name == "clamp"));

    // Verify normalize has 3 params
    let normalize_func = prog
        .functions
        .iter()
        .find(|f| f.name == "normalize")
        .unwrap();
    assert_eq!(normalize_func.params.len(), 3);

    // Verify clamp has 3 params
    let clamp_func = prog.functions.iter().find(|f| f.name == "clamp").unwrap();
    assert_eq!(clamp_func.params.len(), 3);
}

#[test]
#[serial]
fn test_symbol_with_recursive_function() {
    symbol::clear();

    let src = r#"
        pred item(symbol, u32).
        item(product_a, 5).
        item(product_b, 3).
        item(product_c, 7).

        func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).

        pred item_factorial(symbol, u32, u32).
        item_factorial(Name, Count, Fact) :- item(Name, Count), Fact is factorial(Count).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("product_a")), "product_a");
    assert_eq!(symbol::resolve(symbol::intern("product_b")), "product_b");
    assert_eq!(symbol::resolve(symbol::intern("product_c")), "product_c");

    // Verify recursive function exists
    let factorial = prog
        .functions
        .iter()
        .find(|f| f.name == "factorial")
        .expect("factorial function should exist");
    assert_eq!(factorial.params.len(), 1);
}

#[test]
#[serial]
fn test_symbol_with_private_function() {
    symbol::clear();

    let src = r#"
        pred status(symbol, f64).
        status(online, 1.0).
        status(offline, 0.0).
        status(degraded, 0.5).

        private func internal_score(X) = X * 100.
        func public_score(X) = internal_score(X) + 10.

        pred scored_status(symbol, f64).
        scored_status(Label, Score) :- status(Label, Val), Score is public_score(Val).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("online")), "online");
    assert_eq!(symbol::resolve(symbol::intern("offline")), "offline");
    assert_eq!(symbol::resolve(symbol::intern("degraded")), "degraded");

    // Verify functions
    let internal_score = prog
        .functions
        .iter()
        .find(|f| f.name == "internal_score")
        .unwrap();
    assert!(internal_score.is_private);

    let public_score = prog
        .functions
        .iter()
        .find(|f| f.name == "public_score")
        .unwrap();
    assert!(!public_score.is_private);
}

#[test]
#[serial]
fn test_symbol_with_function_in_multi_column_predicate() {
    symbol::clear();

    let src = r#"
        pred employee(symbol, symbol, f64).
        employee(alice, engineering, 75000.0).
        employee(bob, sales, 65000.0).
        employee(carol, engineering, 85000.0).

        func bonus(Salary) = Salary * 0.1.

        pred employee_bonus(symbol, symbol, f64, f64).
        employee_bonus(Name, Dept, Salary, Bonus) :-
            employee(Name, Dept, Salary),
            Bonus is bonus(Salary).

        ?- employee_bonus(N, D, S, B).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify all name symbols
    assert_eq!(symbol::resolve(symbol::intern("alice")), "alice");
    assert_eq!(symbol::resolve(symbol::intern("bob")), "bob");
    assert_eq!(symbol::resolve(symbol::intern("carol")), "carol");

    // Verify department symbols
    assert_eq!(
        symbol::resolve(symbol::intern("engineering")),
        "engineering"
    );
    assert_eq!(symbol::resolve(symbol::intern("sales")), "sales");

    // Verify facts count
    let facts: Vec<_> = prog.facts().collect();
    assert_eq!(facts.len(), 3);

    // Verify query
    assert!(!prog.queries.is_empty());
}

#[test]
#[serial]
fn test_symbol_with_typed_function() {
    symbol::clear();

    let src = r#"
        pred reading(symbol, f64).
        reading(sensor_x, 42.5).
        reading(sensor_y, -10.0).

        func scale(X: f64) -> f64 = X * 1.5.

        pred scaled_reading(symbol, f64).
        scaled_reading(Label, Scaled) :- reading(Label, Val), Scaled is scale(Val).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("sensor_x")), "sensor_x");
    assert_eq!(symbol::resolve(symbol::intern("sensor_y")), "sensor_y");

    // Verify typed function
    let scale_func = prog.functions.iter().find(|f| f.name == "scale").unwrap();
    assert!(scale_func.params[0].typ.is_some());
    assert!(scale_func.return_type.is_some());
}

#[test]
#[serial]
fn test_symbol_preservation_across_function_chain() {
    symbol::clear();

    let src = r#"
        pred source(symbol, f64).
        source(alpha, 10.0).
        source(beta, 20.0).
        source(gamma, 30.0).

        func step1(X) = X + 1.
        func step2(X) = X * 2.
        func step3(X) = X - 5.

        pred transformed(symbol, f64).
        transformed(Label, Result) :-
            source(Label, V0),
            V1 is step1(V0),
            V2 is step2(V1),
            Result is step3(V2).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols preserved correctly
    let alpha = symbol::intern("alpha");
    let beta = symbol::intern("beta");
    let gamma = symbol::intern("gamma");

    assert_eq!(symbol::resolve(alpha), "alpha");
    assert_eq!(symbol::resolve(beta), "beta");
    assert_eq!(symbol::resolve(gamma), "gamma");

    // Verify all three functions exist
    assert_eq!(prog.functions.len(), 3);
    assert!(prog.functions.iter().any(|f| f.name == "step1"));
    assert!(prog.functions.iter().any(|f| f.name == "step2"));
    assert!(prog.functions.iter().any(|f| f.name == "step3"));

    // Verify the rule has all four predicates in body (source + 3 is expressions)
    let rules: Vec<_> = prog.proper_rules().collect();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].body.len(), 4);
}

#[test]
#[serial]
fn test_symbol_with_function_comparison() {
    symbol::clear();

    let src = r#"
        pred score(symbol, f64).
        score(team_red, 85.0).
        score(team_blue, 90.0).
        score(team_green, 75.0).

        func threshold(X) = if X >= 80.0 then 1 else 0.

        pred qualified(symbol).
        qualified(Team) :- score(Team, S), T is threshold(S), T = 1.
    "#;

    let prog = parse_program(src).unwrap();

    // Verify team symbols
    assert_eq!(symbol::resolve(symbol::intern("team_red")), "team_red");
    assert_eq!(symbol::resolve(symbol::intern("team_blue")), "team_blue");
    assert_eq!(symbol::resolve(symbol::intern("team_green")), "team_green");

    // Verify the threshold function
    assert!(prog.functions.iter().any(|f| f.name == "threshold"));

    // Verify qualified predicate takes symbol
    let qualified_pred = prog
        .predicates
        .iter()
        .find(|p| p.name == "qualified")
        .unwrap();
    assert_eq!(qualified_pred.types.len(), 1);
}

#[test]
#[serial]
fn test_symbol_in_query_with_function() {
    symbol::clear();

    let src = r#"
        pred value(symbol, f64).
        value(x, 5.0).
        value(y, 10.0).

        func square(N) = N * N.

        pred squared(symbol, f64).
        squared(Label, Sq) :- value(Label, V), Sq is square(V).

        ?- squared(x, S).
        ?- squared(y, S).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("x")), "x");
    assert_eq!(symbol::resolve(symbol::intern("y")), "y");

    // Verify we have 2 queries
    assert_eq!(prog.queries.len(), 2);

    // Verify function exists
    assert!(prog.functions.iter().any(|f| f.name == "square"));
}

#[test]
#[serial]
fn test_symbol_interning_order_with_functions() {
    symbol::clear();

    let src = r#"
        func first(X) = X + 1.

        pred tag(symbol).
        tag(aaa).
        tag(bbb).
        tag(ccc).

        func second(X) = X * 2.

        pred more_tags(symbol).
        more_tags(ddd).
        more_tags(eee).
    "#;

    let prog = parse_program(src).unwrap();

    // Symbols should be interned in parse order
    let aaa = symbol::intern("aaa");
    let bbb = symbol::intern("bbb");
    let ccc = symbol::intern("ccc");
    let ddd = symbol::intern("ddd");
    let eee = symbol::intern("eee");

    // All should resolve correctly
    assert_eq!(symbol::resolve(aaa), "aaa");
    assert_eq!(symbol::resolve(bbb), "bbb");
    assert_eq!(symbol::resolve(ccc), "ccc");
    assert_eq!(symbol::resolve(ddd), "ddd");
    assert_eq!(symbol::resolve(eee), "eee");

    // Verify both functions exist
    assert!(prog.functions.iter().any(|f| f.name == "first"));
    assert!(prog.functions.iter().any(|f| f.name == "second"));
}

#[test]
#[serial]
fn test_symbol_with_predicate_based_function() {
    symbol::clear();

    let src = r#"
        pred parent(symbol, symbol).
        parent(alice, bob).
        parent(bob, carol).

        func get_parent(Child) = P :- parent(Child, P).

        pred has_grandparent(symbol).
        has_grandparent(X) :- parent(X, P), parent(P, _).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("alice")), "alice");
    assert_eq!(symbol::resolve(symbol::intern("bob")), "bob");
    assert_eq!(symbol::resolve(symbol::intern("carol")), "carol");

    // Verify predicate-based function
    let get_parent = prog
        .functions
        .iter()
        .find(|f| f.name == "get_parent")
        .expect("get_parent function should exist");

    use xlog_logic::ast::FuncBody;
    match &get_parent.body {
        FuncBody::Predicate { result, body } => {
            assert_eq!(result, "P");
            assert!(!body.is_empty());
        }
        _ => panic!("Expected predicate body for get_parent"),
    }
}

#[test]
#[serial]
fn test_symbol_with_imported_function_declaration() {
    symbol::clear();

    let src = r#"
        use math::{abs, clamp}.

        pred sensor(symbol, f64).
        sensor(temp, -5.5).
        sensor(humidity, 75.0).

        pred adjusted(symbol, f64).
        adjusted(Name, Val) :- sensor(Name, Raw), Val is abs(Raw).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("temp")), "temp");
    assert_eq!(symbol::resolve(symbol::intern("humidity")), "humidity");

    // Verify import
    assert_eq!(prog.imports.len(), 1);
    let import = &prog.imports[0];
    assert_eq!(import.module_path, vec!["math"]);

    let imports = import.imports.as_ref().expect("Expected specific imports");
    assert!(imports.contains(&"abs".to_string()));
    assert!(imports.contains(&"clamp".to_string()));
}

#[test]
#[serial]
fn test_symbol_count_stability_with_functions() {
    symbol::clear();

    // Parse a program with both symbols and functions
    let src = r#"
        pred item(symbol).
        item(one).
        item(two).
        item(three).

        func double(X) = X * 2.
    "#;

    let _prog = parse_program(src).unwrap();

    let count1 = symbol::count();

    // Parse again - should not add new symbols
    let _prog2 = parse_program(src).unwrap();

    let count2 = symbol::count();

    // Count should remain stable since same symbols are reused
    assert_eq!(
        count1, count2,
        "Symbol count should remain stable on repeated parsing"
    );
}
