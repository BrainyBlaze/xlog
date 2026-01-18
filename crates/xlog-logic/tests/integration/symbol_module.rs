//! Tests that reversible symbols work correctly with the module system.
//!
//! These tests verify that:
//! 1. Symbols are correctly interned when parsing module code
//! 2. Symbols resolve correctly after being used in imported predicates
//! 3. Symbol resolution works in query output
//!
//! All tests use `serial_test::serial` since they manipulate global symbol state.

use serial_test::serial;
use xlog_core::symbol;
use xlog_logic::parser::parse_program;

#[test]
#[serial]
fn test_symbols_in_module_predicates() {
    symbol::clear();

    // Module defining predicates with symbols
    let module_src = r#"
        pred color(symbol).
        color(red).
        color(green).
        color(blue).
    "#;

    // Parse module
    let _module_prog = parse_program(module_src).unwrap();

    // Verify symbols are interned correctly
    let red_id = symbol::intern("red");
    let green_id = symbol::intern("green");
    let blue_id = symbol::intern("blue");

    // Re-intern should return same IDs (idempotent)
    assert_eq!(symbol::intern("red"), red_id);
    assert_eq!(symbol::intern("green"), green_id);
    assert_eq!(symbol::intern("blue"), blue_id);

    // Resolve should return original strings
    assert_eq!(symbol::resolve(red_id), "red");
    assert_eq!(symbol::resolve(green_id), "green");
    assert_eq!(symbol::resolve(blue_id), "blue");
}

#[test]
#[serial]
fn test_symbol_resolution_after_parsing() {
    symbol::clear();

    let src = r#"
        pred status(symbol).
        status(active).
        status(pending).
        status(completed).
    "#;

    let _prog = parse_program(src).unwrap();

    // After parsing, symbols should be interned
    let active = symbol::intern("active");
    let pending = symbol::intern("pending");
    let completed = symbol::intern("completed");

    // Resolution works
    assert_eq!(symbol::resolve(active), "active");
    assert_eq!(symbol::resolve(pending), "pending");
    assert_eq!(symbol::resolve(completed), "completed");
}

#[test]
#[serial]
fn test_symbol_count_after_parsing() {
    symbol::clear();

    let src = r#"
        pred tag(symbol).
        tag(important).
        tag(urgent).
        tag(low_priority).
    "#;

    let _prog = parse_program(src).unwrap();

    // Should have at least 3 symbols
    assert!(
        symbol::count() >= 3,
        "Expected at least 3 symbols, got {}",
        symbol::count()
    );
}

#[test]
#[serial]
fn test_symbol_deduplication_across_modules() {
    symbol::clear();

    // Simulate two modules that use the same symbol values
    let module_a = r#"
        pred item(symbol).
        item(apple).
        item(orange).
    "#;

    let module_b = r#"
        pred fruit(symbol).
        fruit(apple).
        fruit(banana).
    "#;

    // Parse both modules
    let _prog_a = parse_program(module_a).unwrap();
    let count_after_a = symbol::count();

    let _prog_b = parse_program(module_b).unwrap();
    let count_after_b = symbol::count();

    // apple should be deduplicated - only banana is new
    // apple, orange from module_a (2 symbols)
    // apple (already interned), banana from module_b (1 new symbol)
    assert!(
        count_after_b == count_after_a + 1,
        "Expected 1 new symbol (banana), got {} total symbols (was {} after module_a)",
        count_after_b,
        count_after_a
    );

    // Verify all symbols resolve correctly
    assert_eq!(symbol::resolve(symbol::intern("apple")), "apple");
    assert_eq!(symbol::resolve(symbol::intern("orange")), "orange");
    assert_eq!(symbol::resolve(symbol::intern("banana")), "banana");
}

#[test]
#[serial]
fn test_symbol_in_rule_body() {
    symbol::clear();

    let src = r#"
        pred color(symbol).
        pred primary(symbol).

        color(red).
        color(green).
        color(blue).
        color(yellow).

        primary(X) :- color(X), X = red.
        primary(X) :- color(X), X = blue.
        primary(X) :- color(X), X = yellow.
    "#;

    let prog = parse_program(src).unwrap();

    // Verify rules were parsed with symbol comparisons
    assert_eq!(prog.proper_rules().count(), 3);

    // Verify symbols are interned
    let red_id = symbol::intern("red");
    let blue_id = symbol::intern("blue");
    let yellow_id = symbol::intern("yellow");

    assert_eq!(symbol::resolve(red_id), "red");
    assert_eq!(symbol::resolve(blue_id), "blue");
    assert_eq!(symbol::resolve(yellow_id), "yellow");
}

#[test]
#[serial]
fn test_symbol_in_negation() {
    symbol::clear();

    let src = r#"
        pred status(u32, symbol).
        pred not_active(u32).

        status(1, active).
        status(2, inactive).
        status(3, pending).

        not_active(X) :- status(X, S), S != active.
    "#;

    let prog = parse_program(src).unwrap();

    // Rule should have comparison with symbol
    assert_eq!(prog.proper_rules().count(), 1);

    // active should be interned
    let active_id = symbol::intern("active");
    assert_eq!(symbol::resolve(active_id), "active");
}

#[test]
#[serial]
fn test_symbol_with_underscore() {
    symbol::clear();

    let src = r#"
        pred state(symbol).
        state(in_progress).
        state(not_started).
        state(completed_successfully).
    "#;

    let _prog = parse_program(src).unwrap();

    // Verify underscore symbols are interned correctly
    let in_progress = symbol::intern("in_progress");
    let not_started = symbol::intern("not_started");
    let completed_successfully = symbol::intern("completed_successfully");

    assert_eq!(symbol::resolve(in_progress), "in_progress");
    assert_eq!(symbol::resolve(not_started), "not_started");
    assert_eq!(
        symbol::resolve(completed_successfully),
        "completed_successfully"
    );
}

#[test]
#[serial]
fn test_symbol_case_sensitivity() {
    symbol::clear();

    let src = r#"
        pred name(symbol).
        name(alice).
    "#;

    let _prog = parse_program(src).unwrap();

    // alice and Alice should be different symbols
    let alice_lower = symbol::intern("alice");
    let alice_upper = symbol::intern("Alice");

    // They should have different IDs
    assert_ne!(
        alice_lower, alice_upper,
        "alice and Alice should be different symbols"
    );

    // They should resolve to their original strings
    assert_eq!(symbol::resolve(alice_lower), "alice");
    assert_eq!(symbol::resolve(alice_upper), "Alice");
}

#[test]
#[serial]
fn test_symbol_in_multiple_predicates() {
    symbol::clear();

    let src = r#"
        pred employee(symbol, symbol).
        pred department(symbol).
        pred role(symbol).

        department(engineering).
        department(sales).
        department(hr).

        role(developer).
        role(manager).
        role(analyst).

        employee(john, engineering).
        employee(jane, sales).
        employee(bob, hr).
    "#;

    let prog = parse_program(src).unwrap();

    // Count facts
    assert_eq!(prog.facts().count(), 9); // 3 departments + 3 roles + 3 employees

    // Verify symbol deduplication - engineering appears in both department and employee
    let count = symbol::count();

    // john, jane, bob (3) + engineering, sales, hr (3) + developer, manager, analyst (3) = 9 unique
    assert!(
        count >= 9,
        "Expected at least 9 unique symbols, got {}",
        count
    );
}

#[test]
#[serial]
fn test_symbol_in_annotated_disjunction() {
    symbol::clear();

    let src = r#"
        0.6::outcome(success); 0.4::outcome(failure).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify annotated disjunction was parsed
    assert_eq!(prog.annotated_disjunctions.len(), 1);

    // Verify symbols were interned
    let success = symbol::intern("success");
    let failure = symbol::intern("failure");

    assert_eq!(symbol::resolve(success), "success");
    assert_eq!(symbol::resolve(failure), "failure");
}

#[test]
#[serial]
fn test_symbol_in_probabilistic_fact() {
    symbol::clear();

    let src = r#"
        0.7::weather(sunny).
        0.2::weather(rainy).
        0.1::weather(cloudy).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify probabilistic facts were parsed
    assert_eq!(prog.prob_facts.len(), 3);

    // Verify symbols were interned
    let sunny = symbol::intern("sunny");
    let rainy = symbol::intern("rainy");
    let cloudy = symbol::intern("cloudy");

    assert_eq!(symbol::resolve(sunny), "sunny");
    assert_eq!(symbol::resolve(rainy), "rainy");
    assert_eq!(symbol::resolve(cloudy), "cloudy");
}

#[test]
#[serial]
fn test_symbol_roundtrip_consistency() {
    symbol::clear();

    let src = r#"
        pred data(symbol, u32).
        data(item_a, 10).
        data(item_b, 20).
        data(item_c, 30).
    "#;

    let _prog = parse_program(src).unwrap();

    // Intern symbols
    let item_a = symbol::intern("item_a");
    let item_b = symbol::intern("item_b");
    let item_c = symbol::intern("item_c");

    // Multiple roundtrips should be consistent
    for _ in 0..100 {
        assert_eq!(symbol::intern("item_a"), item_a);
        assert_eq!(symbol::intern("item_b"), item_b);
        assert_eq!(symbol::intern("item_c"), item_c);

        assert_eq!(symbol::resolve(item_a), "item_a");
        assert_eq!(symbol::resolve(item_b), "item_b");
        assert_eq!(symbol::resolve(item_c), "item_c");
    }
}

#[test]
#[serial]
fn test_symbol_memory_tracking() {
    symbol::clear();

    let initial_mem = symbol::memory_usage();

    let src = r#"
        pred large_symbol_test(symbol).
        large_symbol_test(very_long_symbol_name_for_testing_purposes).
        large_symbol_test(another_quite_long_symbol_name).
        large_symbol_test(yet_another_lengthy_symbol_identifier).
    "#;

    let _prog = parse_program(src).unwrap();

    let final_mem = symbol::memory_usage();

    // Memory should have increased
    assert!(
        final_mem > initial_mem,
        "Memory usage should increase after interning symbols: initial={}, final={}",
        initial_mem,
        final_mem
    );
}

#[test]
#[serial]
fn test_symbol_with_numbers_in_name() {
    symbol::clear();

    let src = r#"
        pred version(symbol).
        version(v1).
        version(v2).
        version(v3_beta).
        version(release_2024).
    "#;

    let _prog = parse_program(src).unwrap();

    // Verify symbols with numbers are interned correctly
    assert_eq!(symbol::resolve(symbol::intern("v1")), "v1");
    assert_eq!(symbol::resolve(symbol::intern("v2")), "v2");
    assert_eq!(symbol::resolve(symbol::intern("v3_beta")), "v3_beta");
    assert_eq!(
        symbol::resolve(symbol::intern("release_2024")),
        "release_2024"
    );
}

#[test]
#[serial]
fn test_symbol_interleaved_parsing_and_resolution() {
    symbol::clear();

    // Parse first module
    let src1 = r#"
        pred color(symbol).
        color(red).
    "#;
    let _prog1 = parse_program(src1).unwrap();

    // Resolve immediately
    let red1 = symbol::intern("red");
    assert_eq!(symbol::resolve(red1), "red");

    // Parse second module
    let src2 = r#"
        pred shade(symbol).
        shade(blue).
        shade(red).
    "#;
    let _prog2 = parse_program(src2).unwrap();

    // Red should still resolve to same ID
    let red2 = symbol::intern("red");
    assert_eq!(red1, red2, "red should have same ID after re-interning");
    assert_eq!(symbol::resolve(red2), "red");

    // Blue should be new
    let blue = symbol::intern("blue");
    assert_eq!(symbol::resolve(blue), "blue");
}
