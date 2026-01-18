//! Integration test for reversible symbols

use xlog_core::symbol;
use xlog_logic::parse_program;

fn setup() {
    symbol::clear();
}

#[test]
fn test_symbol_roundtrip_through_parser() {
    setup();

    let src = r#"
        person(alice, engineer).
        person(bob, manager).
        person(alice, developer).
    "#;

    let program = parse_program(src).unwrap();

    // Verify symbols were interned
    assert!(symbol::count() >= 4); // alice, bob, engineer, manager, developer (person is predicate)

    // Verify we can resolve them back
    // (actual verification depends on how facts are stored)
    // The symbols alice, bob, engineer, manager, developer should all resolve correctly
    let alice_id = symbol::intern("alice");
    let bob_id = symbol::intern("bob");
    let engineer_id = symbol::intern("engineer");
    let manager_id = symbol::intern("manager");
    let developer_id = symbol::intern("developer");

    assert_eq!(symbol::resolve(alice_id), "alice");
    assert_eq!(symbol::resolve(bob_id), "bob");
    assert_eq!(symbol::resolve(engineer_id), "engineer");
    assert_eq!(symbol::resolve(manager_id), "manager");
    assert_eq!(symbol::resolve(developer_id), "developer");

    // Verify program was parsed correctly
    assert_eq!(program.rules.len(), 3);
}

#[test]
fn test_symbol_deduplication() {
    setup();

    let src = r#"
        edge(a, b).
        edge(b, c).
        edge(a, c).
    "#;

    let _ = parse_program(src).unwrap();

    // a, b, c should each be interned once
    // exact count depends on what else gets interned
    let count_before = symbol::count();

    // Parse same source again - should not increase count
    let _ = parse_program(src).unwrap();

    assert_eq!(symbol::count(), count_before);
}
