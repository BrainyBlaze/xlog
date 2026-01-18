//! Tests for module system parsing

use xlog_logic::parse_program;

#[test]
fn test_parse_use_all() {
    let src = r#"
        use graph.
        edge(1, 2).
    "#;
    let program = parse_program(src).unwrap();
    assert_eq!(program.imports.len(), 1);
    assert_eq!(program.imports[0].module_path, vec!["graph"]);
    assert!(program.imports[0].imports.is_none());
}

#[test]
fn test_parse_use_selective() {
    let src = r#"
        use graph::{edge, reach}.
        node(1).
    "#;
    let program = parse_program(src).unwrap();
    assert_eq!(program.imports.len(), 1);
    assert_eq!(program.imports[0].module_path, vec!["graph"]);
    assert_eq!(
        program.imports[0].imports,
        Some(vec!["edge".to_string(), "reach".to_string()])
    );
}

#[test]
fn test_parse_use_nested() {
    let src = r#"
        use utils/math.
        use deep/nested/module::{foo}.
    "#;
    let program = parse_program(src).unwrap();
    assert_eq!(program.imports.len(), 2);
    assert_eq!(program.imports[0].module_path, vec!["utils", "math"]);
    assert_eq!(
        program.imports[1].module_path,
        vec!["deep", "nested", "module"]
    );
}

#[test]
fn test_parse_private_pred() {
    let src = r#"
        pred public_pred(u32).
        private pred private_pred(u32, u32).
    "#;
    let program = parse_program(src).unwrap();
    assert_eq!(program.predicates.len(), 2);
    assert!(!program.predicates[0].is_private);
    assert!(program.predicates[1].is_private);
}

#[test]
fn test_parse_multiple_imports() {
    let src = r#"
        use a.
        use b.
        use c/d::{x, y, z}.
        fact(1).
    "#;
    let program = parse_program(src).unwrap();
    assert_eq!(program.imports.len(), 3);
}
