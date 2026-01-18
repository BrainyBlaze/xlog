//! Integration tests for module system

use std::path::PathBuf;
use xlog_logic::resolver::ModuleResolver;

fn test_modules_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("modules")
}

#[test]
fn test_basic_import() {
    let dir = test_modules_dir().join("basic");
    let mut resolver = ModuleResolver::new(vec![]);

    let result = resolver.load_module(&dir, &["main".into()]);
    assert!(result.is_ok(), "Failed to load basic module: {:?}", result);

    // Verify both modules were loaded
    assert!(resolver.is_loaded("main"), "main not loaded");
    assert!(resolver.is_loaded("helper"), "helper not loaded");
}

#[test]
fn test_nested_import() {
    let dir = test_modules_dir().join("nested");
    let mut resolver = ModuleResolver::new(vec![]);

    let result = resolver.load_module(&dir, &["main".into()]);
    assert!(result.is_ok(), "Failed to load nested module: {:?}", result);

    // Verify nested module path
    assert!(resolver.is_loaded("main"), "main not loaded");
    assert!(resolver.is_loaded("lib/utils"), "lib/utils not loaded");
}

#[test]
fn test_circular_import_detected() {
    let dir = test_modules_dir().join("circular");
    let mut resolver = ModuleResolver::new(vec![]);

    let result = resolver.load_module(&dir, &["a".into()]);
    assert!(
        matches!(result, Err(xlog_logic::module::ModuleError::CircularImport { .. })),
        "Expected CircularImport error, got: {:?}",
        result
    );
}

#[test]
fn test_visibility_exports() {
    let dir = test_modules_dir().join("visibility");
    let mut resolver = ModuleResolver::new(vec![]);

    // Load internal module
    let result = resolver.load_module(&dir, &["internal".into()]);
    assert!(result.is_ok(), "Failed to load visibility module: {:?}", result);

    // Check that public_pred is exported but private_pred is not
    let module = resolver.get_module(&["internal".into()]).unwrap();
    assert!(module.exports.contains("public_pred"), "public_pred should be exported");
    assert!(!module.exports.contains("private_pred"), "private_pred should NOT be exported");
}

#[test]
fn test_search_paths() {
    let base = test_modules_dir();
    let search = vec![base.join("basic")];
    let resolver = ModuleResolver::new(search);

    // Should find helper.xlog via search path even from different dir
    let result = resolver.find_module_file(&base, &["helper".into()]);
    assert!(result.is_some(), "helper.xlog not found via search path");
}

#[test]
fn test_transitive_dependencies() {
    // Test: main → mid → base (transitive chain)
    let dir = test_modules_dir().join("transitive");
    let mut resolver = ModuleResolver::new(vec![]);

    // Loading main should recursively load mid and base
    let result = resolver.load_module(&dir, &["main".into()]);
    assert!(result.is_ok(), "Failed to load transitive chain: {:?}", result);

    // Verify all three modules were loaded
    assert!(resolver.is_loaded("main"), "main not loaded");
    assert!(resolver.is_loaded("mid"), "mid not loaded");
    assert!(resolver.is_loaded("base"), "base not loaded");
}

#[test]
fn test_search_path_precedence() {
    // Relative path should take precedence over search path
    let base = test_modules_dir().join("basic");
    let search = vec![test_modules_dir().join("nested")];
    let resolver = ModuleResolver::new(search);

    // helper.xlog exists in basic/, should be found there first
    let result = resolver.find_module_file(&base, &["helper".into()]);
    assert!(result.is_some(), "helper not found");
    // Verify it's from the relative path, not search path
    assert!(result.unwrap().starts_with(&base), "Should find helper in relative path first");
}

#[test]
fn test_module_exports_from_rules() {
    let dir = test_modules_dir().join("basic");
    let mut resolver = ModuleResolver::new(vec![]);

    let result = resolver.load_module(&dir, &["helper".into()]);
    assert!(result.is_ok());

    let module = resolver.get_module(&["helper".into()]).unwrap();
    // helper_pred should be exported (defined in pred decl and rules)
    assert!(module.exports.contains("helper_pred"));
}
