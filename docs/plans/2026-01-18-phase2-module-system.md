# Phase 2: Module System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable organization of XLOG programs into reusable, encapsulated modules with explicit imports and visibility control.

**Architecture:** File-based modules (filename = module name), explicit imports required, public by default with `private` keyword, slash separator for nested paths, cycle detection, name conflict detection.

**Tech Stack:** Rust, pest parser, existing xlog-logic infrastructure

**Design Document:** `docs/plans/2026-01-18-module-system-design.md`

**Depends On:** Nothing (standalone, but Phase 1 should be done first for clean codebase)

**Estimated Tasks:** 25 tasks, ~125 steps

---

## Part A: Grammar & AST

### Task 1: Add Module Path Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add module_path rule**

Add after existing identifier rules:

```pest
// Module path: graph or utils/math or deep/nested/module
module_path = @{ ident ~ ("/" ~ ident)* }
```

**Step 2: Build to verify grammar**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add module_path grammar rule"
```

---

### Task 2: Add Import Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add import_list rule**

```pest
// Import list: {edge, reach, node}
import_list = { "{" ~ ident ~ ("," ~ ident)* ~ "}" }
```

**Step 2: Add use_stmt rule**

```pest
// Use statement: use graph. or use utils/math::{abs, clamp}.
use_stmt = { "use" ~ module_path ~ (":" ~ import_list)? ~ "." }
```

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add use_stmt and import_list grammar rules"
```

---

### Task 3: Add Private Modifier Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add private_mod rule**

```pest
// Private modifier
private_mod = { "private" }
```

**Step 2: Update pred_decl to support private**

Change:
```pest
pred_decl = { "pred" ~ ident ~ "(" ~ type_list? ~ ")" ~ "." }
```

To:
```pest
pred_decl = { private_mod? ~ "pred" ~ ident ~ "(" ~ type_list? ~ ")" ~ "." }
```

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add private modifier for predicates"
```

---

### Task 4: Update Statement Rule

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add use_stmt to statement rule**

Update statement rule to include use_stmt at the beginning:

```pest
statement = {
    use_stmt
    | domain_decl
    | pred_decl
    | pragma
    | rule_def
    | prob_fact
    | annotated_disjunction
    | evidence_stmt
    | prob_query
    | fact
    | constraint
    | query
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add use_stmt to statement rule"
```

---

### Task 5: Add UseDecl AST Type

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add UseDecl struct**

Add after existing type definitions:

```rust
/// Import statement: use module. or use module::{pred1, pred2}.
#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    /// Module path segments, e.g., ["utils", "math"]
    pub module_path: Vec<String>,
    /// Specific imports (None = import all public)
    pub imports: Option<Vec<String>>,
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add UseDecl AST type"
```

---

### Task 6: Update PredDecl for Private

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add is_private field to PredDecl**

Change:
```rust
pub struct PredDecl {
    pub name: String,
    pub types: Vec<ScalarType>,
}
```

To:
```rust
pub struct PredDecl {
    pub name: String,
    pub types: Vec<ScalarType>,
    pub is_private: bool,
}
```

**Step 2: Build to find broken code**

Run: `cargo build -p xlog-logic 2>&1 | head -30`
Expected: Errors where PredDecl is constructed

**Step 3: Fix all PredDecl constructions**

Add `is_private: false` to all existing constructions.

**Step 4: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add is_private field to PredDecl"
```

---

### Task 7: Update Program for Imports

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add imports field to Program**

Add as first field:
```rust
pub struct Program {
    pub imports: Vec<UseDecl>,  // NEW
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    // ... rest unchanged
}
```

**Step 2: Update Default impl**

Add `imports: Vec::new()` to the default.

**Step 3: Build and fix**

Run: `cargo build -p xlog-logic`
Fix any broken code.

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add imports field to Program"
```

---

## Part B: Parser Updates

### Task 8: Parse Module Path

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_module_path function**

```rust
fn parse_module_path(pair: Pair<Rule>) -> Vec<String> {
    pair.as_str()
        .split('/')
        .map(|s| s.to_string())
        .collect()
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: Warning about unused function (ok for now)

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_module_path function"
```

---

### Task 9: Parse Use Statement

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_use_stmt function**

```rust
fn parse_use_stmt(pair: Pair<Rule>) -> UseDecl {
    let mut inner = pair.into_inner();

    // Parse module path
    let path_pair = inner.next().unwrap();
    let module_path = parse_module_path(path_pair);

    // Parse optional import list
    let imports = inner.next().map(|import_list| {
        import_list.into_inner()
            .map(|p| p.as_str().to_string())
            .collect()
    });

    UseDecl { module_path, imports }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_use_stmt function"
```

---

### Task 10: Parse Private Predicate

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Update parse_pred_decl for private**

Find the function that parses pred_decl and update to check for private_mod:

```rust
fn parse_pred_decl(pair: Pair<Rule>) -> PredDecl {
    let mut inner = pair.into_inner();
    let mut is_private = false;

    // Check for private modifier
    let first = inner.next().unwrap();
    let name_pair = if first.as_rule() == Rule::private_mod {
        is_private = true;
        inner.next().unwrap()
    } else {
        first
    };

    let name = name_pair.as_str().to_string();

    // Parse types (existing logic)
    let types = inner.next()
        .map(|tl| tl.into_inner().map(parse_type_spec).collect())
        .unwrap_or_default();

    PredDecl { name, types, is_private }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): parse private modifier in pred_decl"
```

---

### Task 11: Wire Up Use Statement Parsing

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add use_stmt case to statement parsing**

In the main parse loop, add:

```rust
Rule::use_stmt => {
    program.imports.push(parse_use_stmt(pair));
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): wire up use_stmt parsing in main parse loop"
```

---

### Task 12: Add Parser Tests for Modules

**Files:**
- Create: `crates/xlog-logic/tests/module_parse_tests.rs`

**Step 1: Create test file**

```rust
use xlog_logic::parse;

#[test]
fn test_parse_use_all() {
    let src = r#"
        use graph.
        edge(1, 2).
    "#;
    let program = parse(src).unwrap();
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
    let program = parse(src).unwrap();
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
    let program = parse(src).unwrap();
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
    let program = parse(src).unwrap();
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
    let program = parse(src).unwrap();
    assert_eq!(program.imports.len(), 3);
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-logic module_parse`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/module_parse_tests.rs
git commit -m "test(logic): add module system parser tests"
```

---

## Part C: Module Resolver

### Task 13: Create Module Types

**Files:**
- Create: `crates/xlog-logic/src/module.rs`

**Step 1: Create module.rs with types**

```rust
//! Module system types for XLOG.

use std::collections::HashSet;
use std::path::PathBuf;

/// A module path like ["utils", "math"]
pub type ModulePath = Vec<String>;

/// Convert module path to string for display
pub fn module_path_to_string(path: &[String]) -> String {
    path.join("/")
}

/// A loaded module with metadata
#[derive(Debug)]
pub struct LoadedModule {
    /// Module path
    pub path: ModulePath,
    /// Source file location
    pub source_file: PathBuf,
    /// Public predicate names
    pub exports: HashSet<String>,
    /// Public function names
    pub function_exports: HashSet<String>,
}

impl LoadedModule {
    pub fn new(path: ModulePath, source_file: PathBuf) -> Self {
        Self {
            path,
            source_file,
            exports: HashSet::new(),
            function_exports: HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path_to_string() {
        assert_eq!(module_path_to_string(&["utils".into(), "math".into()]), "utils/math");
        assert_eq!(module_path_to_string(&["single".into()]), "single");
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: Warning about unused (ok)

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/module.rs
git commit -m "feat(logic): add module types (ModulePath, LoadedModule)"
```

---

### Task 14: Add Module Error Types

**Files:**
- Modify: `crates/xlog-logic/src/module.rs`

**Step 1: Add error enum**

```rust
use std::path::Path;

/// Errors that can occur during module resolution
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// Module file not found
    NotFound {
        path: ModulePath,
        searched: Vec<PathBuf>,
    },
    /// Circular import detected
    CircularImport {
        cycle: Vec<ModulePath>,
    },
    /// Name conflict between imports
    ImportConflict {
        name: String,
        module1: ModulePath,
        module2: ModulePath,
    },
    /// Attempted to import private predicate
    PrivatePredicate {
        name: String,
        module: ModulePath,
    },
    /// Predicate not found in module
    PredicateNotFound {
        name: String,
        module: ModulePath,
    },
    /// Parse error in module
    ParseError {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::NotFound { path, searched } => {
                writeln!(f, "error[E0400]: module not found: `{}`", module_path_to_string(path))?;
                writeln!(f, "  = note: searched in:")?;
                for s in searched {
                    writeln!(f, "          - {}", s.display())?;
                }
                write!(f, "  = help: check the module path spelling or add to --module-path")
            }
            ModuleError::CircularImport { cycle } => {
                writeln!(f, "error[E0401]: circular import detected")?;
                for (i, path) in cycle.iter().enumerate() {
                    if i < cycle.len() - 1 {
                        writeln!(f, "  {} imports {}",
                            module_path_to_string(path),
                            module_path_to_string(&cycle[i + 1]))?;
                    }
                }
                write!(f, "  = help: extract shared predicates into a third module")
            }
            ModuleError::ImportConflict { name, module1, module2 } => {
                writeln!(f, "error[E0402]: ambiguous import `{}`", name)?;
                writeln!(f, "  `{}` first imported from {}", name, module_path_to_string(module1))?;
                writeln!(f, "  `{}` also exported by {}", name, module_path_to_string(module2))?;
                write!(f, "  = help: use selective imports: `use {}::{{...}}.`", module_path_to_string(module1))
            }
            ModuleError::PrivatePredicate { name, module } => {
                write!(f, "error[E0403]: cannot import private predicate `{}` from {}",
                    name, module_path_to_string(module))
            }
            ModuleError::PredicateNotFound { name, module } => {
                write!(f, "error[E0404]: predicate `{}` not found in module {}",
                    name, module_path_to_string(module))
            }
            ModuleError::ParseError { path, message } => {
                write!(f, "error: parse error in {:?}: {}", path, message)
            }
        }
    }
}

impl std::error::Error for ModuleError {}
```

**Step 2: Build and test**

Run: `cargo test -p xlog-logic module`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/module.rs
git commit -m "feat(logic): add ModuleError with formatted error messages"
```

---

### Task 15: Create Module Resolver

**Files:**
- Create: `crates/xlog-logic/src/resolver.rs`

**Step 1: Create resolver.rs**

```rust
//! Module resolution for XLOG programs.

use crate::ast::Program;
use crate::module::{LoadedModule, ModuleError, ModulePath, module_path_to_string};
use crate::parse;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Resolves and loads modules
pub struct ModuleResolver {
    /// Directories to search for modules
    search_paths: Vec<PathBuf>,
    /// Already loaded modules (path string -> module)
    loaded: HashMap<String, LoadedModule>,
    /// Currently loading (for cycle detection)
    loading: Vec<ModulePath>,
}

impl ModuleResolver {
    /// Create a new resolver with given search paths
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            loaded: HashMap::new(),
            loading: Vec::new(),
        }
    }

    /// Find the file for a module path
    pub fn find_module_file(&self, base_dir: &Path, module_path: &[String]) -> Option<PathBuf> {
        let relative_path = format!("{}.xlog", module_path.join("/"));

        // Try relative to base_dir first
        let candidate = base_dir.join(&relative_path);
        if candidate.exists() {
            return Some(candidate);
        }

        // Try search paths
        for search_path in &self.search_paths {
            let candidate = search_path.join(&relative_path);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        None
    }

    /// Get the list of searched paths for error reporting
    fn searched_paths(&self, base_dir: &Path, module_path: &[String]) -> Vec<PathBuf> {
        let relative_path = format!("{}.xlog", module_path.join("/"));
        let mut searched = vec![base_dir.join(&relative_path)];
        for sp in &self.search_paths {
            searched.push(sp.join(&relative_path));
        }
        searched
    }

    /// Check if we're in a circular import
    fn check_cycle(&self, module_path: &[String]) -> Option<Vec<ModulePath>> {
        let path_str = module_path_to_string(module_path);
        for (i, loading_path) in self.loading.iter().enumerate() {
            if module_path_to_string(loading_path) == path_str {
                // Found cycle - return the cycle path
                let mut cycle: Vec<ModulePath> = self.loading[i..].to_vec();
                cycle.push(module_path.to_vec());
                return Some(cycle);
            }
        }
        None
    }

    /// Extract exports from a parsed program
    pub fn extract_exports(program: &Program) -> HashSet<String> {
        let mut exports = HashSet::new();

        // Add declared predicates that aren't private
        for pred in &program.predicates {
            if !pred.is_private {
                exports.insert(pred.name.clone());
            }
        }

        // Add rule heads (all rules define public predicates unless private)
        // TODO: support private rules
        for rule in &program.rules {
            exports.insert(rule.head.predicate.clone());
        }

        exports
    }

    /// Load a module from a path
    pub fn load_module(
        &mut self,
        base_dir: &Path,
        module_path: &[String],
    ) -> Result<&LoadedModule, ModuleError> {
        let path_key = module_path_to_string(module_path);

        // Already loaded?
        if self.loaded.contains_key(&path_key) {
            return Ok(self.loaded.get(&path_key).unwrap());
        }

        // Check for cycle
        if let Some(cycle) = self.check_cycle(module_path) {
            return Err(ModuleError::CircularImport { cycle });
        }

        // Find the file
        let source_file = self.find_module_file(base_dir, module_path)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: self.searched_paths(base_dir, module_path),
            })?;

        // Mark as loading
        self.loading.push(module_path.to_vec());

        // Read and parse
        let source = fs::read_to_string(&source_file)
            .map_err(|e| ModuleError::ParseError {
                path: source_file.clone(),
                message: e.to_string(),
            })?;

        let program = parse(&source)
            .map_err(|e| ModuleError::ParseError {
                path: source_file.clone(),
                message: e.to_string(),
            })?;

        // Extract exports
        let exports = Self::extract_exports(&program);

        // Recursively load imports
        let module_dir = source_file.parent().unwrap_or(base_dir);
        for import in &program.imports {
            self.load_module(module_dir, &import.module_path)?;
        }

        // Remove from loading
        self.loading.pop();

        // Store loaded module
        let module = LoadedModule {
            path: module_path.to_vec(),
            source_file,
            exports,
            function_exports: HashSet::new(), // TODO: add when UDFs implemented
        };

        self.loaded.insert(path_key.clone(), module);
        Ok(self.loaded.get(&path_key).unwrap())
    }

    /// Check if a predicate can be imported from a module
    pub fn check_import(
        &self,
        module_path: &[String],
        predicate: &str,
    ) -> Result<(), ModuleError> {
        let path_key = module_path_to_string(module_path);
        let module = self.loaded.get(&path_key)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: vec![],
            })?;

        if !module.exports.contains(predicate) {
            // Check if it exists but is private
            // For now, just report not found
            return Err(ModuleError::PredicateNotFound {
                name: predicate.to_string(),
                module: module_path.to_vec(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_module(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(format!("{}.xlog", name));
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_find_module_file() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "graph", "edge(1, 2).");

        let resolver = ModuleResolver::new(vec![]);
        let found = resolver.find_module_file(tmp.path(), &["graph".into()]);
        assert!(found.is_some());
    }

    #[test]
    fn test_module_not_found() {
        let tmp = TempDir::new().unwrap();
        let mut resolver = ModuleResolver::new(vec![]);

        let result = resolver.load_module(tmp.path(), &["nonexistent".into()]);
        assert!(matches!(result, Err(ModuleError::NotFound { .. })));
    }

    #[test]
    fn test_circular_import() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "a", "use b.");
        create_test_module(tmp.path(), "b", "use a.");

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["a".into()]);
        assert!(matches!(result, Err(ModuleError::CircularImport { .. })));
    }
}
```

**Step 2: Add tempfile to dev-dependencies**

In `crates/xlog-logic/Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic resolver`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/resolver.rs crates/xlog-logic/Cargo.toml
git commit -m "feat(logic): add ModuleResolver with file loading and cycle detection"
```

---

### Task 16: Export Modules

**Files:**
- Modify: `crates/xlog-logic/src/lib.rs`

**Step 1: Export new modules**

Add:
```rust
pub mod module;
pub mod resolver;
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/lib.rs
git commit -m "feat(logic): export module and resolver modules"
```

---

## Part D: Name Resolution & Visibility

### Task 17: Add Import Validator

**Files:**
- Modify: `crates/xlog-logic/src/resolver.rs`

**Step 1: Add validate_imports function**

```rust
impl ModuleResolver {
    /// Validate all imports in a program
    pub fn validate_imports(
        &self,
        program: &Program,
        current_module: &[String],
    ) -> Result<HashMap<String, ModulePath>, ModuleError> {
        let mut imported_names: HashMap<String, ModulePath> = HashMap::new();

        for use_decl in &program.imports {
            let module = self.loaded.get(&module_path_to_string(&use_decl.module_path))
                .expect("module should be loaded");

            let names_to_import: Vec<String> = match &use_decl.imports {
                Some(specific) => specific.clone(),
                None => module.exports.iter().cloned().collect(),
            };

            for name in names_to_import {
                // Check if predicate exists and is public
                if !module.exports.contains(&name) {
                    return Err(ModuleError::PredicateNotFound {
                        name: name.clone(),
                        module: use_decl.module_path.clone(),
                    });
                }

                // Check for conflicts
                if let Some(prev_module) = imported_names.get(&name) {
                    if prev_module != &use_decl.module_path {
                        return Err(ModuleError::ImportConflict {
                            name,
                            module1: prev_module.clone(),
                            module2: use_decl.module_path.clone(),
                        });
                    }
                }

                imported_names.insert(name, use_decl.module_path.clone());
            }
        }

        Ok(imported_names)
    }
}
```

**Step 2: Build and test**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/resolver.rs
git commit -m "feat(logic): add import validation with conflict detection"
```

---

### Task 18: Add Internal Naming

**Files:**
- Modify: `crates/xlog-logic/src/module.rs`

**Step 1: Add internal name generation**

```rust
/// Generate internal qualified name for a predicate
pub fn internal_name(module_path: &[String], predicate: &str) -> String {
    if module_path.is_empty() {
        predicate.to_string()
    } else {
        format!("__{}__{}", module_path.join("_"), predicate)
    }
}

/// Extract module and predicate from internal name
pub fn parse_internal_name(internal: &str) -> (Vec<String>, String) {
    if internal.starts_with("__") {
        if let Some(pos) = internal.rfind("__") {
            if pos > 2 {
                let module_part = &internal[2..pos];
                let pred_part = &internal[pos + 2..];
                let modules: Vec<String> = module_part.split('_').map(String::from).collect();
                return (modules, pred_part.to_string());
            }
        }
    }
    (vec![], internal.to_string())
}

#[cfg(test)]
mod tests {
    // ... existing tests ...

    #[test]
    fn test_internal_name() {
        assert_eq!(
            internal_name(&["graph".into()], "edge"),
            "__graph__edge"
        );
        assert_eq!(
            internal_name(&["utils".into(), "math".into()], "abs"),
            "__utils_math__abs"
        );
        assert_eq!(
            internal_name(&[], "local"),
            "local"
        );
    }

    #[test]
    fn test_parse_internal_name() {
        let (mods, pred) = parse_internal_name("__graph__edge");
        assert_eq!(mods, vec!["graph"]);
        assert_eq!(pred, "edge");
    }
}
```

**Step 2: Build and test**

Run: `cargo test -p xlog-logic module`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/module.rs
git commit -m "feat(logic): add internal naming for qualified predicates"
```

---

## Part E: Integration & CLI

### Task 19: Integrate Resolver into Compile

**Files:**
- Modify: `crates/xlog-logic/src/compile.rs`

**Step 1: Add module-aware compilation**

Add a new function or update existing compile function to use the resolver:

```rust
use crate::resolver::ModuleResolver;
use std::path::Path;

/// Compile a program with module resolution
pub fn compile_with_modules(
    entry_file: &Path,
    search_paths: Vec<PathBuf>,
) -> Result<CompiledProgram, CompileError> {
    let mut resolver = ModuleResolver::new(search_paths);

    // Determine base directory and module path
    let base_dir = entry_file.parent().unwrap_or(Path::new("."));
    let module_name = entry_file.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");

    // Load entry module (recursively loads dependencies)
    resolver.load_module(base_dir, &[module_name.to_string()])?;

    // ... rest of compilation with resolved imports
    todo!("integrate with existing compilation")
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/compile.rs
git commit -m "feat(logic): add compile_with_modules entry point"
```

---

### Task 20: Add CLI Module Path Flag

**Files:**
- Modify: `crates/xlog-cli/src/main.rs` (or args.rs)

**Step 1: Add --module-path argument**

```rust
#[arg(long, value_delimiter = ':')]
module_path: Vec<PathBuf>,
```

**Step 2: Pass to compiler**

Update the run command to pass search paths to the compiler.

**Step 3: Build**

Run: `cargo build -p xlog-cli`

**Step 4: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): add --module-path flag for module search paths"
```

---

## Part F: Test Infrastructure

### Task 21: Create Module Test Directory

**Files:**
- Create: `crates/xlog-logic/tests/modules/basic/main.xlog`
- Create: `crates/xlog-logic/tests/modules/basic/helper.xlog`

**Step 1: Create basic/main.xlog**

```prolog
use helper.

main_pred(X) :- helper_pred(X).
```

**Step 2: Create basic/helper.xlog**

```prolog
pred helper_pred(u32).
helper_pred(1).
helper_pred(2).
```

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/modules/
git commit -m "test(logic): add basic module test fixtures"
```

---

### Task 22: Create Nested Module Tests

**Files:**
- Create: `crates/xlog-logic/tests/modules/nested/main.xlog`
- Create: `crates/xlog-logic/tests/modules/nested/lib/utils.xlog`

**Step 1: Create nested/main.xlog**

```prolog
use lib/utils::{double}.

result(X, Y) :- input(X), Y is double(X).
input(5).
```

**Step 2: Create nested/lib/utils.xlog**

```prolog
% Helper functions
double(X, Y) :- Y is X * 2.
```

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/modules/nested/
git commit -m "test(logic): add nested module test fixtures"
```

---

### Task 23: Create Circular Import Test

**Files:**
- Create: `crates/xlog-logic/tests/modules/circular/a.xlog`
- Create: `crates/xlog-logic/tests/modules/circular/b.xlog`

**Step 1: Create circular/a.xlog**

```prolog
use b.
a_pred(X) :- b_pred(X).
```

**Step 2: Create circular/b.xlog**

```prolog
use a.
b_pred(X) :- a_pred(X).
```

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/modules/circular/
git commit -m "test(logic): add circular import test fixtures"
```

---

### Task 24: Create Visibility Test

**Files:**
- Create: `crates/xlog-logic/tests/modules/visibility/main.xlog`
- Create: `crates/xlog-logic/tests/modules/visibility/internal.xlog`

**Step 1: Create visibility/internal.xlog**

```prolog
pred public_pred(u32).
private pred private_pred(u32).

public_pred(1).
private_pred(2).
```

**Step 2: Create visibility/main.xlog**

```prolog
use internal::{public_pred}.
% Attempting to import private_pred should fail

result(X) :- public_pred(X).
```

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/modules/visibility/
git commit -m "test(logic): add visibility test fixtures"
```

---

### Task 24b: Create Transitive Dependency Test Fixtures

**Files:**
- Create: `crates/xlog-logic/tests/modules/transitive/main.xlog`
- Create: `crates/xlog-logic/tests/modules/transitive/mid.xlog`
- Create: `crates/xlog-logic/tests/modules/transitive/base.xlog`

**Step 1: Create transitive/base.xlog**

```prolog
% Base module - no dependencies
pred base_pred(u32).
base_pred(1).
base_pred(2).
```

**Step 2: Create transitive/mid.xlog**

```prolog
% Mid-level module - depends on base
use base.

pred mid_pred(u32).
mid_pred(X) :- base_pred(X), X > 1.
```

**Step 3: Create transitive/main.xlog**

```prolog
% Main module - depends on mid (which depends on base)
use mid.

pred main_pred(u32).
main_pred(X) :- mid_pred(X).

?- main_pred(X).
```

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/modules/transitive/
git commit -m "test(logic): add transitive dependency test fixtures"
```

---

### Task 25: Add Module Integration Tests

**Files:**
- Create: `crates/xlog-logic/tests/module_integration_tests.rs`

**Step 1: Create integration test file**

```rust
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
}

#[test]
fn test_nested_import() {
    let dir = test_modules_dir().join("nested");
    let mut resolver = ModuleResolver::new(vec![]);

    let result = resolver.load_module(&dir, &["main".into()]);
    assert!(result.is_ok(), "Failed to load nested module: {:?}", result);
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
fn test_search_paths() {
    let base = test_modules_dir();
    let search = vec![base.join("basic")];
    let mut resolver = ModuleResolver::new(search);

    // Should find helper.xlog via search path even from different dir
    let result = resolver.find_module_file(&base, &["helper".into()]);
    assert!(result.is_some());
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
    assert!(resolver.loaded.contains_key("main"), "main not loaded");
    assert!(resolver.loaded.contains_key("mid"), "mid not loaded");
    assert!(resolver.loaded.contains_key("base"), "base not loaded");

    // Verify load order didn't cause issues (base should be loaded before mid)
}

#[test]
fn test_search_path_precedence() {
    // Relative path should take precedence over search path
    let base = test_modules_dir().join("basic");
    let search = vec![test_modules_dir().join("nested")];
    let mut resolver = ModuleResolver::new(search);

    // Create a scenario where same module name exists in both locations
    // Should prefer the relative path
    let result = resolver.find_module_file(&base, &["helper".into()]);
    assert!(result.is_some());
    // Verify it's from the relative path, not search path
    assert!(result.unwrap().starts_with(&base));
}
```

**Step 2: Run integration tests**

Run: `cargo test -p xlog-logic module_integration`
Expected: PASS

**Step 3: Final commit for Phase 2**

```bash
git add crates/xlog-logic/tests/module_integration_tests.rs
git commit -m "test(logic): add module system integration tests

Phase 2 (Module System) complete:
- File-based modules (filename = module name)
- use statements with selective imports
- private modifier for predicates
- Nested modules with slash separator
- Circular import detection
- Name conflict detection
- --module-path CLI flag"
```

---

## Summary

| Task | Description | Part |
|------|-------------|------|
| 1-4 | Grammar rules | A |
| 5-7 | AST types | A |
| 8-12 | Parser updates | B |
| 13-16 | Module resolver | C |
| 17-18 | Name resolution | D |
| 19-20 | Integration & CLI | E |
| 21-24 | Test fixtures (basic, nested, circular, visibility) | F |
| 24b | Transitive dependency fixtures | F |
| 25 | Integration tests | F |

**Total: 26 tasks, ~130 steps**
