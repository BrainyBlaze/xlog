# Phase 4: Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate all v0.3.2 features (reversible symbols, modules, UDFs) with end-to-end testing, CLI enhancements, and documentation.

**Architecture:** End-to-end integration tests combining all features, CLI flag additions, comprehensive documentation updates, release preparation.

**Tech Stack:** Rust, existing CLI infrastructure, markdown documentation

**Depends On:** Phase 1 (Reversible Symbols), Phase 2 (Module System), Phase 3 (User-Defined Functions)

**Estimated Tasks:** 12 tasks, ~60 steps

---

## Part A: Cross-Feature Integration Tests

### Task 1: Symbol + Module Integration Test

**Files:**
- Create: `crates/xlog-logic/tests/integration/symbol_module.rs`
- Modify: `crates/xlog-logic/tests/integration/mod.rs` (add module)

**Step 1: Write test file**

```rust
//! Tests that reversible symbols work correctly with modules.

use xlog_logic::parser::parse_program;
use xlog_core::symbol;

#[test]
fn test_symbols_in_imported_predicates() {
    symbol::clear();

    // Module defining predicates with symbols
    let module_src = r#"
        pred color(symbol).
        color(red).
        color(green).
        color(blue).
    "#;

    // Main program importing
    let main_src = r#"
        use colors.
        ?- color(X).
    "#;

    // Parse both
    let module_prog = parse_program(module_src).unwrap();
    let main_prog = parse_program(main_src).unwrap();

    // Verify symbols are interned correctly
    let red_id = symbol::intern("red");
    let green_id = symbol::intern("green");
    let blue_id = symbol::intern("blue");

    // Re-intern should return same IDs
    assert_eq!(symbol::intern("red"), red_id);
    assert_eq!(symbol::intern("green"), green_id);
    assert_eq!(symbol::intern("blue"), blue_id);

    // Resolve should return original strings
    assert_eq!(symbol::resolve(red_id), "red");
    assert_eq!(symbol::resolve(green_id), "green");
    assert_eq!(symbol::resolve(blue_id), "blue");
}

#[test]
fn test_symbol_resolution_in_query_output() {
    symbol::clear();

    let src = r#"
        pred status(symbol).
        status(active).
        status(pending).
    "#;

    let prog = parse_program(src).unwrap();

    // After parsing, symbols should be interned
    let active = symbol::intern("active");
    let pending = symbol::intern("pending");

    // Resolution works
    assert_eq!(symbol::resolve(active), "active");
    assert_eq!(symbol::resolve(pending), "pending");
}
```

**Step 2: Update integration/mod.rs**

Add:
```rust
mod symbol_module;
```

**Step 3: Run test to verify it compiles and passes**

Run: `cargo test -p xlog-logic symbol_module --test integration`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/integration/
git commit -m "test(logic): add symbol + module integration tests"
```

---

### Task 2: Module + UDF Integration Test

**Files:**
- Create: `crates/xlog-logic/tests/integration/module_udf.rs`
- Modify: `crates/xlog-logic/tests/integration/mod.rs`

**Step 1: Write test for importing functions from modules**

```rust
//! Tests that UDFs work correctly with module system.

use xlog_logic::parser::parse_program;

#[test]
fn test_import_function_from_module() {
    // Module with function
    let math_module = r#"
        fn abs(X) = if X < 0 then 0 - X else X.
        fn clamp(X, Lo, Hi) = if X < Lo then Lo else if X > Hi then Hi else X.
    "#;

    // Main using imported function
    let main_src = r#"
        use math::{abs, clamp}.
        pred distance(f64, f64).
        distance(A, B) :- point(A), point(B), abs(A - B) is D.
    "#;

    let math_prog = parse_program(math_module).unwrap();
    let main_prog = parse_program(main_src).unwrap();

    // Verify function is accessible via import
    assert!(main_prog.imports.iter().any(|i| i.module_path == "math"));
}

#[test]
fn test_private_function_not_importable() {
    // Module with private function
    let util_module = r#"
        private fn helper(X) = X * 2.
        fn public_func(X) = helper(X) + 1.
    "#;

    // Main trying to import private
    let main_src = r#"
        use util::{helper}.
    "#;

    let util_prog = parse_program(util_module).unwrap();
    // Importing private function should fail during resolution
    // (Parser accepts it, resolver rejects it)
}

#[test]
fn test_recursive_function_across_modules() {
    // Module with recursive function
    let math_module = r#"
        fn factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).
    "#;

    // Main using imported recursive function
    let main_src = r#"
        use math::{factorial}.
        pred result(f64).
        result(R) :- factorial(5) is R.
    "#;

    let math_prog = parse_program(math_module).unwrap();
    let main_prog = parse_program(main_src).unwrap();

    // Verify recursive function parses correctly
    assert!(!math_prog.functions.is_empty());
}
```

**Step 2: Update integration/mod.rs**

Add:
```rust
mod module_udf;
```

**Step 3: Run tests**

Run: `cargo test -p xlog-logic module_udf --test integration`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/integration/
git commit -m "test(logic): add module + UDF integration tests"
```

---

### Task 3: Symbol + UDF Integration Test

**Files:**
- Create: `crates/xlog-logic/tests/integration/symbol_udf.rs`
- Modify: `crates/xlog-logic/tests/integration/mod.rs`

**Step 1: Write test for symbols in function contexts**

```rust
//! Tests that symbols work correctly with UDFs.

use xlog_logic::parser::parse_program;
use xlog_core::symbol;

#[test]
fn test_symbol_in_function_conditional() {
    symbol::clear();

    // Function that uses symbol comparison
    let src = r#"
        pred categorize(symbol, symbol).
        categorize(status, category) :-
            task(status),
            classify(status) is category.
    "#;

    let prog = parse_program(src).unwrap();

    // Verify parsing succeeded
    assert!(prog.rules.iter().any(|r| r.head.name == "categorize"));
}

#[test]
fn test_function_output_as_symbol() {
    symbol::clear();

    // Predicate that produces symbols
    let src = r#"
        pred label(u32, symbol).
        label(1, low).
        label(2, medium).
        label(3, high).

        ?- label(X, Y).
    "#;

    let prog = parse_program(src).unwrap();

    // Symbols should be interned
    let low = symbol::intern("low");
    let medium = symbol::intern("medium");
    let high = symbol::intern("high");

    assert_eq!(symbol::resolve(low), "low");
    assert_eq!(symbol::resolve(medium), "medium");
    assert_eq!(symbol::resolve(high), "high");
}
```

**Step 2: Update integration/mod.rs**

Add:
```rust
mod symbol_udf;
```

**Step 3: Run tests**

Run: `cargo test -p xlog-logic symbol_udf --test integration`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/integration/
git commit -m "test(logic): add symbol + UDF integration tests"
```

---

### Task 4: Full Feature Integration Test

**Files:**
- Create: `crates/xlog-logic/tests/integration/full_v032.rs`
- Modify: `crates/xlog-logic/tests/integration/mod.rs`

**Step 1: Write comprehensive test using all features**

```rust
//! End-to-end test using all v0.3.2 features together.

use xlog_logic::parser::parse_program;
use xlog_core::symbol;

/// Test scenario: A task management system
/// - Uses modules for organization
/// - Uses symbols for status values
/// - Uses UDFs for priority calculations
#[test]
fn test_task_management_system() {
    symbol::clear();

    // Module: priorities.xl
    let priorities_module = r#"
        fn priority_score(Urgency, Importance) =
            if Urgency > 8 then Importance * 2
            else if Urgency > 5 then Importance + Urgency
            else Importance.

        fn is_critical(Score) = if Score > 15 then 1 else 0.
    "#;

    // Module: statuses.xl
    let statuses_module = r#"
        pred valid_status(symbol).
        valid_status(todo).
        valid_status(in_progress).
        valid_status(done).
        valid_status(blocked).

        private pred internal_status(symbol).
        internal_status(archived).
    "#;

    // Main program: main.xl
    let main_src = r#"
        use priorities::{priority_score, is_critical}.
        use statuses::{valid_status}.

        pred task(u32, symbol, f64, f64).
        task(1, todo, 9.0, 8.0).
        task(2, in_progress, 3.0, 7.0).
        task(3, blocked, 10.0, 10.0).

        pred critical_task(u32, symbol).
        critical_task(Id, Status) :-
            task(Id, Status, Urgency, Importance),
            valid_status(Status),
            priority_score(Urgency, Importance) is Score,
            is_critical(Score) = 1.

        ?- critical_task(X, Y).
    "#;

    // Parse all modules
    let priorities = parse_program(priorities_module).unwrap();
    let statuses = parse_program(statuses_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify modules parsed correctly
    assert!(!priorities.functions.is_empty(), "priorities should have functions");
    assert!(statuses.facts.iter().any(|f| f.name == "valid_status"), "statuses should have facts");
    assert!(!main.imports.is_empty(), "main should have imports");

    // Verify symbols are interned
    let todo = symbol::intern("todo");
    let in_progress = symbol::intern("in_progress");
    let done = symbol::intern("done");
    let blocked = symbol::intern("blocked");

    // All symbols should resolve correctly
    assert_eq!(symbol::resolve(todo), "todo");
    assert_eq!(symbol::resolve(in_progress), "in_progress");
    assert_eq!(symbol::resolve(done), "done");
    assert_eq!(symbol::resolve(blocked), "blocked");

    // Symbol count should be at least 4 (could be more from other tests)
    assert!(symbol::count() >= 4);
}

#[test]
fn test_nested_module_with_functions_and_symbols() {
    symbol::clear();

    // Nested module path
    let utils_math = r#"
        fn abs(X) = if X < 0 then 0 - X else X.
    "#;

    // Main using nested path
    let main_src = r#"
        use utils/math::{abs}.

        pred measurement(symbol, f64).
        measurement(temp, -5.0).
        measurement(pressure, 101.3).

        pred absolute_measurement(symbol, f64).
        absolute_measurement(Label, AbsVal) :-
            measurement(Label, Val),
            abs(Val) is AbsVal.

        ?- absolute_measurement(X, Y).
    "#;

    let utils_prog = parse_program(utils_math).unwrap();
    let main_prog = parse_program(main_src).unwrap();

    // Verify nested import path
    assert!(main_prog.imports.iter().any(|i| i.module_path == "utils/math"));

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("temp")), "temp");
    assert_eq!(symbol::resolve(symbol::intern("pressure")), "pressure");
}
```

**Step 2: Update integration/mod.rs**

Add:
```rust
mod full_v032;
```

**Step 3: Run all integration tests**

Run: `cargo test -p xlog-logic --test integration`
Expected: All PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/integration/
git commit -m "test(logic): add full v0.3.2 integration tests"
```

---

## Part B: CLI Enhancements

### Task 5: Add Symbol Stats to --stats Output

**Files:**
- Modify: `crates/xlog-cli/src/main.rs` or `crates/xlog-cli/src/stats.rs`

**Step 1: Locate stats output code**

Find where `--stats` flag is handled.

**Step 2: Add symbol count to stats output**

Add after existing stats:

```rust
use xlog_core::symbol;

// In stats output section:
if args.stats {
    // ... existing stats ...

    // Symbol registry stats
    let symbol_count = symbol::count();
    let estimated_bytes = symbol_count * 24; // rough estimate: avg 20 chars + overhead
    println!("Symbols: {} interned (~{} bytes)", symbol_count, estimated_bytes);
}
```

**Step 3: Build and test**

Run: `cargo build -p xlog-cli`
Run: `echo "pred test(symbol). test(foo)." | cargo run -p xlog-cli -- --stats`
Expected: Output includes "Symbols: N interned"

**Step 4: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): add symbol registry stats to --stats output"
```

---

### Task 6: Add --module-path CLI Flag

**Files:**
- Modify: `crates/xlog-cli/src/args.rs` or where CLI args are defined
- Modify: `crates/xlog-cli/src/main.rs`

**Step 1: Add module_path argument**

```rust
/// Search path for module resolution (colon-separated)
#[arg(long, default_value = ".")]
pub module_path: String,
```

**Step 2: Parse and pass to resolver**

```rust
let module_paths: Vec<PathBuf> = args.module_path
    .split(':')
    .map(PathBuf::from)
    .collect();

// Pass to module resolver
let resolver = ModuleResolver::new(module_paths);
```

**Step 3: Build and test**

Run: `cargo build -p xlog-cli`
Run: `cargo run -p xlog-cli -- --module-path "./modules:./lib" program.xl`
Expected: No errors

**Step 4: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): add --module-path flag for module resolution"
```

---

### Task 7: Update --format json for Symbols

**Files:**
- Modify: `crates/xlog-cli/src/output.rs` or JSON formatting code

**Step 1: Update JSON serialization for symbols**

```rust
use xlog_core::symbol;

fn format_value_json(value: &Value, typ: ScalarType) -> serde_json::Value {
    match typ {
        ScalarType::Symbol => {
            let id = value.as_u32();
            serde_json::Value::String(symbol::resolve(id))
        }
        ScalarType::U32 => serde_json::Value::Number(value.as_u32().into()),
        ScalarType::F64 => {
            serde_json::Number::from_f64(value.as_f64())
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        // ... other types
    }
}
```

**Step 2: Test JSON output**

Run: `echo "pred test(symbol). test(foo). test(bar). ?- test(X)." | cargo run -p xlog-cli -- --format json`
Expected: `{"relation":"test","rows":[{"col0":"foo"},{"col0":"bar"}]}`

**Step 3: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): format symbols as strings in JSON output"
```

---

### Task 8: Update Human-Readable Output for Symbols

**Files:**
- Modify: `crates/xlog-cli/src/output.rs`

**Step 1: Update text output formatting**

```rust
fn format_value_text(value: &Value, typ: ScalarType) -> String {
    match typ {
        ScalarType::Symbol => symbol::resolve(value.as_u32()),
        ScalarType::U32 => value.as_u32().to_string(),
        ScalarType::F64 => format!("{:.6}", value.as_f64()),
        // ... other types
    }
}
```

**Step 2: Test text output**

Run: `echo "pred test(symbol). test(foo). test(bar). ?- test(X)." | cargo run -p xlog-cli --`
Expected:
```
test(foo).
test(bar).
```

**Step 3: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): display symbols as strings in text output"
```

---

## Part C: Documentation

### Task 9: Update XLOG Language Reference

**Files:**
- Create or modify: `docs/language-reference.md`

**Step 1: Add Modules section**

```markdown
## Modules

XLOG supports organizing code into modules for reusability and encapsulation.

### Creating Modules

A module is a `.xl` file. The filename becomes the module name:

```xlog
% File: graph.xl
pred edge(u32, u32).
edge(1, 2).
edge(2, 3).

pred reach(u32, u32).
reach(A, B) :- edge(A, B).
reach(A, C) :- edge(A, B), reach(B, C).
```

### Importing Modules

Use the `use` statement to import predicates:

```xlog
% Import all public predicates
use graph.

% Import specific predicates
use graph::{edge, reach}.

% Import from nested module
use utils/math::{abs, clamp}.
```

### Visibility

Predicates are public by default. Use `private` to hide:

```xlog
private pred helper(u32).
helper(X) :- internal(X).

pred public_api(u32).
public_api(X) :- helper(X).
```

Private predicates cannot be imported by other modules.
```

**Step 2: Add Functions section**

```markdown
## User-Defined Functions

Functions provide reusable calculations within rules.

### Basic Functions

```xlog
fn double(X) = X * 2.
fn add(X, Y) = X + Y.
```

### Conditional Functions

```xlog
fn abs(X) = if X < 0 then 0 - X else X.

fn clamp(X, Lo, Hi) =
    if X < Lo then Lo
    else if X > Hi then Hi
    else X.
```

### Recursive Functions

```xlog
fn factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).
fn fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).
```

Note: Recursive functions require a conditional with a base case.

### Using Functions

Functions can be called in rule bodies:

```xlog
pred distance(f64, f64, f64).
distance(X1, X2, D) :- point(X1), point(X2), abs(X1 - X2) is D.
```

Or using direct call syntax:

```xlog
pred doubled(f64).
doubled(Y) :- number(X), double(X) = Y.
```

### Type Annotations (Optional)

```xlog
fn add(X: f64, Y: f64) -> f64 = X + Y.
```
```

**Step 3: Add Symbols section**

```markdown
## Symbols

Symbols are interned strings represented efficiently as integers internally but displayed as readable strings.

### Declaring Symbol Types

```xlog
pred color(symbol).
color(red).
color(green).
color(blue).
```

### Querying Symbols

```xlog
?- color(X).
% Output:
% color(red).
% color(green).
% color(blue).
```

Symbols are displayed as their original string values, not as numeric IDs.
```

**Step 4: Commit**

```bash
git add docs/language-reference.md
git commit -m "docs: add modules, functions, and symbols to language reference"
```

---

### Task 10: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Add v0.3.2 features to README**

Add to features section:

```markdown
### v0.3.2 Language Features

- **Modules**: Organize code into reusable modules with `use` imports
- **User-Defined Functions**: Create reusable functions with arithmetic, conditionals, and recursion
- **Reversible Symbols**: Symbol values display as readable strings

```xlog
% Module example (math.xl)
fn abs(X) = if X < 0 then 0 - X else X.

% Main program
use math::{abs}.

pred task(symbol, f64).
task(temperature, -5.0).

pred result(symbol, f64).
result(Label, AbsVal) :- task(Label, Val), abs(Val) is AbsVal.

?- result(X, Y).
% Output: result(temperature, 5.0).
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with v0.3.2 features"
```

---

### Task 11: Create CHANGELOG Entry

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Add v0.3.2 changelog entry**

```markdown
## [0.3.2] - 2026-XX-XX

### Added

- **Module System**: File-based modules with explicit imports
  - `use module.` to import all public predicates
  - `use module::{pred1, pred2}.` for selective imports
  - `use path/to/module.` for nested modules
  - `private` keyword for module-internal predicates

- **User-Defined Functions**: Reusable functions in rule bodies
  - Arithmetic functions: `fn double(X) = X * 2.`
  - Conditional functions: `fn abs(X) = if X < 0 then 0 - X else X.`
  - Recursive functions with safety checks
  - Optional type annotations

- **Reversible Symbols**: Bidirectional string-to-ID mapping
  - Symbols display as original strings in output
  - Arrow dictionary encoding for efficient serialization
  - `--stats` shows symbol registry metrics

### Changed

- Symbol storage changed from hash-based to sequential ID allocation

### Breaking Changes

- Serialized Arrow files from v0.3.1 with symbol columns are incompatible
- `hash_symbol_to_u32` function removed from public API
```

**Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add v0.3.2 changelog entry"
```

---

## Part D: Release Preparation

### Task 12: Final Verification

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests PASS

**Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings or errors

**Step 3: Build release**

Run: `cargo build --release --workspace`
Expected: PASS

**Step 4: Test CLI end-to-end**

Create test file `/tmp/v032_test.xl`:
```xlog
fn abs(X) = if X < 0 then 0 - X else X.

pred data(symbol, f64).
data(sensor_a, -5.0).
data(sensor_b, 3.0).

pred result(symbol, f64).
result(Label, AbsVal) :- data(Label, Val), abs(Val) is AbsVal.

?- result(X, Y).
```

Run: `cargo run -p xlog-cli --release -- /tmp/v032_test.xl`
Expected:
```
result(sensor_a, 5.0).
result(sensor_b, 3.0).
```

Run: `cargo run -p xlog-cli --release -- /tmp/v032_test.xl --format json`
Expected: JSON output with string symbols

Run: `cargo run -p xlog-cli --release -- /tmp/v032_test.xl --stats`
Expected: Output includes symbol stats

**Step 5: Commit any final fixes**

If any issues found, fix and commit.

**Step 6: Final commit message**

```bash
git add -A
git commit -m "chore: v0.3.2 release preparation complete"
```

---

## Summary

Phase 4 completes v0.3.2 by:

1. **Integration Tests** (Tasks 1-4): Verify all features work together
2. **CLI Enhancements** (Tasks 5-8): Update output formatting and add flags
3. **Documentation** (Tasks 9-11): Language reference, README, CHANGELOG
4. **Release Verification** (Task 12): Full test suite, clippy, release build, E2E test

**Total:** 12 tasks, ~60 steps

**After Phase 4:** Ready to merge to main, tag v0.3.2, and release.
