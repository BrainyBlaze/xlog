# Phase 1: Reversible Symbols Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace one-way symbol hashing with a global intern table enabling bidirectional string↔ID lookup.

**Architecture:** Global singleton `SymbolRegistry` with RwLock for thread safety. Sequential ID allocation. Panics on invalid resolve (bugs should crash).

**Tech Stack:** Rust std library (RwLock, OnceLock, HashMap), Arrow dictionary encoding

**Design Document:** `docs/plans/2026-01-18-reversible-symbols-design.md`

**Depends On:** Nothing (standalone)

**Estimated Tasks:** 15 tasks, ~75 steps

---

## Task 1: Create Symbol Registry Module

**Files:**
- Create: `crates/xlog-core/src/symbol.rs`

**Step 1: Create symbol.rs with SymbolRegistry struct**

```rust
//! Global symbol interning for reversible string-to-ID mapping.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

static REGISTRY: OnceLock<RwLock<SymbolRegistry>> = OnceLock::new();

struct SymbolRegistry {
    to_id: HashMap<String, u32>,
    to_string: Vec<String>,
}

impl SymbolRegistry {
    fn new() -> Self {
        Self {
            to_id: HashMap::new(),
            to_string: Vec::new(),
        }
    }
}

fn registry() -> &'static RwLock<SymbolRegistry> {
    REGISTRY.get_or_init(|| RwLock::new(SymbolRegistry::new()))
}
```

**Step 2: Add intern() function**

```rust
/// Intern a string, returning its unique ID.
/// Thread-safe. Returns existing ID if already interned.
pub fn intern(s: &str) -> u32 {
    // Fast path: check if already interned (read lock)
    {
        let reg = registry().read().unwrap();
        if let Some(&id) = reg.to_id.get(s) {
            return id;
        }
    }
    // Slow path: insert new (write lock)
    let mut reg = registry().write().unwrap();
    // Double-check after acquiring write lock
    if let Some(&id) = reg.to_id.get(s) {
        return id;
    }
    let id = reg.to_string.len() as u32;
    reg.to_string.push(s.to_string());
    reg.to_id.insert(s.to_string(), id);
    id
}
```

**Step 3: Add resolve() function**

```rust
/// Resolve an ID to its string. Panics if ID is invalid.
pub fn resolve(id: u32) -> String {
    let reg = registry().read().unwrap();
    reg.to_string
        .get(id as usize)
        .cloned()
        .expect("invalid symbol ID: this is a bug")
}
```

**Step 4: Add clear() and count() functions**

```rust
/// Clear all symbols. For testing/REPL only.
/// WARNING: Invalidates all existing symbol IDs.
pub fn clear() {
    let mut reg = registry().write().unwrap();
    reg.to_id.clear();
    reg.to_string.clear();
}

/// Number of interned symbols.
pub fn count() -> usize {
    registry().read().unwrap().to_string.len()
}

/// Estimated memory usage in bytes.
pub fn memory_usage() -> usize {
    let reg = registry().read().unwrap();
    let string_bytes: usize = reg.to_string.iter().map(|s| s.len()).sum();
    let map_overhead = reg.to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>());
    string_bytes + map_overhead
}
```

**Step 5: Verify file compiles**

Run: `cargo build -p xlog-core 2>&1 | head -20`
Expected: Error about module not exported (expected)

---

## Task 2: Add Basic Unit Tests

**Files:**
- Modify: `crates/xlog-core/src/symbol.rs`

**Step 1: Add test module with setup helper**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Each test must call setup() to get clean state
    fn setup() {
        clear();
    }
```

**Step 2: Add test_intern_sequential**

```rust
    #[test]
    fn test_intern_sequential() {
        setup();
        assert_eq!(intern("foo"), 0);
        assert_eq!(intern("bar"), 1);
        assert_eq!(intern("baz"), 2);
    }
```

**Step 3: Add test_intern_idempotent**

```rust
    #[test]
    fn test_intern_idempotent() {
        setup();
        let id1 = intern("hello");
        let id2 = intern("hello");
        assert_eq!(id1, id2);
        assert_eq!(count(), 1); // only one entry
    }
```

**Step 4: Add test_resolve_roundtrip**

```rust
    #[test]
    fn test_resolve_roundtrip() {
        setup();
        let id = intern("world");
        assert_eq!(resolve(id), "world");
    }
```

**Step 5: Add test_resolve_invalid**

```rust
    #[test]
    #[should_panic(expected = "invalid symbol ID")]
    fn test_resolve_invalid() {
        setup();
        resolve(9999); // should panic
    }
```

**Step 6: Add test_clear**

```rust
    #[test]
    fn test_clear() {
        setup();
        intern("a");
        intern("b");
        assert_eq!(count(), 2);
        clear();
        assert_eq!(count(), 0);
        assert_eq!(intern("a"), 0); // IDs restart from 0
    }
```

**Step 7: Add test_empty_string**

```rust
    #[test]
    fn test_empty_string() {
        setup();
        let id = intern("");
        assert_eq!(resolve(id), "");
    }
```

**Step 8: Add test_unicode**

```rust
    #[test]
    fn test_unicode() {
        setup();
        let id = intern("日本語");
        assert_eq!(resolve(id), "日本語");

        let id2 = intern("émoji🎉");
        assert_eq!(resolve(id2), "émoji🎉");
    }
```

**Step 9: Close test module**

```rust
}
```

---

## Task 3: Export Symbol Module

**Files:**
- Modify: `crates/xlog-core/src/lib.rs`

**Step 1: Add symbol module export**

Add after other module declarations:

```rust
pub mod symbol;
```

**Step 2: Run tests**

Run: `cargo test -p xlog-core symbol -- --test-threads=1`
Expected: 8 tests pass

Note: `--test-threads=1` required because tests share global state.

**Step 3: Commit**

```bash
git add crates/xlog-core/src/symbol.rs crates/xlog-core/src/lib.rs
git commit -m "feat(core): add symbol registry with intern/resolve API

- Global thread-safe intern table with RwLock
- Sequential ID allocation (0, 1, 2, ...)
- Panics on invalid resolve (catches bugs)
- clear() for testing, count() and memory_usage() for stats"
```

---

## Task 4: Add Concurrent Test

**Files:**
- Modify: `crates/xlog-core/src/symbol.rs`

**Step 1: Add concurrent intern test**

Add to tests module:

```rust
    #[test]
    fn test_concurrent_intern() {
        setup();
        use std::thread;
        use std::collections::HashSet;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let mut ids = Vec::new();
                    for j in 0..100 {
                        let s = format!("thread{}_{}", i, j);
                        let id = intern(&s);
                        ids.push((s, id));
                    }
                    ids
                })
            })
            .collect();

        let mut all_results = Vec::new();
        for h in handles {
            all_results.extend(h.join().unwrap());
        }

        // Verify all symbols resolve correctly
        for (s, id) in &all_results {
            assert_eq!(&resolve(*id), s);
        }

        // Verify we have 1000 unique symbols
        assert_eq!(count(), 1000);

        // Verify no duplicate IDs for different strings
        let unique_ids: HashSet<u32> = all_results.iter().map(|(_, id)| *id).collect();
        assert_eq!(unique_ids.len(), 1000);
    }
```

**Step 2: Run concurrent test**

Run: `cargo test -p xlog-core test_concurrent -- --test-threads=1`
Expected: PASS

**Step 3: Add large scale test (100K symbols)**

Add to tests module:

```rust
    #[test]
    fn test_large_scale() {
        setup();
        use std::time::Instant;

        let start = Instant::now();

        // Intern 100K unique symbols
        for i in 0..100_000 {
            let s = format!("symbol_{:06}", i);
            let id = intern(&s);
            assert_eq!(id, i as u32);
        }

        let intern_time = start.elapsed();

        // Verify all resolve correctly
        let start = Instant::now();
        for i in 0..100_000 {
            let expected = format!("symbol_{:06}", i);
            assert_eq!(resolve(i as u32), expected);
        }
        let resolve_time = start.elapsed();

        // Verify count
        assert_eq!(count(), 100_000);

        // Log performance (not assertions, just info)
        println!("100K intern: {:?}, 100K resolve: {:?}", intern_time, resolve_time);

        // Memory should be reasonable (rough check: < 10MB for 100K symbols)
        let mem = memory_usage();
        assert!(mem < 10_000_000, "memory usage {} exceeds 10MB", mem);
    }
```

**Step 4: Run large scale test**

Run: `cargo test -p xlog-core test_large_scale -- --test-threads=1 --nocapture`
Expected: PASS with timing output

**Step 5: Commit**

```bash
git add crates/xlog-core/src/symbol.rs
git commit -m "test(core): add concurrent and large-scale symbol tests"
```

---

## Task 5: Remove Old Hash Function

**Files:**
- Modify: `crates/xlog-core/src/types.rs`
- Modify: `crates/xlog-core/src/lib.rs`

**Step 1: Delete hash_symbol_to_u32 from types.rs**

In `crates/xlog-core/src/types.rs`, find and delete:

```rust
/// Hash a symbol string into its on-device `u32` representation.
///
/// Notes:
/// - This is an MVP encoding used consistently across compilation/runtime.
/// - It is not reversible and is not collision-free.
pub fn hash_symbol_to_u32(s: &str) -> u32 {
    s.bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
}
```

**Step 2: Remove from lib.rs exports**

In `crates/xlog-core/src/lib.rs`, change:

```rust
pub use types::{hash_symbol_to_u32, ScalarType, Schema, RelId, AggOp};
```

To:

```rust
pub use types::{ScalarType, Schema, RelId, AggOp};
```

**Step 3: Build to find broken call sites**

Run: `cargo build 2>&1 | grep -E "(hash_symbol|cannot find)" | head -20`
Expected: List of files with errors

**Step 4: Commit removal**

```bash
git add crates/xlog-core/src/types.rs crates/xlog-core/src/lib.rs
git commit -m "refactor(core): remove hash_symbol_to_u32 function

Breaking change: all call sites must migrate to symbol::intern()"
```

---

## Task 6: Update xlog-runtime Call Sites

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`
- Modify: `crates/xlog-runtime/tests/executor_config_tests.rs`

**Step 1: Find usages in executor.rs**

Run: `grep -n "hash_symbol" crates/xlog-runtime/src/executor.rs`

**Step 2: Update imports in executor.rs**

Replace:
```rust
use xlog_core::hash_symbol_to_u32;
```

With:
```rust
use xlog_core::symbol;
```

**Step 3: Replace hash calls with intern in executor.rs**

Replace all occurrences of:
```rust
hash_symbol_to_u32(s)
```

With:
```rust
symbol::intern(s)
```

**Step 4: Update executor_config_tests.rs similarly**

Apply same changes to test file.

**Step 5: Build xlog-runtime**

Run: `cargo build -p xlog-runtime`
Expected: PASS

**Step 6: Run xlog-runtime tests**

Run: `cargo test -p xlog-runtime`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs crates/xlog-runtime/tests/executor_config_tests.rs
git commit -m "refactor(runtime): migrate from hash_symbol_to_u32 to symbol::intern"
```

---

## Task 7: Update xlog-gpu Call Sites

**Files:**
- Modify: `crates/xlog-gpu/src/logic.rs`

**Step 1: Find usages**

Run: `grep -n "hash_symbol" crates/xlog-gpu/src/logic.rs`

**Step 2: Update imports**

Replace hash_symbol_to_u32 import with:
```rust
use xlog_core::symbol;
```

**Step 3: Replace hash calls**

Replace `hash_symbol_to_u32(s)` with `symbol::intern(s)`

**Step 4: Build and test**

Run: `cargo build -p xlog-gpu && cargo test -p xlog-gpu`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-gpu/src/logic.rs
git commit -m "refactor(gpu): migrate from hash_symbol_to_u32 to symbol::intern"
```

---

## Task 8: Update xlog-prob Call Sites

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs`

**Step 1: Find usages**

Run: `grep -n "hash_symbol" crates/xlog-prob/src/provenance.rs`

**Step 2: Update imports and replace calls**

Same pattern as previous tasks.

**Step 3: Build and test**

Run: `cargo build -p xlog-prob && cargo test -p xlog-prob`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs
git commit -m "refactor(prob): migrate from hash_symbol_to_u32 to symbol::intern"
```

---

## Task 9: Update xlog-logic Call Sites

**Files:**
- Modify: `crates/xlog-logic/examples/xlog_run.rs`

**Step 1: Find usages**

Run: `grep -n "hash_symbol" crates/xlog-logic/examples/xlog_run.rs`

**Step 2: Update imports and replace calls**

**Step 3: Build**

Run: `cargo build -p xlog-logic --examples`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/examples/xlog_run.rs
git commit -m "refactor(logic): migrate example from hash_symbol_to_u32 to symbol::intern"
```

---

## Task 10: Verify Full Build

**Step 1: Clean build**

Run: `cargo build --all`
Expected: PASS with no hash_symbol_to_u32 errors

**Step 2: Run all tests**

Run: `cargo test --all`
Expected: PASS

**Step 3: Commit checkpoint**

```bash
git commit --allow-empty -m "checkpoint: all call sites migrated to symbol::intern"
```

---

## Task 11: Update Parser to Intern Symbols

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Find where symbols are created**

Run: `grep -n "Term::Symbol" crates/xlog-logic/src/parser.rs`

**Step 2: Add import**

```rust
use xlog_core::symbol;
```

**Step 3: Update symbol parsing**

Where symbols are created from string literals, ensure they're interned:

```rust
// When parsing a symbol literal (lowercase identifier used as constant)
Term::Symbol(symbol::intern(s))
```

Note: The AST stores `Term::Symbol(u32)` - the interned ID, not the string.

**Step 4: Build and test**

Run: `cargo build -p xlog-logic && cargo test -p xlog-logic`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): intern symbols during parsing"
```

---

## Task 12: Update Arrow Serialization

**Files:**
- Modify: `crates/xlog-core/src/types.rs`

**Step 1: Update to_arrow_type for Symbol**

Change the Symbol case:

```rust
pub fn to_arrow_type(&self) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType;
    match self {
        ScalarType::Bool => DataType::Boolean,
        ScalarType::U32 => DataType::UInt32,
        ScalarType::I32 => DataType::Int32,
        ScalarType::U64 => DataType::UInt64,
        ScalarType::I64 => DataType::Int64,
        ScalarType::F32 => DataType::Float32,
        ScalarType::F64 => DataType::Float64,
        ScalarType::Symbol => DataType::Dictionary(
            Box::new(DataType::UInt32),
            Box::new(DataType::Utf8),
        ),
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-core`
Expected: PASS (may have test failures)

**Step 3: Fix any broken Arrow tests**

Tests expecting `UInt32` for Symbol now need to expect `Dictionary`.

**Step 4: Commit**

```bash
git add crates/xlog-core/src/types.rs
git commit -m "feat(core): use Arrow dictionary encoding for Symbol type"
```

---

## Task 13: Add Arrow Conversion Functions

**Files:**
- Modify: `crates/xlog-core/src/symbol.rs`

**Step 1: Add arrow imports**

```rust
use arrow::array::{Array, DictionaryArray, StringArray, UInt32Array};
use arrow::datatypes::UInt32Type;
use std::collections::HashSet;
use std::sync::Arc;
```

**Step 2: Add symbol_column_to_arrow**

```rust
/// Convert a column of symbol IDs to Arrow DictionaryArray.
pub fn to_arrow(ids: &[u32]) -> DictionaryArray<UInt32Type> {
    // Collect unique IDs preserving order
    let mut seen = HashSet::new();
    let unique_ids: Vec<u32> = ids.iter()
        .filter(|id| seen.insert(**id))
        .copied()
        .collect();

    // Build string dictionary
    let dict_strings: Vec<String> = unique_ids.iter()
        .map(|&id| resolve(id))
        .collect();
    let dictionary = StringArray::from(dict_strings);

    // Map original IDs to dictionary indices
    let id_to_index: HashMap<u32, u32> = unique_ids.iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();

    let keys: Vec<u32> = ids.iter()
        .map(|id| *id_to_index.get(id).unwrap())
        .collect();
    let keys_array = UInt32Array::from(keys);

    DictionaryArray::try_new(keys_array, Arc::new(dictionary)).unwrap()
}
```

**Step 3: Add arrow_to_symbol_column**

```rust
/// Convert Arrow DictionaryArray back to symbol IDs.
pub fn from_arrow(arr: &DictionaryArray<UInt32Type>) -> Vec<u32> {
    let dict = arr.values().as_any().downcast_ref::<StringArray>()
        .expect("dictionary values must be StringArray");

    // Intern all dictionary values
    let dict_to_symbol: Vec<u32> = dict.iter()
        .map(|s| intern(s.expect("null not supported in symbols")))
        .collect();

    // Map keys through dictionary
    arr.keys().iter()
        .map(|k| {
            let idx = k.expect("null keys not supported") as usize;
            dict_to_symbol[idx]
        })
        .collect()
}
```

**Step 4: Add Arrow round-trip test**

```rust
    #[test]
    fn test_arrow_roundtrip() {
        setup();
        let ids = vec![
            intern("apple"),
            intern("banana"),
            intern("apple"),
            intern("cherry"),
            intern("banana"),
        ];

        let arrow = to_arrow(&ids);
        let back = from_arrow(&arrow);

        assert_eq!(ids, back);
    }
```

**Step 5: Build and test**

Run: `cargo test -p xlog-core symbol -- --test-threads=1`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-core/src/symbol.rs
git commit -m "feat(core): add Arrow dictionary serialization for symbols"
```

---

## Task 14: Add CLI Output Formatting

**Files:**
- Modify: `crates/xlog-cli/src/output.rs` (or create if doesn't exist)

**Step 1: Check if output.rs exists**

Run: `ls crates/xlog-cli/src/`

**Step 2: Add or update format_value function**

```rust
use xlog_core::{ScalarType, symbol};

/// Format a value for display based on its type
pub fn format_value(value: u64, typ: ScalarType) -> String {
    match typ {
        ScalarType::Symbol => symbol::resolve(value as u32),
        ScalarType::U32 => (value as u32).to_string(),
        ScalarType::U64 => value.to_string(),
        ScalarType::I32 => (value as i32).to_string(),
        ScalarType::I64 => (value as i64).to_string(),
        ScalarType::F32 => f32::from_bits(value as u32).to_string(),
        ScalarType::F64 => f64::from_bits(value).to_string(),
        ScalarType::Bool => if value != 0 { "true" } else { "false" }.to_string(),
    }
}
```

**Step 3: Update JSON output to use strings for symbols**

Ensure JSON serialization converts symbol IDs to strings.

**Step 4: Build and test**

Run: `cargo build -p xlog-cli`

**Step 5: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): display symbols as strings in output"
```

---

## Task 15: Add Stats Output for Symbols

**Files:**
- Modify: `crates/xlog-cli/src/main.rs` or stats output location

**Step 1: Find stats output code**

Run: `grep -r "stats" crates/xlog-cli/src/ | grep -i output`

**Step 2: Add symbol stats to --stats output**

```rust
use xlog_core::symbol;

// In stats output section:
println!("Symbols: {} interned ({} bytes)",
    symbol::count(),
    symbol::memory_usage());
```

**Step 3: Build and test manually**

Run: `cargo build -p xlog-cli`

Run a query with --stats and verify symbol count appears.

**Step 4: Commit**

```bash
git add crates/xlog-cli/
git commit -m "feat(cli): show symbol stats in --stats output"
```

---

## Task 16: Phase 1 Integration Test

**Files:**
- Create: `crates/xlog-logic/tests/symbol_integration_test.rs`

**Step 1: Create integration test**

```rust
//! Integration test for reversible symbols

use xlog_core::symbol;
use xlog_logic::parse;

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

    let program = parse(src).unwrap();

    // Verify symbols were interned
    assert!(symbol::count() >= 4); // alice, bob, engineer, manager, developer (person is predicate)

    // Verify we can resolve them back
    // (actual verification depends on how facts are stored)
}

#[test]
fn test_symbol_deduplication() {
    setup();

    let src = r#"
        edge(a, b).
        edge(b, c).
        edge(a, c).
    "#;

    let _ = parse(src).unwrap();

    // a, b, c should each be interned once
    // exact count depends on what else gets interned
    let count_before = symbol::count();

    // Parse same source again - should not increase count
    let _ = parse(src).unwrap();

    assert_eq!(symbol::count(), count_before);
}
```

**Step 2: Run integration tests**

Run: `cargo test -p xlog-logic symbol_integration -- --test-threads=1`
Expected: PASS

**Step 3: Final commit for Phase 1**

```bash
git add crates/xlog-logic/tests/symbol_integration_test.rs
git commit -m "test(logic): add symbol integration tests

Phase 1 (Reversible Symbols) complete:
- Global intern table with sequential IDs
- Thread-safe with RwLock
- Arrow dictionary serialization
- CLI displays symbols as strings
- Stats shows symbol count and memory"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create symbol registry | symbol.rs |
| 2 | Add unit tests | symbol.rs |
| 3 | Export module | lib.rs |
| 4 | Add concurrent test | symbol.rs |
| 5 | Remove old hash | types.rs, lib.rs |
| 6 | Update xlog-runtime | executor.rs, tests |
| 7 | Update xlog-gpu | logic.rs |
| 8 | Update xlog-prob | provenance.rs |
| 9 | Update xlog-logic | xlog_run.rs |
| 10 | Verify full build | - |
| 11 | Update parser | parser.rs |
| 12 | Update Arrow schema | types.rs |
| 13 | Add Arrow conversion | symbol.rs |
| 14 | Add CLI formatting | output.rs |
| 15 | Add stats output | main.rs |
| 16 | Integration test | symbol_integration_test.rs |

**Total: 16 tasks, ~80 steps**
