# Reversible Symbol Values Design (v0.3.2)

> **Status:** Approved
> **Author:** Claude + Human
> **Date:** 2026-01-18
> **Target:** v0.3.2 (Language Core)

---

## Overview

This document specifies reversible symbol values for XLOG, replacing the current one-way hash with a global intern table that enables `u32` → `String` lookup. This allows readable output and proper Arrow serialization.

---

## Core Data Structure

### Symbol Registry

A global, thread-safe intern table mapping strings to sequential IDs:

```rust
use std::sync::{RwLock, OnceLock};
use std::collections::HashMap;

static REGISTRY: OnceLock<RwLock<SymbolRegistry>> = OnceLock::new();

struct SymbolRegistry {
    to_id: HashMap<String, u32>,
    to_string: Vec<String>,  // index = id
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

### Public API

```rust
/// Intern a string, returning its unique ID.
/// Thread-safe. Returns existing ID if already interned.
pub fn intern(s: &str) -> u32 {
    let mut reg = registry().write().unwrap();
    if let Some(&id) = reg.to_id.get(s) {
        return id;
    }
    let id = reg.to_string.len() as u32;
    reg.to_string.push(s.to_string());
    reg.to_id.insert(s.to_string(), id);
    id
}

/// Resolve an ID to its string. Panics if ID is invalid.
pub fn resolve(id: u32) -> &'static str {
    let reg = registry().read().unwrap();
    reg.to_string.get(id as usize)
        .map(|s| s.as_str())
        .expect("invalid symbol ID")
}

/// Clear all symbols. For testing/REPL only.
pub fn clear() {
    let mut reg = registry().write().unwrap();
    reg.to_id.clear();
    reg.to_string.clear();
}

/// Number of interned symbols.
pub fn count() -> usize {
    registry().read().unwrap().to_string.len()
}
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Global singleton | Simple API, symbols work across modules/queries |
| ID allocation | Sequential | Compact, predictable, easy debugging |
| Thread safety | RwLock | Good read performance, safe concurrent reads |
| Resolve failure | Panic | Invalid ID is a bug, catches errors immediately |
| Reset capability | Explicit clear() | Testable, REPL-friendly |

---

## Migration

### Removing `hash_symbol_to_u32`

The current hash function in `xlog-core/src/types.rs` will be removed:

```rust
// DELETE this function
pub fn hash_symbol_to_u32(s: &str) -> u32 {
    s.bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
}
```

### Call Sites to Update

| File | Change |
|------|--------|
| `xlog-core/src/lib.rs` | Remove `hash_symbol_to_u32` export, add `symbol` module |
| `xlog-logic/examples/xlog_run.rs` | Replace with `symbol::intern()` |
| `xlog-runtime/src/executor.rs` | Replace with `symbol::intern()` |
| `xlog-runtime/tests/executor_config_tests.rs` | Replace with `symbol::intern()` |
| `xlog-gpu/src/logic.rs` | Replace with `symbol::intern()` |
| `xlog-prob/src/provenance.rs` | Replace with `symbol::intern()` |

### Parser Integration

Symbols are interned during parsing:

```rust
// In parser.rs, when handling Term::Symbol
Rule::ident => {
    let s = pair.as_str();
    // Check if it's a symbol literal (lowercase identifier as constant)
    Term::Symbol(symbol::intern(s))
}
```

### Backward Compatibility

None. This is a breaking change for serialized data. Old Arrow files with hashed symbol columns will not be readable. This is acceptable for v0.3.2 as we're still pre-1.0.

---

## Output Formatting

### Display in CLI

Query results show original strings instead of numeric IDs:

```
% Before (current)
?- person(X, Y).
person(12345, 67890).
person(12345, 11111).

% After (with reversible symbols)
?- person(X, Y).
person(alice, engineer).
person(alice, manager).
```

### Output Formatting API

```rust
/// Format a symbol value for display
pub fn format_symbol(id: u32) -> String {
    resolve(id).to_string()
}

/// Format a row value, handling symbols specially
pub fn format_value(value: &Value, typ: ScalarType) -> String {
    match typ {
        ScalarType::Symbol => format_symbol(value.as_u32()),
        ScalarType::U32 => value.as_u32().to_string(),
        ScalarType::F64 => value.as_f64().to_string(),
        // ... other types
    }
}
```

### JSON Output

With `--format json`, symbols serialize as strings:

```json
{
  "relation": "person",
  "rows": [
    {"col0": "alice", "col1": "engineer"},
    {"col0": "alice", "col1": "manager"}
  ]
}
```

### Stats Output

The `--stats` flag shows symbol table metrics:

```
Symbols: 1,234 interned (48 KB)
```

---

## Arrow Serialization

### Dictionary Encoding

Symbol columns use Arrow's native dictionary type:

```rust
use arrow::array::{DictionaryArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, UInt32Type};

/// Convert a symbol column to Arrow DictionaryArray
pub fn symbol_column_to_arrow(ids: &[u32]) -> DictionaryArray<UInt32Type> {
    // Collect unique IDs and build dictionary
    let unique_ids: Vec<u32> = ids.iter().copied().collect::<HashSet<_>>()
        .into_iter().collect();

    // Build string dictionary from symbol registry
    let dict_strings: Vec<&str> = unique_ids.iter()
        .map(|&id| resolve(id))
        .collect();
    let dictionary = StringArray::from(dict_strings);

    // Map original IDs to dictionary indices
    let id_to_index: HashMap<u32, u32> = unique_ids.iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();
    let keys: Vec<u32> = ids.iter()
        .map(|id| id_to_index[id])
        .collect();
    let keys_array = UInt32Array::from(keys);

    DictionaryArray::try_new(keys_array, Arc::new(dictionary)).unwrap()
}
```

### Schema Representation

Symbol columns have dictionary type in Arrow schema:

```rust
impl ScalarType {
    pub fn to_arrow_type(&self) -> DataType {
        match self {
            // ... other types
            ScalarType::Symbol => DataType::Dictionary(
                Box::new(DataType::UInt32),
                Box::new(DataType::Utf8),
            ),
        }
    }
}
```

### Import from Arrow

Reading dictionary columns back:

```rust
/// Convert Arrow DictionaryArray back to symbol IDs
pub fn arrow_to_symbol_column(arr: &DictionaryArray<UInt32Type>) -> Vec<u32> {
    let dict = arr.values().as_any().downcast_ref::<StringArray>().unwrap();

    // Intern all dictionary values
    let dict_to_symbol: Vec<u32> = dict.iter()
        .map(|s| symbol::intern(s.unwrap()))
        .collect();

    // Map keys through dictionary
    arr.keys().iter()
        .map(|k| dict_to_symbol[k.unwrap() as usize])
        .collect()
}
```

---

## Testing Strategy

### Test Cases

| Category | Test |
|----------|------|
| Basic intern | `intern("foo")` returns 0, `intern("bar")` returns 1 |
| Idempotent | `intern("foo")` twice returns same ID |
| Resolve | `resolve(intern("foo"))` returns "foo" |
| Invalid resolve | `resolve(9999)` panics |
| Clear | After `clear()`, `intern("foo")` returns 0 again |
| Thread safety | Concurrent interns don't corrupt state |
| Empty string | `intern("")` works correctly |
| Unicode | `intern("日本語")` round-trips correctly |
| Large scale | 100K symbols don't degrade performance |

### Test Module

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        clear();  // clean slate for each test
    }

    #[test]
    fn test_intern_sequential() {
        setup();
        assert_eq!(intern("foo"), 0);
        assert_eq!(intern("bar"), 1);
        assert_eq!(intern("baz"), 2);
    }

    #[test]
    fn test_intern_idempotent() {
        setup();
        let id1 = intern("foo");
        let id2 = intern("foo");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_resolve_roundtrip() {
        setup();
        let id = intern("hello");
        assert_eq!(resolve(id), "hello");
    }

    #[test]
    #[should_panic(expected = "invalid symbol ID")]
    fn test_resolve_invalid() {
        setup();
        resolve(9999);
    }

    #[test]
    fn test_unicode() {
        setup();
        let id = intern("日本語");
        assert_eq!(resolve(id), "日本語");
    }

    #[test]
    fn test_concurrent_intern() {
        setup();
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    for j in 0..100 {
                        intern(&format!("sym_{}_{}", i, j));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All symbols should resolve
        // (exact IDs non-deterministic due to concurrency)
    }
}
```

### Arrow Round-trip Tests

```rust
#[test]
fn test_arrow_roundtrip() {
    setup();
    let ids = vec![intern("a"), intern("b"), intern("a"), intern("c")];
    let arrow = symbol_column_to_arrow(&ids);
    let back = arrow_to_symbol_column(&arrow);
    assert_eq!(ids, back);
}
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `crates/xlog-core/src/symbol.rs` | `SymbolRegistry`, `intern()`, `resolve()`, `clear()` |

### Files to Modify

| File | Changes |
|------|---------|
| `xlog-core/src/lib.rs` | Add `pub mod symbol`, remove `hash_symbol_to_u32` export |
| `xlog-core/src/types.rs` | Remove `hash_symbol_to_u32`, update `ScalarType::to_arrow_type()` |
| `xlog-logic/src/parser.rs` | Use `symbol::intern()` for symbol literals |
| `xlog-logic/examples/xlog_run.rs` | Replace hash calls with intern |
| `xlog-runtime/src/executor.rs` | Replace hash calls with intern |
| `xlog-runtime/tests/executor_config_tests.rs` | Replace hash calls with intern |
| `xlog-gpu/src/logic.rs` | Replace hash calls with intern |
| `xlog-prob/src/provenance.rs` | Replace hash calls with intern |
| `xlog-cli/src/output.rs` | Use `symbol::resolve()` for display |

### Implementation Order

1. Create `symbol.rs` with core API
2. Add tests for intern/resolve/clear
3. Update `xlog-core/src/lib.rs` exports
4. Remove `hash_symbol_to_u32` from types.rs
5. Update all call sites (7 files)
6. Update Arrow serialization in types.rs
7. Add Arrow round-trip tests
8. Update CLI output formatting
9. Add `--stats` symbol metrics
10. Integration tests with end-to-end queries

### Dependencies

- No new external crates
- Uses `std::sync::{RwLock, OnceLock}`
- Uses existing Arrow dependency

### Breaking Changes

- Old serialized Arrow files with hashed symbols incompatible
- `hash_symbol_to_u32` removed from public API
- Symbol IDs now sequential (0, 1, 2...) not hash-based

---

## Summary

Reversible symbols provide:

- **Global intern table** — Thread-safe singleton with RwLock
- **Sequential IDs** — Compact, predictable allocation
- **Bidirectional lookup** — `intern()` and `resolve()` APIs
- **Readable output** — CLI shows original strings
- **Arrow dictionary encoding** — Native, compact serialization
- **Testable** — `clear()` for test isolation
- **No external dependencies** — Uses std library only
