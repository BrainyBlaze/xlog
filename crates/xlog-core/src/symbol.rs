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
    let owned = s.to_string();
    reg.to_id.insert(owned.clone(), id);
    reg.to_string.push(owned);
    id
}

/// Resolve an ID to its string. Panics if ID is invalid.
pub fn resolve(id: u32) -> String {
    let reg = registry().read().unwrap();
    reg.to_string
        .get(id as usize)
        .cloned()
        .expect("invalid symbol ID: this is a bug")
}

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

#[cfg(test)]
mod tests {
    use super::*;

    // Each test must call setup() to get clean state
    fn setup() {
        clear();
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
        let id1 = intern("hello");
        let id2 = intern("hello");
        assert_eq!(id1, id2);
        assert_eq!(count(), 1); // only one entry
    }

    #[test]
    fn test_resolve_roundtrip() {
        setup();
        let id = intern("world");
        assert_eq!(resolve(id), "world");
    }

    #[test]
    #[should_panic(expected = "invalid symbol ID")]
    fn test_resolve_invalid() {
        setup();
        resolve(9999); // should panic
    }

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

    #[test]
    fn test_empty_string() {
        setup();
        let id = intern("");
        assert_eq!(resolve(id), "");
    }

    #[test]
    fn test_unicode() {
        setup();
        let id = intern("日本語");
        assert_eq!(resolve(id), "日本語");

        let id2 = intern("émoji🎉");
        assert_eq!(resolve(id2), "émoji🎉");
    }
}
