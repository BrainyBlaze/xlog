//! Global symbol interning for reversible string-to-ID mapping.

use arrow::array::{Array, DictionaryArray, StringArray, UInt32Array};
use arrow::datatypes::UInt32Type;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

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
    resolve_checked(id).expect("invalid symbol ID: this is a bug")
}

/// Resolve an ID to its string if present.
pub fn resolve_checked(id: u32) -> Option<String> {
    let reg = registry().read().unwrap();
    reg.to_string.get(id as usize).cloned()
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
    let map_overhead =
        reg.to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>());
    string_bytes + map_overhead
}

/// Convert a column of symbol IDs to Arrow DictionaryArray.
pub fn to_arrow(ids: &[u32]) -> DictionaryArray<UInt32Type> {
    use std::collections::HashSet;

    // Collect unique IDs preserving order
    let mut seen = HashSet::new();
    let unique_ids: Vec<u32> = ids.iter().filter(|id| seen.insert(**id)).copied().collect();

    // Build string dictionary
    let dict_strings: Vec<String> = unique_ids.iter().map(|&id| resolve(id)).collect();
    let dictionary = StringArray::from(dict_strings);

    // Map original IDs to dictionary indices
    let id_to_index: HashMap<u32, u32> = unique_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();

    let keys: Vec<u32> = ids.iter().map(|id| *id_to_index.get(id).unwrap()).collect();
    let keys_array = UInt32Array::from(keys);

    DictionaryArray::try_new(keys_array, Arc::new(dictionary)).unwrap()
}

/// Convert Arrow DictionaryArray back to symbol IDs.
pub fn from_arrow(arr: &DictionaryArray<UInt32Type>) -> Vec<u32> {
    let dict = arr
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values must be StringArray");

    // Intern all dictionary values
    let dict_to_symbol: Vec<u32> = dict
        .iter()
        .map(|s| intern(s.expect("null not supported in symbols")))
        .collect();

    // Map keys through dictionary
    arr.keys()
        .iter()
        .map(|k| {
            let idx = k.expect("null keys not supported") as usize;
            dict_to_symbol[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // Each test must call setup() to get clean state
    fn setup() {
        clear();
    }

    #[test]
    #[serial]
    fn test_intern_sequential() {
        setup();
        assert_eq!(intern("foo"), 0);
        assert_eq!(intern("bar"), 1);
        assert_eq!(intern("baz"), 2);
    }

    #[test]
    #[serial]
    fn test_intern_idempotent() {
        setup();
        let id1 = intern("hello");
        let id2 = intern("hello");
        assert_eq!(id1, id2);
        assert_eq!(count(), 1); // only one entry
    }

    #[test]
    #[serial]
    fn test_resolve_roundtrip() {
        setup();
        let id = intern("world");
        assert_eq!(resolve(id), "world");
    }

    #[test]
    #[serial]
    #[should_panic(expected = "invalid symbol ID")]
    fn test_resolve_invalid() {
        setup();
        resolve(9999); // should panic
    }

    #[test]
    #[serial]
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
    #[serial]
    fn test_empty_string() {
        setup();
        let id = intern("");
        assert_eq!(resolve(id), "");
    }

    #[test]
    #[serial]
    fn test_unicode() {
        setup();
        let id = intern("日本語");
        assert_eq!(resolve(id), "日本語");

        let id2 = intern("émoji🎉");
        assert_eq!(resolve(id2), "émoji🎉");
    }

    #[test]
    #[serial]
    fn test_concurrent_intern() {
        setup();
        use std::collections::HashSet;
        use std::thread;

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

    #[test]
    #[serial]
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
        println!(
            "100K intern: {:?}, 100K resolve: {:?}",
            intern_time, resolve_time
        );

        // Memory should be reasonable (rough check: < 10MB for 100K symbols)
        let mem = memory_usage();
        assert!(mem < 10_000_000, "memory usage {} exceeds 10MB", mem);
    }

    #[test]
    #[serial]
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
}
