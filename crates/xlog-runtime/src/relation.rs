//! Relation management for GPU-based Datalog execution
//!
//! This module provides [`RelationStore`], a container for managing named relations
//! stored as GPU buffers. It provides CRUD operations for relations during query
//! execution.

use std::collections::HashMap;

use xlog_core::Schema;
use xlog_cuda::CudaBuffer;

/// Storage for named relations as GPU buffers
///
/// `RelationStore` manages a collection of named relations, each stored as a
/// [`CudaBuffer`]. It provides CRUD operations for relation management during
/// query execution.
///
/// # Thread Safety
///
/// This implementation is NOT thread-safe. It is designed for single-threaded
/// runtime execution in the MVP.
///
/// # Example
///
/// ```ignore
/// use xlog_runtime::RelationStore;
/// use xlog_cuda::CudaBuffer;
/// use xlog_core::Schema;
///
/// let mut store = RelationStore::new();
///
/// // Add a relation
/// store.put("edge", buffer);
///
/// // Check if relation exists
/// if store.contains("edge") {
///     let edge = store.get("edge").unwrap();
/// }
///
/// // Remove a relation
/// let removed = store.remove("edge");
/// ```
pub struct RelationStore {
    /// Map of relation names to GPU buffers
    relations: HashMap<String, VersionedCudaBuffer>,
}

struct VersionedCudaBuffer {
    buffer: CudaBuffer,
    version: u64,
}

impl RelationStore {
    /// Create a new empty relation store
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
        }
    }

    /// Get a reference to a relation by name
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    ///
    /// # Returns
    /// `Some(&CudaBuffer)` if the relation exists, `None` otherwise
    pub fn get(&self, name: &str) -> Option<&CudaBuffer> {
        self.relations.get(name).map(|e| &e.buffer)
    }

    /// Get a mutable reference to a relation by name
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    ///
    /// # Returns
    /// `Some(&mut CudaBuffer)` if the relation exists, `None` otherwise
    pub fn get_mut(&mut self, name: &str) -> Option<&mut CudaBuffer> {
        self.relations.get_mut(name).map(|e| {
            // Any mutable access may change the contents; bump the version so cached
            // indexes can be invalidated conservatively.
            e.version = e.version.saturating_add(1);
            &mut e.buffer
        })
    }

    /// Get a relation by name along with its current version.
    pub fn get_with_version(&self, name: &str) -> Option<(&CudaBuffer, u64)> {
        self.relations
            .get(name)
            .map(|e| (&e.buffer, e.version))
    }

    /// Get the current version for a relation.
    pub fn version(&self, name: &str) -> Option<u64> {
        self.relations.get(name).map(|e| e.version)
    }

    /// Store a relation with the given name
    ///
    /// If a relation with the same name already exists, it will be replaced.
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    /// * `buffer` - The GPU buffer containing the relation data
    pub fn put(&mut self, name: &str, buffer: CudaBuffer) {
        let version = self
            .relations
            .get(name)
            .map(|e| e.version.saturating_add(1))
            .unwrap_or(1);
        self.relations.insert(
            name.to_string(),
            VersionedCudaBuffer { buffer, version },
        );
    }

    /// Get a relation by name, or insert an empty buffer with the given schema
    ///
    /// This is useful for semi-naive evaluation where delta relations may not
    /// exist yet on the first iteration. If the relation doesn't exist, an empty
    /// buffer with the given schema is inserted into the store.
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    /// * `schema` - The schema to use if creating an empty buffer
    ///
    /// # Returns
    /// A reference to the existing buffer, or the newly inserted empty buffer
    pub fn get_or_insert_empty(&mut self, name: &str, schema: &Schema) -> &CudaBuffer {
        let entry = self.relations.entry(name.to_string()).or_insert_with(|| {
            VersionedCudaBuffer {
                buffer: CudaBuffer {
                    columns: Vec::new(),
                    num_rows: 0,
                    schema: schema.clone(),
                },
                version: 1,
            }
        });
        &entry.buffer
    }

    /// Get a mutable reference to a relation, or insert an empty buffer with the given schema
    ///
    /// This is useful for semi-naive evaluation where delta relations may not
    /// exist yet on the first iteration. If the relation doesn't exist, an empty
    /// buffer with the given schema is inserted into the store.
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    /// * `schema` - The schema to use if creating an empty buffer
    ///
    /// # Returns
    /// A mutable reference to the existing buffer, or the newly inserted empty buffer
    pub fn get_or_insert_empty_mut(&mut self, name: &str, schema: &Schema) -> &mut CudaBuffer {
        let entry = self.relations.entry(name.to_string()).or_insert_with(|| {
            VersionedCudaBuffer {
                buffer: CudaBuffer {
                    columns: Vec::new(),
                    num_rows: 0,
                    schema: schema.clone(),
                },
                version: 1,
            }
        });
        entry.version = entry.version.saturating_add(1);
        &mut entry.buffer
    }

    /// Check if a relation exists in the store
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    ///
    /// # Returns
    /// `true` if the relation exists, `false` otherwise
    pub fn contains(&self, name: &str) -> bool {
        self.relations.contains_key(name)
    }

    /// Remove a relation from the store
    ///
    /// # Arguments
    /// * `name` - The name of the relation
    ///
    /// # Returns
    /// `Some(CudaBuffer)` if the relation existed, `None` otherwise
    pub fn remove(&mut self, name: &str) -> Option<CudaBuffer> {
        self.relations.remove(name).map(|e| e.buffer)
    }

    /// Clear all relations from the store
    ///
    /// This removes all stored relations. The GPU memory will be freed
    /// when the CudaBuffer instances are dropped.
    pub fn clear(&mut self) {
        self.relations.clear();
    }

    /// Get the number of relations in the store
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Get an iterator over relation names
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.relations.keys().map(|s| s.as_str())
    }
}

impl Default for RelationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U64),
        ])
    }

    #[test]
    fn test_new_store_is_empty() {
        let store = RelationStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_default_store_is_empty() {
        let store = RelationStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn test_put_and_get() {
        let mut store = RelationStore::new();
        let buffer = CudaBuffer::empty();

        store.put("test_rel", buffer);

        assert!(store.contains("test_rel"));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);

        let retrieved = store.get("test_rel");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_get_nonexistent() {
        let store = RelationStore::new();
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_contains() {
        let mut store = RelationStore::new();

        assert!(!store.contains("test"));

        store.put("test", CudaBuffer::empty());

        assert!(store.contains("test"));
        assert!(!store.contains("other"));
    }

    #[test]
    fn test_remove() {
        let mut store = RelationStore::new();
        store.put("test", CudaBuffer::empty());

        assert!(store.contains("test"));

        let removed = store.remove("test");
        assert!(removed.is_some());
        assert!(!store.contains("test"));
        assert!(store.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut store = RelationStore::new();
        let removed = store.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear() {
        let mut store = RelationStore::new();
        store.put("rel1", CudaBuffer::empty());
        store.put("rel2", CudaBuffer::empty());
        store.put("rel3", CudaBuffer::empty());

        assert_eq!(store.len(), 3);

        store.clear();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_get_or_insert_empty_existing() {
        let mut store = RelationStore::new();
        let schema = test_schema();

        // Create a buffer with some metadata
        let buffer = CudaBuffer {
            columns: Vec::new(),
            num_rows: 100,
            schema: schema.clone(),
        };

        store.put("existing", buffer);

        // Should return a reference to the existing buffer (not insert a new one)
        let result = store.get_or_insert_empty("existing", &schema);
        assert_eq!(result.num_rows, 100);
        assert_eq!(result.schema, schema);

        // Verify store still has only one relation
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_get_or_insert_empty_nonexistent() {
        let mut store = RelationStore::new();
        let schema = test_schema();

        // Store should be empty initially
        assert!(store.is_empty());

        // Should insert an empty buffer and return a reference to it
        let result = store.get_or_insert_empty("nonexistent", &schema);
        assert_eq!(result.num_rows, 0);
        assert_eq!(result.schema, schema);
        assert!(result.is_empty());

        // Verify the buffer was actually inserted into the store
        assert!(store.contains("nonexistent"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_get_mut() {
        let mut store = RelationStore::new();
        let buffer = CudaBuffer {
            columns: Vec::new(),
            num_rows: 10,
            schema: Schema::new(vec![]),
        };

        store.put("test", buffer);

        // Modify via get_mut
        {
            let buf_mut = store.get_mut("test").unwrap();
            buf_mut.num_rows = 50;
        }

        // Verify the change persisted
        assert_eq!(store.get("test").unwrap().num_rows, 50);
    }

    #[test]
    fn test_get_mut_nonexistent() {
        let mut store = RelationStore::new();
        assert!(store.get_mut("nonexistent").is_none());
    }

    #[test]
    fn test_get_or_insert_empty_mut() {
        let mut store = RelationStore::new();
        let schema = test_schema();

        // Get mutable reference to a new empty buffer
        {
            let buf_mut = store.get_or_insert_empty_mut("new_rel", &schema);
            assert_eq!(buf_mut.num_rows, 0);
            buf_mut.num_rows = 42;
        }

        // Verify the change persisted and the buffer is in the store
        assert!(store.contains("new_rel"));
        assert_eq!(store.get("new_rel").unwrap().num_rows, 42);
    }

    #[test]
    fn test_put_replaces_existing() {
        let mut store = RelationStore::new();

        let buffer1 = CudaBuffer {
            columns: Vec::new(),
            num_rows: 10,
            schema: Schema::new(vec![]),
        };

        let buffer2 = CudaBuffer {
            columns: Vec::new(),
            num_rows: 20,
            schema: Schema::new(vec![]),
        };

        store.put("test", buffer1);
        assert_eq!(store.get("test").unwrap().num_rows, 10);

        store.put("test", buffer2);
        assert_eq!(store.get("test").unwrap().num_rows, 20);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_names_iterator() {
        let mut store = RelationStore::new();
        store.put("alpha", CudaBuffer::empty());
        store.put("beta", CudaBuffer::empty());
        store.put("gamma", CudaBuffer::empty());

        let mut names: Vec<&str> = store.names().collect();
        names.sort();

        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_multiple_operations() {
        let mut store = RelationStore::new();

        // Add some relations
        store.put("a", CudaBuffer::empty());
        store.put("b", CudaBuffer::empty());
        store.put("c", CudaBuffer::empty());
        assert_eq!(store.len(), 3);

        // Remove one
        store.remove("b");
        assert_eq!(store.len(), 2);
        assert!(!store.contains("b"));

        // Replace one
        store.put(
            "a",
            CudaBuffer {
                columns: Vec::new(),
                num_rows: 50,
                schema: Schema::new(vec![]),
            },
        );
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a").unwrap().num_rows, 50);

        // Clear all
        store.clear();
        assert!(store.is_empty());
    }
}
