//! Relation management for GPU-based Datalog execution
//!
//! This module provides [`RelationStore`], a container for managing named relations
//! stored as GPU buffers. It provides CRUD operations for relations during query
//! execution.

use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::Schema;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

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
/// let mut store = RelationStore::new(provider);
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
    /// CUDA kernel provider for GPU allocations
    provider: Arc<CudaKernelProvider>,
    /// Map of relation names to GPU buffers
    relations: HashMap<String, VersionedCudaBuffer>,
}

struct VersionedCudaBuffer {
    buffer: CudaBuffer,
    version: u64,
}

impl RelationStore {
    /// Create a new empty relation store
    pub fn new(provider: Arc<CudaKernelProvider>) -> Self {
        Self {
            provider,
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
        self.relations.get(name).map(|e| (&e.buffer, e.version))
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
        self.relations
            .insert(name.to_string(), VersionedCudaBuffer { buffer, version });
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
    pub fn get_or_insert_empty(
        &mut self,
        name: &str,
        schema: &Schema,
    ) -> xlog_core::Result<&CudaBuffer> {
        if !self.relations.contains_key(name) {
            let buffer = self.provider.create_empty_buffer(schema.clone())?;
            self.relations
                .insert(name.to_string(), VersionedCudaBuffer { buffer, version: 1 });
        }
        Ok(&self
            .relations
            .get(name)
            .expect("Relation must exist after insertion")
            .buffer)
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
    pub fn get_or_insert_empty_mut(
        &mut self,
        name: &str,
        schema: &Schema,
    ) -> xlog_core::Result<&mut CudaBuffer> {
        if !self.relations.contains_key(name) {
            let buffer = self.provider.create_empty_buffer(schema.clone())?;
            self.relations
                .insert(name.to_string(), VersionedCudaBuffer { buffer, version: 1 });
        }
        let entry = self
            .relations
            .get_mut(name)
            .expect("Relation must exist after insertion");
        entry.version = entry.version.saturating_add(1);
        Ok(&mut entry.buffer)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xlog_core::{MemoryBudget, ScalarType};
    use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("Skipping: CUDA runtime unavailable: {}", e);
                return None;
            }
        };
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(1024 * 1024 * 1024),
        ));
        CudaKernelProvider::new(device, memory).ok().map(Arc::new)
    }

    fn setup_store() -> Option<(RelationStore, Arc<CudaKernelProvider>)> {
        let provider = setup_provider()?;
        let store = RelationStore::new(provider.clone());
        Some((store, provider))
    }

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U64),
        ])
    }

    fn device_row_count(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> u32 {
        let mut host_rows = [0u32];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .expect("dtoh row count");
        host_rows[0]
    }

    fn make_buffer(provider: &CudaKernelProvider, schema: Schema, rows: usize) -> CudaBuffer {
        if schema.arity() == 0 {
            if rows == 0 {
                return provider.create_empty_buffer(schema).expect("empty buffer");
            }
            let rows_u32 = u32::try_from(rows).expect("row count fits u32");
            let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
                .expect("htod row count");
            return CudaBuffer::from_columns(Vec::new(), rows as u64, d_num_rows, schema);
        }
        if rows == 0 {
            return provider.create_empty_buffer(schema).expect("empty buffer");
        }
        let mut columns: Vec<Vec<u8>> = Vec::with_capacity(schema.arity());
        for col_idx in 0..schema.arity() {
            let size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            columns.push(vec![0u8; rows * size]);
        }
        let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        provider
            .create_buffer_from_slices(&slices, schema)
            .expect("buffer")
    }

    #[test]
    fn test_new_store_is_empty() {
        let Some((store, _provider)) = setup_store() else {
            return;
        };
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_put_and_get() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let buffer = provider
            .create_empty_buffer(Schema::new(vec![]))
            .expect("empty");

        store.put("test_rel", buffer);

        assert!(store.contains("test_rel"));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);

        let retrieved = store.get("test_rel");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_get_nonexistent() {
        let Some((store, _provider)) = setup_store() else {
            return;
        };
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_contains() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };

        assert!(!store.contains("test"));

        store.put(
            "test",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );

        assert!(store.contains("test"));
        assert!(!store.contains("other"));
    }

    #[test]
    fn test_remove() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        store.put(
            "test",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );

        assert!(store.contains("test"));

        let removed = store.remove("test");
        assert!(removed.is_some());
        assert!(!store.contains("test"));
        assert!(store.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let Some((mut store, _provider)) = setup_store() else {
            return;
        };
        let removed = store.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let empty = provider
            .create_empty_buffer(Schema::new(vec![]))
            .expect("empty");
        store.put("rel1", empty);
        store.put(
            "rel2",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );
        store.put(
            "rel3",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );

        assert_eq!(store.len(), 3);

        store.clear();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_get_or_insert_empty_existing() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let schema = test_schema();

        let buffer = make_buffer(&provider, schema.clone(), 100);
        store.put("existing", buffer);

        let result = store.get_or_insert_empty("existing", &schema).unwrap();
        assert_eq!(device_row_count(&provider, result), 100);
        assert_eq!(result.schema(), &schema);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_get_or_insert_empty_nonexistent() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let schema = test_schema();

        assert!(store.is_empty());

        let result = store.get_or_insert_empty("nonexistent", &schema).unwrap();
        assert_eq!(device_row_count(&provider, result), 0);
        assert_eq!(result.schema(), &schema);
        assert!(result.is_empty());

        assert!(store.contains("nonexistent"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_get_mut() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let buffer = make_buffer(&provider, Schema::new(vec![]), 10);
        store.put("test", buffer);

        {
            let buf_mut = store.get_mut("test").unwrap();
            buf_mut.row_cap = 50;
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&[50u32], &mut buf_mut.d_num_rows)
                .expect("htod row count");
        }

        assert_eq!(device_row_count(&provider, store.get("test").unwrap()), 50);
    }

    #[test]
    fn test_get_mut_nonexistent() {
        let Some((mut store, _provider)) = setup_store() else {
            return;
        };
        assert!(store.get_mut("nonexistent").is_none());
    }

    #[test]
    fn test_get_or_insert_empty_mut() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        let schema = test_schema();

        {
            let buf_mut = store.get_or_insert_empty_mut("new_rel", &schema).unwrap();
            assert_eq!(device_row_count(&provider, buf_mut), 0);
            buf_mut.row_cap = 42;
            provider
                .device()
                .inner()
                .htod_sync_copy_into(&[42u32], &mut buf_mut.d_num_rows)
                .expect("htod row count");
        }

        assert!(store.contains("new_rel"));
        assert_eq!(
            device_row_count(&provider, store.get("new_rel").unwrap()),
            42
        );
    }

    #[test]
    fn test_put_replaces_existing() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };

        let buffer1 = make_buffer(&provider, Schema::new(vec![]), 10);
        let buffer2 = make_buffer(&provider, Schema::new(vec![]), 20);

        store.put("test", buffer1);
        assert_eq!(device_row_count(&provider, store.get("test").unwrap()), 10);

        store.put("test", buffer2);
        assert_eq!(device_row_count(&provider, store.get("test").unwrap()), 20);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_names_iterator() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };
        store.put(
            "alpha",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );
        store.put(
            "beta",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );
        store.put(
            "gamma",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );

        let mut names: Vec<&str> = store.names().collect();
        names.sort();

        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_multiple_operations() {
        let Some((mut store, provider)) = setup_store() else {
            return;
        };

        let empty = provider
            .create_empty_buffer(Schema::new(vec![]))
            .expect("empty");
        store.put("a", empty);
        store.put(
            "b",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );
        store.put(
            "c",
            provider
                .create_empty_buffer(Schema::new(vec![]))
                .expect("empty"),
        );
        assert_eq!(store.len(), 3);

        store.remove("b");
        assert_eq!(store.len(), 2);
        assert!(!store.contains("b"));

        store.put("a", make_buffer(&provider, Schema::new(vec![]), 50));
        assert_eq!(store.len(), 2);
        assert_eq!(device_row_count(&provider, store.get("a").unwrap()), 50);

        store.clear();
        assert!(store.is_empty());
    }
}
