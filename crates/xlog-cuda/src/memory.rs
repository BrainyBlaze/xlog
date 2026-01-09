//! CUDA memory management
//!
//! This module provides GPU memory management with budget enforcement.
//! It wraps cudarc's allocation functions and tracks total allocated memory.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use xlog_core::{MemoryBudget, Result, Schema, XlogError};

use crate::CudaDevice;

/// GPU memory manager with budget enforcement
///
/// Tracks allocated GPU memory and enforces a memory budget.
/// When the budget would be exceeded, returns `XlogError::ResourceExhausted`.
pub struct GpuMemoryManager {
    /// The CUDA device for memory operations
    device: Arc<CudaDevice>,
    /// Memory budget configuration
    budget: MemoryBudget,
    /// Currently allocated bytes (tracked atomically for thread safety)
    allocated: AtomicU64,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    ///
    /// # Arguments
    /// * `device` - The CUDA device to allocate memory on
    /// * `budget` - Memory budget configuration
    pub fn new(device: Arc<CudaDevice>, budget: MemoryBudget) -> Self {
        Self {
            device,
            budget,
            allocated: AtomicU64::new(0),
        }
    }

    /// Allocate GPU memory for `len` elements of type `T`
    ///
    /// # Arguments
    /// * `len` - Number of elements to allocate
    ///
    /// # Returns
    /// A `CudaSlice<T>` containing the allocated memory
    ///
    /// # Errors
    /// - `XlogError::ResourceExhausted` if allocation would exceed budget
    /// - `XlogError::Kernel` if CUDA allocation fails
    pub fn alloc<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        // Fix Issue 2: Use checked_mul to prevent integer overflow before cast
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| XlogError::Kernel("Allocation size overflow".to_string()))?;

        // Fix Issue 1: Use compare_exchange loop to prevent TOCTOU race condition
        // Two threads could both pass check_budget() but exceed budget together
        loop {
            let current = self.allocated.load(Ordering::SeqCst);
            let new_val = current.saturating_add(bytes);
            if new_val > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted {
                    context: "GPU memory allocation".to_string(),
                    estimated_bytes: bytes,
                    budget_bytes: self.budget.device_bytes,
                });
            }
            if self
                .allocated
                .compare_exchange(current, new_val, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }

        // Perform allocation
        // SAFETY: We have reserved budget atomically and the device is valid.
        // cudarc's alloc returns properly aligned memory for type T.
        let slice = unsafe {
            self.device.inner().alloc::<T>(len).map_err(|e| {
                // Rollback the allocation tracking if CUDA allocation fails
                self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                XlogError::Kernel(format!("GPU allocation failed: {}", e))
            })?
        };

        Ok(slice)
    }

    /// Check if an allocation of `bytes` would exceed the budget
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to allocate
    ///
    /// # Returns
    /// `Ok(())` if allocation is within budget
    ///
    /// # Errors
    /// `XlogError::ResourceExhausted` if allocation would exceed budget
    pub fn check_budget(&self, bytes: u64) -> Result<()> {
        let current = self.allocated.load(Ordering::SeqCst);
        let proposed = current.saturating_add(bytes);

        if proposed > self.budget.device_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "GPU memory allocation".to_string(),
                estimated_bytes: bytes,
                budget_bytes: self.budget.device_bytes,
            });
        }

        Ok(())
    }

    /// Get the current allocated memory in bytes
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get the memory budget
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Get the underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Record that memory has been freed
    ///
    /// Note: cudarc automatically frees memory when CudaSlice is dropped.
    /// This method should be called to update tracking when memory is freed.
    pub fn record_free(&self, bytes: u64) {
        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Get remaining budget in bytes
    pub fn remaining_bytes(&self) -> u64 {
        let allocated = self.allocated.load(Ordering::SeqCst);
        self.budget.device_bytes.saturating_sub(allocated)
    }

    /// Reset allocation tracking
    ///
    /// This should be called when GPU memory has been freed but the tracker
    /// hasn't been updated (e.g., when CudaSlice instances are dropped without
    /// calling record_free). This is a temporary workaround until proper
    /// RAII-based tracking is implemented.
    pub fn reset_tracking(&self) {
        self.allocated.store(0, Ordering::SeqCst);
    }
}

/// Column-oriented GPU buffer
///
/// Holds columnar data on the GPU with an associated schema.
/// Each column is stored as a separate `CudaSlice<u8>`.
pub struct CudaBuffer {
    /// Column data stored as raw bytes
    pub columns: Vec<CudaSlice<u8>>,
    /// Number of rows in the buffer
    pub num_rows: u64,
    /// Schema describing the column types
    pub schema: Schema,
}

impl CudaBuffer {
    /// Create an empty buffer with no columns or rows
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            num_rows: 0,
            schema: Schema::new(vec![]),
        }
    }

    /// Create a buffer from existing columns
    ///
    /// # Arguments
    /// * `columns` - Pre-allocated column data
    /// * `num_rows` - Number of rows in the buffer
    /// * `schema` - Schema describing the columns
    ///
    /// # Panics
    /// Panics if the number of columns doesn't match the schema arity
    pub fn from_columns(columns: Vec<CudaSlice<u8>>, num_rows: u64, schema: Schema) -> Self {
        assert_eq!(
            columns.len(),
            schema.arity(),
            "Number of columns ({}) must match schema arity ({})",
            columns.len(),
            schema.arity()
        );
        Self {
            columns,
            num_rows,
            schema,
        }
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> u64 {
        self.num_rows
    }

    /// Check if the buffer is empty (has no rows)
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the number of columns (arity)
    pub fn arity(&self) -> usize {
        self.schema.arity()
    }

    /// Estimated memory usage in bytes
    pub fn estimated_bytes(&self) -> u64 {
        self.num_rows * self.schema.row_size_bytes() as u64
    }

    /// Get a reference to a specific column by index
    pub fn column(&self, index: usize) -> Option<&CudaSlice<u8>> {
        self.columns.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    // Test CudaBuffer without requiring a GPU
    #[test]
    fn test_cuda_buffer_empty() {
        let buffer = CudaBuffer::empty();
        assert!(buffer.is_empty());
        assert_eq!(buffer.num_rows(), 0);
        assert_eq!(buffer.arity(), 0);
        assert_eq!(buffer.estimated_bytes(), 0);
    }

    #[test]
    fn test_cuda_buffer_schema() {
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U64),
        ]);

        // Create an empty buffer with schema but no columns (simulating)
        let buffer = CudaBuffer {
            columns: Vec::new(),
            num_rows: 100,
            schema: schema.clone(),
        };

        assert_eq!(buffer.num_rows(), 100);
        assert_eq!(buffer.arity(), 2);
        // 4 bytes (U32) + 8 bytes (U64) = 12 bytes per row * 100 rows
        assert_eq!(buffer.estimated_bytes(), 1200);
        assert_eq!(buffer.schema(), &schema);
    }

    // Tests requiring GPU
    #[test]
    fn test_memory_manager_creation() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024); // 1 MB
        let manager = GpuMemoryManager::new(device, budget);

        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.budget().device_bytes, 1024 * 1024);
        assert_eq!(manager.remaining_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_memory_manager_alloc() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024); // 1 MB
        let manager = GpuMemoryManager::new(device, budget);

        // Allocate 256 u32 values = 1024 bytes
        let _slice = manager.alloc::<u32>(256).expect("Allocation should succeed");

        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 1024 * 1024 - 1024);
    }

    #[test]
    fn test_memory_manager_budget_exceeded() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024); // 1 KB limit
        let manager = GpuMemoryManager::new(device, budget);

        // Try to allocate 512 u32 values = 2048 bytes (exceeds 1KB budget)
        let result = manager.alloc::<u32>(512);

        assert!(result.is_err());
        if let Err(XlogError::ResourceExhausted { estimated_bytes, budget_bytes, .. }) = result {
            assert_eq!(estimated_bytes, 2048);
            assert_eq!(budget_bytes, 1024);
        } else {
            panic!("Expected ResourceExhausted error");
        }
    }

    #[test]
    fn test_memory_manager_check_budget() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1000);
        let manager = GpuMemoryManager::new(device, budget);

        // Check that 500 bytes is within budget
        assert!(manager.check_budget(500).is_ok());

        // Check that 1001 bytes exceeds budget
        assert!(manager.check_budget(1001).is_err());
    }

    #[test]
    fn test_memory_manager_multiple_allocs() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(4096); // 4 KB
        let manager = GpuMemoryManager::new(device, budget);

        // First allocation: 256 u32 = 1024 bytes
        let _slice1 = manager.alloc::<u32>(256).expect("First allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 1024);

        // Second allocation: 256 u32 = 1024 bytes
        let _slice2 = manager.alloc::<u32>(256).expect("Second allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 2048);

        // Third allocation that would exceed budget
        let result = manager.alloc::<u32>(1024); // 4096 bytes, would make total 6144
        assert!(result.is_err());

        // Allocated should still be 2048
        assert_eq!(manager.allocated_bytes(), 2048);
    }

    #[test]
    fn test_memory_manager_record_free() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(4096);
        let manager = GpuMemoryManager::new(device, budget);

        // Allocate
        let _slice = manager.alloc::<u32>(256).expect("Allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 1024);

        // Record free (simulating drop)
        manager.record_free(1024);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn test_cuda_buffer_from_columns() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024);
        let manager = GpuMemoryManager::new(device, budget);

        let schema = Schema::new(vec![
            ("col1".to_string(), ScalarType::U32),
            ("col2".to_string(), ScalarType::U32),
        ]);

        // Allocate columns (100 rows * 4 bytes = 400 bytes each)
        let col1 = manager.alloc::<u8>(400).expect("Alloc col1");
        let col2 = manager.alloc::<u8>(400).expect("Alloc col2");

        let buffer = CudaBuffer::from_columns(vec![col1, col2], 100, schema);

        assert_eq!(buffer.num_rows(), 100);
        assert_eq!(buffer.arity(), 2);
        assert!(!buffer.is_empty());
        assert!(buffer.column(0).is_some());
        assert!(buffer.column(1).is_some());
        assert!(buffer.column(2).is_none());
    }

    #[test]
    #[should_panic(expected = "Number of columns")]
    fn test_cuda_buffer_from_columns_mismatch() {
        let schema = Schema::new(vec![
            ("col1".to_string(), ScalarType::U32),
            ("col2".to_string(), ScalarType::U32),
        ]);

        // This should panic: 0 columns but schema has 2
        CudaBuffer::from_columns(vec![], 100, schema);
    }
}
