//! Core traits for XLOG extensibility

use crate::types::{AggOp, Schema};
use crate::Result;

/// Opaque handle to GPU memory buffer
/// Actual implementation lives in xlog-cuda
#[derive(Debug)]
pub struct GpuBuffer {
    /// Number of rows in this buffer
    pub num_rows: u64,
    /// Schema of this buffer
    pub schema: Schema,
    /// Opaque handle (will be CudaSlice in xlog-cuda)
    handle: GpuBufferHandle,
}

#[derive(Debug)]
enum GpuBufferHandle {
    Empty,
    // Cuda(cudarc::driver::CudaSlice<u8>) - added in xlog-cuda
}

impl GpuBuffer {
    /// Create an empty buffer
    pub fn empty() -> Self {
        Self {
            num_rows: 0,
            schema: Schema::new(vec![]),
            handle: GpuBufferHandle::Empty,
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Estimated memory usage in bytes
    pub fn estimated_bytes(&self) -> u64 {
        self.num_rows * self.schema.row_size_bytes() as u64
    }
}

/// Trait for GPU kernel execution providers
///
/// This abstraction allows swapping CUDA for other backends (HIP, SYCL)
pub trait KernelProvider: Send + Sync {
    /// Perform a hash join between two buffers
    fn hash_join(
        &self,
        left: &GpuBuffer,
        right: &GpuBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<GpuBuffer>;

    /// Remove duplicate rows based on key columns
    fn dedup(&self, input: &GpuBuffer, key_cols: &[usize]) -> Result<GpuBuffer>;

    /// Compute union of two buffers
    fn union(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;

    /// Compute set difference (a - b)
    fn diff(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;

    /// Perform groupby aggregation
    fn groupby_agg(
        &self,
        input: &GpuBuffer,
        key_cols: &[usize],
        agg: AggOp,
        value_col: usize,
    ) -> Result<GpuBuffer>;
}

/// Trait for relation storage backends
pub trait RelationStore: Send + Sync {
    /// Get a relation by ID
    fn get(&self, name: &str) -> Option<&GpuBuffer>;

    /// Store a relation
    fn put(&mut self, name: &str, buffer: GpuBuffer);

    /// Check if relation exists
    fn contains(&self, name: &str) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider;

    impl KernelProvider for MockProvider {
        fn hash_join(
            &self,
            _left: &GpuBuffer,
            _right: &GpuBuffer,
            _left_keys: &[usize],
            _right_keys: &[usize],
        ) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn dedup(&self, _input: &GpuBuffer, _key_cols: &[usize]) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn union(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn diff(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn groupby_agg(
            &self,
            _input: &GpuBuffer,
            _key_cols: &[usize],
            _agg: AggOp,
            _value_col: usize,
        ) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }
    }

    #[test]
    fn test_mock_provider_compiles() {
        let provider = MockProvider;
        let empty = GpuBuffer::empty();
        assert!(provider.dedup(&empty, &[0]).is_ok());
    }

    #[test]
    fn test_gpu_buffer_empty() {
        let buf = GpuBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.estimated_bytes(), 0);
    }
}
