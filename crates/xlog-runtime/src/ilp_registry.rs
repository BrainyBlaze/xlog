//! ILP (Inductive Logic Programming) registry for tensor mask management.

use std::collections::HashMap;
use xlog_core::XlogError;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

/// Reads the device-side row count using only public APIs (RD-22).
/// `CudaKernelProvider::device_row_count` is private (provider.rs:6904).
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    let mut host_rows = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    Ok(host_rows[0] as usize)
}

/// Registry for ILP tensor masks.
pub struct IlpRegistry {
    masks: HashMap<String, IlpMask>,
}

/// A registered ILP mask pair (hard + soft) with schema size.
pub struct IlpMask {
    /// Flat 1D CudaBuffer (N*N*N f32 elements), imported via DLPack.
    /// Python flattens 3D tensor to 1D before export (RD-16).
    pub hard: CudaBuffer,
    pub soft: CudaBuffer,
    pub schema_size: usize,
}

/// Tag metadata from TensorMaskedJoin execution.
/// Retains per-entry projected join buffers for batch credit queries.
pub struct IlpTaggedResult {
    pub entries: Vec<IlpTagEntry>,
}

/// Metadata for a single active rule (i,j,k), its result cardinality,
/// and the projected join result buffer (retained for batch credit queries).
pub struct IlpTagEntry {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub num_rows: u32,
    /// The projected join result buffer, retained for batch credit queries.
    pub buffer: Option<CudaBuffer>,
}

impl IlpRegistry {
    pub fn new() -> Self {
        Self {
            masks: HashMap::new(),
        }
    }

    pub fn insert_mask(
        &mut self,
        name: String,
        hard: CudaBuffer,
        soft: CudaBuffer,
        schema_size: usize,
    ) {
        self.masks
            .insert(name, IlpMask { hard, soft, schema_size });
    }

    pub fn get_mask(&self, name: &str) -> Option<&IlpMask> {
        self.masks.get(name)
    }
}
