//! ILP (Inductive Logic Programming) registry for tensor mask management.

use std::collections::HashMap;
use xlog_core::{ScalarType, Schema, XlogError};
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

    /// Insert a mask built from sparse candidate data.
    ///
    /// Builds dense flat N*N*N hard/soft arrays on host and uploads them as
    /// single-column f32 buffers for executor consumption.
    pub fn insert_mask_from_sparse(
        &mut self,
        name: String,
        schema_size: usize,
        active_ijk: &[(u32, u32, u32)],
        active_soft: &[f32],
        provider: &CudaKernelProvider,
    ) -> Result<(), XlogError> {
        if active_ijk.len() != active_soft.len() {
            return Err(XlogError::Execution(format!(
                "active_ijk length {} != active_soft length {}",
                active_ijk.len(),
                active_soft.len()
            )));
        }

        let total = schema_size
            .checked_mul(schema_size)
            .and_then(|v| v.checked_mul(schema_size))
            .ok_or_else(|| XlogError::Execution(
                format!("schema_size overflow for N={}", schema_size)
            ))?;

        let mut hard_host = vec![0f32; total];
        let mut soft_host = vec![0f32; total];

        for (idx, &(i, j, k)) in active_ijk.iter().enumerate() {
            let flat = (i as usize)
                .checked_mul(schema_size)
                .and_then(|v| v.checked_mul(schema_size))
                .and_then(|v| v.checked_add((j as usize) * schema_size))
                .and_then(|v| v.checked_add(k as usize))
                .ok_or_else(|| XlogError::Execution(format!(
                    "flat index overflow for ({},{},{}) with N={}",
                    i, j, k, schema_size
                )))?;
            if flat >= total {
                return Err(XlogError::Execution(format!(
                    "candidate ({},{},{}) out of bounds for N={}",
                    i, j, k, schema_size
                )));
            }
            hard_host[flat] = 1.0;
            soft_host[flat] = active_soft[idx];
        }

        let mask_schema = Schema::new(vec![("mask".to_string(), ScalarType::F32)]);
        let hard = provider.create_buffer_from_f32_slice(&hard_host, mask_schema.clone())?;
        let soft = provider.create_buffer_from_f32_slice(&soft_host, mask_schema)?;

        self.masks.insert(name, IlpMask { hard, soft, schema_size });
        Ok(())
    }

    pub fn get_mask(&self, name: &str) -> Option<&IlpMask> {
        self.masks.get(name)
    }
}
