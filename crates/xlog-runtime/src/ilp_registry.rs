//! ILP (Inductive Logic Programming) registry for tensor mask management.

use std::collections::HashMap;
use xlog_core::XlogError;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

/// Reads the device-side row count through the provider's public metadata API.
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    provider.device_row_count(buffer)
}

/// Registry for ILP tensor masks.
pub struct IlpRegistry {
    masks: HashMap<String, IlpMask>,
}

/// A registered ILP mask — Dense (imported via DLPack) or Sparse (candidate entries only).
pub enum IlpMask {
    Dense {
        hard: CudaBuffer,
        soft: CudaBuffer,
        schema_size: usize,
    },
    Sparse {
        active_entries: Vec<(u32, u32, u32)>,
        schema_size: usize,
    },
}

impl IlpMask {
    pub fn schema_size(&self) -> usize {
        match self {
            IlpMask::Dense { schema_size, .. } => *schema_size,
            IlpMask::Sparse { schema_size, .. } => *schema_size,
        }
    }
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

    /// Clear all registered masks, releasing GPU buffers.
    pub fn clear(&mut self) {
        self.masks.clear();
    }

    pub fn insert_mask(
        &mut self,
        name: String,
        hard: CudaBuffer,
        soft: CudaBuffer,
        schema_size: usize,
    ) {
        self.masks.insert(
            name,
            IlpMask::Dense {
                hard,
                soft,
                schema_size,
            },
        );
    }

    /// Insert a mask built from sparse candidate data.
    ///
    /// Performs deterministic top-k ranking (desc soft value, then lower index)
    /// and stores the selected (i,j,k) entries directly — no dense buffer.
    pub fn insert_mask_from_sparse(
        &mut self,
        name: String,
        schema_size: usize,
        active_ijk: &[(u32, u32, u32)],
        active_soft: &[f32],
        budget: usize,
    ) -> Result<(), XlogError> {
        if active_ijk.len() != active_soft.len() {
            return Err(XlogError::Execution(format!(
                "active_ijk length {} != active_soft length {}",
                active_ijk.len(),
                active_soft.len()
            )));
        }

        // Deterministic top-k: descending soft value, then ascending index for ties
        let mut ranked: Vec<(usize, f32)> = active_soft.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(budget.min(ranked.len()));

        let entries: Vec<(u32, u32, u32)> =
            ranked.iter().map(|&(idx, _)| active_ijk[idx]).collect();

        self.masks.insert(
            name,
            IlpMask::Sparse {
                active_entries: entries,
                schema_size,
            },
        );
        Ok(())
    }

    /// Insert an already-selected sparse mask, preserving caller order exactly.
    pub fn insert_selected_mask(
        &mut self,
        name: String,
        schema_size: usize,
        active_entries: &[(u32, u32, u32)],
    ) {
        self.masks.insert(
            name,
            IlpMask::Sparse {
                active_entries: active_entries.to_vec(),
                schema_size,
            },
        );
    }

    pub fn get_mask(&self, name: &str) -> Option<&IlpMask> {
        self.masks.get(name)
    }
}
