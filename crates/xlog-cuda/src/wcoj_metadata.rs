use std::collections::BTreeMap;

use cudarc::driver::DeviceRepr;

use crate::memory::TrackedCudaSlice;

pub const WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT: u32 = 1024;

/// Candidate root variable identifier for WCOJ metadata planning.
pub type VertexId = u8;

/// Compact per-root heat distribution used by the K-clique planner.
pub type HeatDist = Vec<f64>;

/// Metadata cached for one candidate root variable.
#[derive(Debug, Clone, PartialEq)]
pub struct RootMetadata {
    /// Column permutation needed to expose this root as the leading key.
    pub column_permutation: Vec<u8>,
    /// Signature of the sorted layout used by this candidate root.
    pub sorted_layout_signature: LayoutSignature,
    /// Heavy-key heat distribution for this root.
    pub heat_distribution: HeatDist,
}

/// Stable identity for a sorted relation layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutSignature {
    /// Runtime relation identifier.
    pub relation_id: u32,
    /// Columns used as the sorted key prefix.
    pub key_columns: Vec<usize>,
    /// Logical rows in the sorted layout.
    pub row_count: u32,
}

pub struct WcojRelationMetadata<K: DeviceRepr> {
    pub unique_keys: TrackedCudaSlice<K>,
    pub fan_out: TrackedCudaSlice<u32>,
    pub prefix_sum: TrackedCudaSlice<u32>,
    /// Per-candidate-root metadata cached for planner reuse.
    pub per_candidate_root: BTreeMap<VertexId, RootMetadata>,
    pub total: u64,
    pub key_count: u32,
    pub row_count: u32,
}

/// D1 widening — which triangle output variable supplies the aggregate
/// value for the fused group-by-root sum/min/max kernels. The group key is
/// always the variable-order root X; the value must itself be a triangle
/// output variable (Y or Z) so the kernel can read it during traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcojRootAggValue {
    /// Aggregate over Y (`e_xy.col1` of the root row).
    Y,
    /// Aggregate over Z (the matched intersection value).
    Z,
}

/// S1d — which 4-cycle output variable supplies the aggregate value for
/// the fused group-by-root sum/min/max kernels. The group key is always
/// the variable-order root W; the value must itself be a 4-cycle output
/// variable (X, Y or Z) so the kernel can read it during traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wcoj4CycleRootAggValue {
    /// Aggregate over X (`e1.col1` of the root row).
    X,
    /// Aggregate over Y (`e2.col1` of the resolved work item).
    Y,
    /// Aggregate over Z (`e3.col1` of the resolved work item).
    Z,
}

pub struct WcojTriangleHgWorkPlanU32 {
    pub xy_work_prefix: TrackedCudaSlice<u32>,
    pub xy_yz_start: TrackedCudaSlice<u32>,
    pub xy_yz_end: TrackedCudaSlice<u32>,
    pub xy_xz_start: TrackedCudaSlice<u32>,
    pub xy_xz_end: TrackedCudaSlice<u32>,
    pub block_counts: TrackedCudaSlice<u32>,
    pub block_offsets: TrackedCudaSlice<u32>,
    pub scratch_x: TrackedCudaSlice<u32>,
    pub scratch_y: TrackedCudaSlice<u32>,
    pub scratch_z: TrackedCudaSlice<u32>,
    pub total_work: u32,
    pub block_work_unit: u32,
    pub row_count: u32,
}

pub struct WcojTriangleHgCountPhaseU32 {
    pub total_rows_device: TrackedCudaSlice<u32>,
    pub total_rows: u32,
}

pub struct WcojTriangleHgWorkPlanU64 {
    pub xy_work_prefix: TrackedCudaSlice<u32>,
    pub xy_yz_start: TrackedCudaSlice<u32>,
    pub xy_yz_end: TrackedCudaSlice<u32>,
    pub xy_xz_start: TrackedCudaSlice<u32>,
    pub xy_xz_end: TrackedCudaSlice<u32>,
    pub block_counts: TrackedCudaSlice<u32>,
    pub block_offsets: TrackedCudaSlice<u32>,
    pub total_work: u32,
    pub block_work_unit: u32,
    pub row_count: u32,
}

pub struct WcojCycle4HgWorkPlanU32 {
    pub e1_work_prefix: TrackedCudaSlice<u32>,
    pub e2_work_prefix: TrackedCudaSlice<u32>,
    pub e1_e2_start: TrackedCudaSlice<u32>,
    pub e1_e2_end: TrackedCudaSlice<u32>,
    pub block_counts: TrackedCudaSlice<u32>,
    pub block_offsets: TrackedCudaSlice<u32>,
    pub total_work: u32,
    pub block_work_unit: u32,
    pub row_count: u32,
}

pub struct WcojCycle4HgWorkPlanU64 {
    pub e1_work_prefix: TrackedCudaSlice<u32>,
    pub e2_work_prefix: TrackedCudaSlice<u32>,
    pub e1_e2_start: TrackedCudaSlice<u32>,
    pub e1_e2_end: TrackedCudaSlice<u32>,
    pub block_counts: TrackedCudaSlice<u32>,
    pub block_offsets: TrackedCudaSlice<u32>,
    pub total_work: u32,
    pub block_work_unit: u32,
    pub row_count: u32,
}

impl<K: DeviceRepr> WcojRelationMetadata<K> {
    pub fn metadata_bytes(&self) -> u64 {
        let key_bytes = self.unique_keys.len() as u64 * std::mem::size_of::<K>() as u64;
        let fan_out_bytes = self.fan_out.len() as u64 * std::mem::size_of::<u32>() as u64;
        let prefix_bytes = self.prefix_sum.len() as u64 * std::mem::size_of::<u32>() as u64;
        key_bytes
            .saturating_add(fan_out_bytes)
            .saturating_add(prefix_bytes)
    }
}
