use cudarc::driver::DeviceRepr;

use crate::memory::TrackedCudaSlice;

pub const WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT: u32 = 1024;

pub struct WcojRelationMetadata<K: DeviceRepr> {
    pub unique_keys: TrackedCudaSlice<K>,
    pub fan_out: TrackedCudaSlice<u32>,
    pub prefix_sum: TrackedCudaSlice<u32>,
    pub total: u64,
    pub key_count: u32,
    pub row_count: u32,
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
