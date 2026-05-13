use cudarc::driver::DeviceRepr;

use crate::memory::TrackedCudaSlice;

pub struct WcojRelationMetadata<K: DeviceRepr> {
    pub unique_keys: TrackedCudaSlice<K>,
    pub fan_out: TrackedCudaSlice<u32>,
    pub prefix_sum: TrackedCudaSlice<u32>,
    pub total: u64,
    pub key_count: u32,
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
