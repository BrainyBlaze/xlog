//! Multi-GPU memory management
//!
//! Provides memory allocation across multiple GPU devices.

use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};

use crate::{GpuDevicePool, GpuMemoryManager};
use crate::memory::TrackedCudaSlice;

/// Memory manager for multiple GPU devices
pub struct MultiGpuMemoryManager {
    pool: Arc<GpuDevicePool>,
    managers: Vec<Arc<GpuMemoryManager>>,
}

impl MultiGpuMemoryManager {
    /// Create a new multi-GPU memory manager
    pub fn new(pool: Arc<GpuDevicePool>, budget_per_device: MemoryBudget) -> Result<Self> {
        let mut managers = Vec::with_capacity(pool.device_count());

        for device in pool.devices() {
            let mgr = GpuMemoryManager::new(device.clone(), budget_per_device.clone());
            managers.push(Arc::new(mgr));
        }

        Ok(Self { pool, managers })
    }

    /// Get the number of devices
    pub fn device_count(&self) -> usize {
        self.pool.device_count()
    }

    /// Allocate memory on a specific device
    pub fn alloc_on_device<T: cudarc::driver::DeviceRepr>(
        &self,
        device_idx: usize,
        len: usize,
    ) -> Result<TrackedCudaSlice<T>> {
        let mgr = self.managers.get(device_idx)
            .ok_or_else(|| XlogError::Kernel(format!(
                "Device {} not found", device_idx
            )))?;
        mgr.alloc::<T>(len)
    }

    /// Allocate memory on the next device (round-robin)
    pub fn alloc_next<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<(usize, TrackedCudaSlice<T>)> {
        let device_idx = self.pool.next_device_idx();
        let slice = self.alloc_on_device::<T>(device_idx, len)?;
        Ok((device_idx, slice))
    }

    /// Get the memory manager for a specific device
    pub fn get_manager(&self, device_idx: usize) -> Option<&Arc<GpuMemoryManager>> {
        self.managers.get(device_idx)
    }

    /// Get remaining bytes on a specific device
    pub fn remaining_bytes(&self, device_idx: usize) -> u64 {
        self.managers.get(device_idx)
            .map(|m| m.remaining_bytes())
            .unwrap_or(0)
    }

    /// Get the device pool
    pub fn pool(&self) -> &Arc<GpuDevicePool> {
        &self.pool
    }
}
