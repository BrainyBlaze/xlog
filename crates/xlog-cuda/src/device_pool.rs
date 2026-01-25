//! Multi-GPU device pool management
//!
//! Provides a pool of CUDA devices for distributing operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use xlog_core::{Result, XlogError};

use crate::CudaDevice;

/// Pool of CUDA devices for multi-GPU operations
///
/// Manages multiple devices and provides round-robin scheduling.
pub struct GpuDevicePool {
    /// Available CUDA devices
    devices: Vec<Arc<CudaDevice>>,
    /// Current device index for round-robin scheduling
    current: AtomicUsize,
}

impl GpuDevicePool {
    /// Create a new device pool with the specified number of devices
    pub fn new(device_count: usize) -> Result<Self> {
        if device_count == 0 {
            return Err(XlogError::Kernel(
                "Device pool requires at least one device".to_string(),
            ));
        }

        // cudarc may panic on driver init failures in restricted containers; treat as a normal error.
        let available = std::panic::catch_unwind(|| cudarc::driver::CudaDevice::count())
            .map_err(|_| {
                XlogError::Kernel(
                    "Failed to count devices: cudarc panicked during driver initialization"
                        .to_string(),
                )
            })?
            .map_err(|e| XlogError::Kernel(format!("Failed to count devices: {}", e)))?;

        if device_count > available as usize {
            return Err(XlogError::Kernel(format!(
                "Requested {} devices but only {} available",
                device_count, available
            )));
        }

        let mut devices = Vec::with_capacity(device_count);
        for ordinal in 0..device_count {
            let device = CudaDevice::new(ordinal).map_err(|e| {
                XlogError::Kernel(format!("Failed to create device {}: {}", ordinal, e))
            })?;
            devices.push(Arc::new(device));
        }

        Ok(Self {
            devices,
            current: AtomicUsize::new(0),
        })
    }

    /// Get the number of devices in the pool
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get a specific device by index
    pub fn get_device(&self, idx: usize) -> Option<&Arc<CudaDevice>> {
        self.devices.get(idx)
    }

    /// Get the next device index using round-robin scheduling
    pub fn next_device_idx(&self) -> usize {
        let idx = self.current.fetch_add(1, Ordering::SeqCst);
        idx % self.devices.len()
    }

    /// Get the next device using round-robin scheduling
    pub fn next_device(&self) -> &Arc<CudaDevice> {
        let idx = self.next_device_idx();
        &self.devices[idx]
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        for (i, device) in self.devices.iter().enumerate() {
            device
                .synchronize()
                .map_err(|e| XlogError::Kernel(format!("Failed to sync device {}: {}", i, e)))?;
        }
        Ok(())
    }

    /// Get all devices
    pub fn devices(&self) -> &[Arc<CudaDevice>] {
        &self.devices
    }
}
