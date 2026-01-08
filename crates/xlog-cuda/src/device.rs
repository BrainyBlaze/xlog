//! CUDA device management
//!
//! This module provides a wrapper around cudarc's CUDA device and stream
//! management, adapted for use with xlog's error handling.

use std::sync::Arc;

use cudarc::driver::CudaDevice as CudarcDevice;
use cudarc::driver::CudaStream;
use xlog_core::{Result, XlogError};

/// CUDA device wrapper for GPU operations
///
/// Wraps a cudarc CudaDevice with an associated stream for kernel execution.
/// The device is reference-counted via Arc for safe sharing.
pub struct CudaDevice {
    /// The underlying cudarc device (already Arc-wrapped)
    device: Arc<CudarcDevice>,
    /// Default stream for kernel execution
    stream: CudaStream,
}

impl CudaDevice {
    /// Create a new CUDA device on the specified GPU ordinal
    ///
    /// # Arguments
    /// * `ordinal` - The GPU device index (0-based)
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if device initialization fails
    ///
    /// # Example
    /// ```ignore
    /// let device = CudaDevice::new(0)?;
    /// ```
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudarcDevice::new(ordinal)
            .map_err(|e| XlogError::Kernel(format!("Failed to create CUDA device {}: {}", ordinal, e)))?;

        let stream = device.fork_default_stream()
            .map_err(|e| XlogError::Kernel(format!("Failed to create CUDA stream: {}", e)))?;

        Ok(Self { device, stream })
    }

    /// Synchronize the device, waiting for all operations to complete
    ///
    /// This blocks until all previously queued operations on this device's
    /// stream have completed.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if synchronization fails
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
            .map_err(|e| XlogError::Kernel(format!("Failed to synchronize device: {}", e)))
    }

    /// Get a reference to the underlying cudarc device
    ///
    /// This is useful for memory allocation and kernel operations
    /// that need direct access to the device.
    pub fn inner(&self) -> &Arc<CudarcDevice> {
        &self.device
    }

    /// Get a reference to the device's execution stream
    ///
    /// The stream is used for async kernel execution and memory transfers.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Get the device ordinal (GPU index)
    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        // Skip if no GPU available
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = CudaDevice::new(0);
        assert!(device.is_ok(), "Failed to create device: {:?}", device.err());
    }

    #[test]
    fn test_device_synchronize() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = CudaDevice::new(0).expect("Failed to create device");
        let result = device.synchronize();
        assert!(result.is_ok(), "Failed to synchronize: {:?}", result.err());
    }

    #[test]
    fn test_device_ordinal() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = CudaDevice::new(0).expect("Failed to create device");
        assert_eq!(device.ordinal(), 0);
    }

    #[test]
    fn test_device_inner_access() {
        if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = CudaDevice::new(0).expect("Failed to create device");
        let inner = device.inner();
        // Verify we can access the inner device
        assert_eq!(inner.ordinal(), 0);
    }

    #[test]
    fn test_invalid_device_ordinal() {
        // Try to create a device with an invalid ordinal
        let result = CudaDevice::new(9999);
        assert!(result.is_err(), "Should fail with invalid ordinal");

        if let Err(XlogError::Kernel(msg)) = result {
            assert!(msg.contains("9999"), "Error should mention device ordinal");
        } else {
            panic!("Expected XlogError::Kernel");
        }
    }
}
