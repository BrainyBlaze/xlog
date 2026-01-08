//! GPU kernel provider for XLOG

pub mod device;
pub mod memory;
pub mod provider;
pub mod kernels;

pub use device::CudaDevice;
