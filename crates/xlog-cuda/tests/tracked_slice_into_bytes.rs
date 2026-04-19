use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaDevice, GpuMemoryManager};

#[test]
fn tracked_slice_into_bytes_preserves_device_pointer_and_length() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(128 * 1024 * 1024),
    ));

    let slice: TrackedCudaSlice<f64> = memory.alloc::<f64>(3).unwrap();
    let ptr_before = slice.device_ptr_value();
    let bytes = slice.into_bytes();
    let ptr_after = bytes.device_ptr_value();

    assert_eq!(bytes.len(), 3 * std::mem::size_of::<f64>());
    assert_eq!(ptr_after, ptr_before);
}
