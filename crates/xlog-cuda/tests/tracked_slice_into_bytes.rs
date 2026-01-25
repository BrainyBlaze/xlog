use std::sync::Arc;

use cudarc::driver::{DevicePtr, DeviceSlice};
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
    let ptr_before = *DevicePtr::device_ptr(&slice);
    let bytes = slice.into_bytes();
    let ptr_after = *DevicePtr::device_ptr(&bytes);

    assert_eq!(bytes.len(), 3 * std::mem::size_of::<f64>());
    assert_eq!(ptr_after, ptr_before);
}
