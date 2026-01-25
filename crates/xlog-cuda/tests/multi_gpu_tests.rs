//! Tests for multi-GPU support

use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{GpuDevicePool, MultiGpuMemoryManager};

#[test]
fn test_device_pool_creation() {
    let pool = match GpuDevicePool::new(1) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return;
        }
    };

    assert_eq!(pool.device_count(), 1);
    assert!(pool.get_device(0).is_some());
}

#[test]
fn test_device_pool_round_robin() {
    // Prefer a 2-device pool if available; otherwise fall back to 1.
    let pool = match GpuDevicePool::new(2).or_else(|_| GpuDevicePool::new(1)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return;
        }
    };

    // Round-robin should cycle through devices
    let d0 = pool.next_device_idx();
    let d1 = pool.next_device_idx();

    if pool.device_count() > 1 {
        assert_ne!(d0, d1);
    }
}

#[test]
fn test_device_pool_zero_devices_error() {
    // Creating a pool with zero devices should return an error
    let result = GpuDevicePool::new(0);
    assert!(result.is_err());

    match result {
        Err(err) => {
            let err_msg = format!("{}", err);
            assert!(err_msg.contains("at least one device"), "Error message should mention requirement for at least one device");
        }
        Ok(_) => panic!("Expected error for zero device count"),
    }
}

#[test]
fn test_multi_gpu_memory_manager() {
    let pool = match GpuDevicePool::new(1) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1GB per device

    let mgr = MultiGpuMemoryManager::new(pool.clone(), budget).unwrap();

    assert_eq!(mgr.device_count(), 1);

    // Allocate on specific device
    let slice = mgr.alloc_on_device::<u32>(0, 256).unwrap();
    assert_eq!(slice.len(), 256);
}
