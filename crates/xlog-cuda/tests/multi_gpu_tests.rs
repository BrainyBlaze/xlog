//! Tests for multi-GPU support

use xlog_cuda::GpuDevicePool;

#[test]
fn test_device_pool_creation() {
    let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let pool = GpuDevicePool::new(device_count as usize).unwrap();

    assert_eq!(pool.device_count(), device_count as usize);
    assert!(pool.get_device(0).is_some());
}

#[test]
fn test_device_pool_round_robin() {
    let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0);
    if device_count < 1 {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let pool = GpuDevicePool::new(device_count as usize).unwrap();

    // Round-robin should cycle through devices
    let d0 = pool.next_device_idx();
    let d1 = pool.next_device_idx();

    if device_count > 1 {
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
