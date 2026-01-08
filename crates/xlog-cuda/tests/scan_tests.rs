// crates/xlog-cuda/tests/scan_tests.rs
use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::default(),
    ));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_prefix_sum_mask_simple() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // mask: [1, 0, 1, 1, 0, 1]
    // exclusive prefix sum: [0, 1, 1, 2, 3, 3]
    // total count: 4
    let mask = vec![1u8, 0, 1, 1, 0, 1];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 4);
    assert_eq!(prefix_sum, vec![0u32, 1, 1, 2, 3, 3]);
}

#[test]
fn test_prefix_sum_mask_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mask = vec![0u8; 10];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 0);
    assert_eq!(prefix_sum, vec![0u32; 10]);
}

#[test]
fn test_prefix_sum_mask_all_ones() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mask = vec![1u8; 5];
    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 5);
    assert_eq!(prefix_sum, vec![0u32, 1, 2, 3, 4]);
}
