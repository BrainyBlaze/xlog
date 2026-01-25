//! Test prefix sum with more than 256 elements

use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024), // 1 GB budget
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_prefix_sum_1000_elements() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    // Create mask with 1000 elements, alternating 0 and 1
    let mask: Vec<u8> = (0..1000).map(|i| (i % 2) as u8).collect();

    let result = provider.prefix_sum_mask(&mask);
    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with 1000 elements, got: {:?}",
        result
    );

    let (prefix_sum, count) = result.unwrap();
    assert_eq!(count, 500, "Should count 500 ones");
    assert_eq!(prefix_sum.len(), 1000);

    // Verify prefix sum values
    let mut expected = 0u32;
    for i in 0..1000 {
        assert_eq!(prefix_sum[i], expected, "prefix_sum[{}] wrong", i);
        expected += mask[i] as u32;
    }
}

#[test]
fn test_prefix_sum_10000_elements() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let mask: Vec<u8> = (0..10000).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();

    let result = provider.prefix_sum_mask(&mask);
    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with 10000 elements"
    );

    let (_prefix_sum, count) = result.unwrap();
    let expected_count = (10000 + 2) / 3; // ceil(10000/3) = 3334
    assert_eq!(count, expected_count as u32);
}

#[test]
fn test_prefix_sum_all_ones_large() {
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let n = 5000;
    let mask: Vec<u8> = vec![1u8; n];

    let result = provider.prefix_sum_mask(&mask);
    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with {} elements",
        n
    );

    let (prefix_sum, count) = result.unwrap();
    assert_eq!(count, n as u32);

    // Exclusive prefix sum of all 1s should be [0, 1, 2, 3, ...]
    for i in 0..n {
        assert_eq!(prefix_sum[i], i as u32, "prefix_sum[{}] wrong", i);
    }
}

#[test]
fn test_prefix_sum_257_elements() {
    // Test just over the old 256-element limit
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let mask: Vec<u8> = (0..257).map(|i| (i % 2) as u8).collect();

    let result = provider.prefix_sum_mask(&mask);
    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with 257 elements, got: {:?}",
        result
    );

    let (prefix_sum, count) = result.unwrap();
    // 257 elements, alternating 0,1,0,1... starting with 0
    // indices 1,3,5,...,255 = 128 ones
    assert_eq!(count, 128);

    // Verify the prefix sum
    let mut expected = 0u32;
    for i in 0..257 {
        assert_eq!(prefix_sum[i], expected, "prefix_sum[{}] wrong", i);
        expected += mask[i] as u32;
    }
}

#[test]
fn test_prefix_sum_exact_block_boundary() {
    // Test at exact block boundary (512 = 2 blocks of 256)
    let provider = match create_test_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no CUDA device");
            return;
        }
    };

    let n = 512;
    let mask: Vec<u8> = vec![1u8; n];

    let result = provider.prefix_sum_mask(&mask);
    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with {} elements",
        n
    );

    let (prefix_sum, count) = result.unwrap();
    assert_eq!(count, n as u32);

    for i in 0..n {
        assert_eq!(prefix_sum[i], i as u32, "prefix_sum[{}] wrong", i);
    }
}
