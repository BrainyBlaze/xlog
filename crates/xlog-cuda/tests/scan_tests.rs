// crates/xlog-cuda/tests/scan_tests.rs
mod common;
use common::setup_provider;

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

#[test]
fn test_prefix_sum_mask_max_size() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Test exactly at the 256-element limit
    let mut mask = vec![0u8; 256];
    mask[0] = 1;
    mask[127] = 1;
    mask[255] = 1;

    let (prefix_sum, count) = provider.prefix_sum_mask(&mask).unwrap();

    assert_eq!(count, 3);
    assert_eq!(prefix_sum[0], 0); // First 1 at index 0
    assert_eq!(prefix_sum[127], 1); // Second 1 at index 127
    assert_eq!(prefix_sum[255], 2); // Third 1 at index 255
}

#[test]
fn test_prefix_sum_mask_over_256() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Test over 256 elements now works with multi-block scan
    let mask = vec![1u8; 257];
    let result = provider.prefix_sum_mask(&mask);

    assert!(
        result.is_ok(),
        "prefix_sum_mask should work with 257 elements"
    );
    let (prefix_sum, count) = result.unwrap();
    assert_eq!(count, 257);
    // Verify exclusive prefix sum: [0, 1, 2, ..., 256]
    for i in 0..257 {
        assert_eq!(prefix_sum[i], i as u32);
    }
}
