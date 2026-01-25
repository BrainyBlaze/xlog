//! Property-based tests for CUDA kernels.
//!
//! This test file exercises the property-based tests defined in the
//! xlog-cuda-tests crate using proptest.
//!
//! # Usage
//!
//! ```bash
//! # Run all property tests
//! cargo test -p xlog-cuda-tests --test properties --release
//!
//! # Run with verbose output
//! cargo test -p xlog-cuda-tests --test properties --release -- --nocapture
//! ```

// Re-export the property tests from the library.
// The actual tests are defined in src/properties.rs using proptest! macros.
// This file makes them runnable as a separate test target.

use xlog_cuda_tests::harness::TestContext;
use xlog_cuda_tests::properties::*;

/// Convenience function to get a test context.
fn get_context() -> Option<TestContext> {
    match TestContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("Skipping property tests: no CUDA device ({})", e);
            None
        }
    }
}

// =============================================================================
// Integration Tests - Run Core Properties
// =============================================================================

/// Integration test: Sort stability with random data.
#[test]
fn integration_sort_stability() {
    let Some(ctx) = get_context() else { return };

    // Test with increasing sizes
    for size in [10, 100, 1000, 5000] {
        let keys: Vec<u32> = (0..size).map(|i| (i * 17 + 5) % 100).collect();
        let vals: Vec<u32> = (0..size).collect();

        if let Err(e) = prop_sort_stability(&ctx, keys, vals) {
            panic!("Sort stability failed at size {}: {}", size, e);
        }
    }

    println!("Sort stability: PASSED for sizes [10, 100, 1000, 5000]");
}

/// Integration test: Join correctness with various patterns.
#[test]
fn integration_join_correctness() {
    let Some(ctx) = get_context() else { return };

    // Test 1: Perfect 1:1 match
    let lk: Vec<u32> = (0..100).collect();
    let lv: Vec<u32> = lk.iter().map(|k| k * 10).collect();
    let rk = lk.clone();
    let rv: Vec<u32> = rk.iter().map(|k| k * 100).collect();
    if let Err(e) = prop_join_correctness(&ctx, lk, lv, rk, rv) {
        panic!("Join correctness (1:1) failed: {}", e);
    }

    // Test 2: No matches
    let lk: Vec<u32> = (0..100).collect();
    let lv: Vec<u32> = lk.iter().map(|k| k * 10).collect();
    let rk: Vec<u32> = (100..200).collect();
    let rv: Vec<u32> = rk.iter().map(|k| k * 100).collect();
    if let Err(e) = prop_join_correctness(&ctx, lk, lv, rk, rv) {
        panic!("Join correctness (no match) failed: {}", e);
    }

    // Test 3: Many-to-many (duplicates)
    let lk: Vec<u32> = vec![1, 1, 2, 2, 3];
    let lv: Vec<u32> = vec![10, 20, 30, 40, 50];
    let rk: Vec<u32> = vec![1, 2, 2, 3, 3];
    let rv: Vec<u32> = vec![100, 200, 300, 400, 500];
    if let Err(e) = prop_join_correctness(&ctx, lk, lv, rk, rv) {
        panic!("Join correctness (many:many) failed: {}", e);
    }

    println!("Join correctness: PASSED for patterns [1:1, no-match, many:many]");
}

/// Integration test: Filter idempotence with various thresholds.
#[test]
fn integration_filter_idempotence() {
    let Some(ctx) = get_context() else { return };

    let data: Vec<u32> = (0..1000).collect();

    // Test with various thresholds
    for threshold in [0, 100, 500, 900, 1000, 2000] {
        if let Err(e) = prop_filter_idempotence(&ctx, data.clone(), threshold) {
            panic!(
                "Filter idempotence failed at threshold {}: {}",
                threshold, e
            );
        }
    }

    println!("Filter idempotence: PASSED for thresholds [0, 100, 500, 900, 1000, 2000]");
}

/// Integration test: Dedup determinism with various patterns.
#[test]
fn integration_dedup_determinism() {
    let Some(ctx) = get_context() else { return };

    // Test 1: All unique
    let keys: Vec<u32> = (0..1000).collect();
    let vals = keys.clone();
    if let Err(e) = prop_dedup_determinism(&ctx, keys, vals) {
        panic!("Dedup determinism (all unique) failed: {}", e);
    }

    // Test 2: All same
    let keys: Vec<u32> = vec![42; 1000];
    let vals: Vec<u32> = (0..1000).collect();
    if let Err(e) = prop_dedup_determinism(&ctx, keys, vals) {
        panic!("Dedup determinism (all same) failed: {}", e);
    }

    // Test 3: Pairs
    let keys: Vec<u32> = (0..500).flat_map(|i| vec![i, i]).collect();
    let vals: Vec<u32> = (0..1000).collect();
    if let Err(e) = prop_dedup_determinism(&ctx, keys, vals) {
        panic!("Dedup determinism (pairs) failed: {}", e);
    }

    // Test 4: Random pattern
    let keys: Vec<u32> = (0..1000).map(|i| (i * 17 + 3) % 200).collect();
    let vals: Vec<u32> = (0..1000).collect();
    if let Err(e) = prop_dedup_determinism(&ctx, keys, vals) {
        panic!("Dedup determinism (random) failed: {}", e);
    }

    println!("Dedup determinism: PASSED for patterns [all-unique, all-same, pairs, random]");
}

// =============================================================================
// Stress Tests
// =============================================================================

/// Stress test: Large sort stability.
#[test]
fn stress_sort_stability_large() {
    let Some(ctx) = get_context() else { return };

    // 50K rows with many duplicates
    let keys: Vec<u32> = (0..50000).map(|i| (i * 7) % 500).collect();
    let vals: Vec<u32> = (0..50000).collect();

    if let Err(e) = prop_sort_stability(&ctx, keys, vals) {
        panic!("Large sort stability failed: {}", e);
    }

    println!("Stress sort stability (50K rows): PASSED");
}

/// Stress test: Large join correctness.
#[test]
fn stress_join_correctness_large() {
    let Some(ctx) = get_context() else { return };

    // 10K x 10K potential matches with ~50% overlap
    let lk: Vec<u32> = (0..10000).map(|i| i * 2).collect();
    let lv: Vec<u32> = lk.iter().map(|k| k * 10).collect();
    let rk: Vec<u32> = (0..10000).map(|i| i * 2 + (i % 2)).collect();
    let rv: Vec<u32> = rk.iter().map(|k| k * 100).collect();

    if let Err(e) = prop_join_correctness(&ctx, lk, lv, rk, rv) {
        panic!("Large join correctness failed: {}", e);
    }

    println!("Stress join correctness (10K x 10K): PASSED");
}

/// Stress test: Large dedup determinism.
#[test]
fn stress_dedup_determinism_large() {
    let Some(ctx) = get_context() else { return };

    // 50K rows with ~5K unique keys
    let keys: Vec<u32> = (0..50000).map(|i| (i * 13 + 7) % 5000).collect();
    let vals: Vec<u32> = (0..50000).collect();

    if let Err(e) = prop_dedup_determinism(&ctx, keys, vals) {
        panic!("Large dedup determinism failed: {}", e);
    }

    println!("Stress dedup determinism (50K rows): PASSED");
}
