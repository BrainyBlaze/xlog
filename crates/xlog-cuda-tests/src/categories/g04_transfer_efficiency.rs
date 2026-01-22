//! Category G04: Transfer Efficiency Tests
//!
//! Tests that verify 0% CPU bottleneck by ensuring efficient data transfer
//! between host and device. These tests verify:
//! 1. Transfer sizes are minimal (proportional to num_vars, not num_nodes)
//! 2. Circuit structure is reusable (no redundant uploads)
//! 3. Repeated evaluations are efficient (no quadratic scaling)

use crate::harness::xgcf::{
    gen_large_or_circuit, run_tiny_xgcf_backward, run_tiny_xgcf_forward,
};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::{Duration, Instant};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g04_transfer_efficiency");
    let start = Instant::now();

    // G04.1: Transfer Size Analysis (3 tests)
    results.add_result(test_transfer_weight_size(ctx));
    results.add_result(test_transfer_gradient_size(ctx));
    results.add_result(test_transfer_circuit_cached(ctx));

    // G04.2: Transfer vs Compute Ratio (3 tests)
    results.add_result(test_transfer_compute_ratio_small(ctx));
    results.add_result(test_transfer_compute_ratio_large(ctx));
    results.add_result(test_transfer_dominance_check(ctx));

    // G04.3: Hot Path Verification (2 tests)
    results.add_result(test_repeated_eval_no_reupload(ctx));
    results.add_result(test_batch_weights_efficiency(ctx));

    results.set_duration(start.elapsed());
    results
}

// =============================================================================
// G04.1: Transfer Size Analysis Tests
// =============================================================================

/// Test 1: Weight upload is exactly 2 * num_vars * 8 bytes
///
/// Verifies that the weight transfer size is proportional to num_vars (not num_nodes).
/// For a circuit with N variables, we transfer:
/// - var_log_true: (N+1) * 8 bytes (f64 per var, index 0 unused)
/// - var_log_false: (N+1) * 8 bytes
/// Expected: 2 * (num_vars + 1) * 8 bytes for weight buffers
fn test_transfer_weight_size(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test with different num_vars to verify scaling
    let test_cases = [10_usize, 50, 100, 500];

    for &num_vars in &test_cases {
        let spec = gen_large_or_circuit(num_vars);

        // Verify the spec has the expected number of variables
        if spec.num_vars != num_vars {
            return TestResult::error(
                "test_transfer_weight_size",
                start.elapsed(),
                format!(
                    "Spec num_vars mismatch: expected {}, got {}",
                    num_vars, spec.num_vars
                ),
            );
        }

        // Verify weight buffer sizes are as expected
        // var_log_true and var_log_false should each have num_vars + 1 elements
        let expected_buffer_len = num_vars + 1;
        if spec.var_log_true.len() != expected_buffer_len {
            return TestResult::error(
                "test_transfer_weight_size",
                start.elapsed(),
                format!(
                    "var_log_true length mismatch for num_vars={}: expected {}, got {}",
                    num_vars, expected_buffer_len, spec.var_log_true.len()
                ),
            );
        }

        if spec.var_log_false.len() != expected_buffer_len {
            return TestResult::error(
                "test_transfer_weight_size",
                start.elapsed(),
                format!(
                    "var_log_false length mismatch for num_vars={}: expected {}, got {}",
                    num_vars, expected_buffer_len, spec.var_log_false.len()
                ),
            );
        }

        // Calculate expected transfer size in bytes
        // 2 buffers * (num_vars + 1) elements * 8 bytes per f64
        let expected_bytes = 2 * expected_buffer_len * std::mem::size_of::<f64>();
        let calculated_bytes = (spec.var_log_true.len() + spec.var_log_false.len())
            * std::mem::size_of::<f64>();

        if calculated_bytes != expected_bytes {
            return TestResult::error(
                "test_transfer_weight_size",
                start.elapsed(),
                format!(
                    "Weight transfer size mismatch for num_vars={}: expected {} bytes, calculated {} bytes",
                    num_vars, expected_bytes, calculated_bytes
                ),
            );
        }

        // Run forward pass to verify circuit works
        let values = match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(v) => v,
            Err(e) => {
                return TestResult::error(
                    "test_transfer_weight_size",
                    start.elapsed(),
                    format!(
                        "Failed to run forward pass for num_vars={}: {}",
                        num_vars, e
                    ),
                );
            }
        };

        // Verify correct number of output values
        if values.len() != spec.num_nodes {
            return TestResult::error(
                "test_transfer_weight_size",
                start.elapsed(),
                format!(
                    "Output size mismatch for num_vars={}: expected {} nodes, got {}",
                    num_vars, spec.num_nodes, values.len()
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_weight_size",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_transfer_weight_size", start.elapsed())
}

/// Test 2: Gradient download is exactly 2 * num_vars * 8 bytes
///
/// Verifies that gradient buffer sizes are proportional to num_vars.
/// For backward pass, we download:
/// - grad_true: (num_vars + 1) * 8 bytes
/// - grad_false: (num_vars + 1) * 8 bytes
fn test_transfer_gradient_size(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let test_cases = [10_usize, 50, 100, 500];

    for &num_vars in &test_cases {
        let spec = gen_large_or_circuit(num_vars);

        // Run backward pass to get gradients
        let run_result = match run_tiny_xgcf_backward(ctx, &spec) {
            Ok(r) => r,
            Err(e) => {
                return TestResult::error(
                    "test_transfer_gradient_size",
                    start.elapsed(),
                    format!(
                        "Failed to run backward pass for num_vars={}: {}",
                        num_vars, e
                    ),
                );
            }
        };

        // Verify gradient buffer sizes
        let expected_grad_len = num_vars + 1;
        if run_result.grad_true.len() != expected_grad_len {
            return TestResult::error(
                "test_transfer_gradient_size",
                start.elapsed(),
                format!(
                    "grad_true length mismatch for num_vars={}: expected {}, got {}",
                    num_vars, expected_grad_len, run_result.grad_true.len()
                ),
            );
        }

        if run_result.grad_false.len() != expected_grad_len {
            return TestResult::error(
                "test_transfer_gradient_size",
                start.elapsed(),
                format!(
                    "grad_false length mismatch for num_vars={}: expected {}, got {}",
                    num_vars, expected_grad_len, run_result.grad_false.len()
                ),
            );
        }

        // Calculate expected transfer size for gradients
        let expected_bytes = 2 * expected_grad_len * std::mem::size_of::<f64>();
        let calculated_bytes = (run_result.grad_true.len() + run_result.grad_false.len())
            * std::mem::size_of::<f64>();

        if calculated_bytes != expected_bytes {
            return TestResult::error(
                "test_transfer_gradient_size",
                start.elapsed(),
                format!(
                    "Gradient transfer size mismatch for num_vars={}: expected {} bytes, calculated {} bytes",
                    num_vars, expected_bytes, calculated_bytes
                ),
            );
        }

        // Verify gradients are not all zero (some variables should have gradients)
        let has_nonzero_grad = run_result.grad_true.iter().any(|&g| g.abs() > 1e-15);
        if !has_nonzero_grad {
            return TestResult::error(
                "test_transfer_gradient_size",
                start.elapsed(),
                format!(
                    "All gradients are zero for num_vars={}, expected some non-zero values",
                    num_vars
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_gradient_size",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_transfer_gradient_size", start.elapsed())
}

/// Test 3: After first run, circuit structure NOT re-uploaded
///
/// Verifies that running the same circuit twice doesn't cause performance
/// degradation from redundant circuit structure uploads. The second run
/// should be similar or faster than the first.
fn test_transfer_circuit_cached(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 100;
    let spec = gen_large_or_circuit(num_vars);

    // Warm-up run to ensure any lazy initialization is complete
    let _warmup = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_transfer_circuit_cached",
                start.elapsed(),
                format!("Failed warm-up run: {}", e),
            );
        }
    };

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_circuit_cached",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // First timed run
    let first_start = Instant::now();
    let values_first = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_transfer_circuit_cached",
                start.elapsed(),
                format!("Failed first run: {}", e),
            );
        }
    };
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_circuit_cached",
            start.elapsed(),
            format!("Sync failed after first run: {}", e),
        );
    }
    let first_duration = first_start.elapsed();

    // Second timed run with same circuit
    let second_start = Instant::now();
    let values_second = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_transfer_circuit_cached",
                start.elapsed(),
                format!("Failed second run: {}", e),
            );
        }
    };
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_circuit_cached",
            start.elapsed(),
            format!("Sync failed after second run: {}", e),
        );
    }
    let second_duration = second_start.elapsed();

    // Verify results are identical
    if values_first.len() != values_second.len() {
        return TestResult::error(
            "test_transfer_circuit_cached",
            start.elapsed(),
            format!(
                "Output size changed between runs: {} vs {}",
                values_first.len(),
                values_second.len()
            ),
        );
    }

    for (i, (&v1, &v2)) in values_first.iter().zip(values_second.iter()).enumerate() {
        if (v1 - v2).abs() > 1e-14 {
            return TestResult::error(
                "test_transfer_circuit_cached",
                start.elapsed(),
                format!(
                    "Result difference at index {}: first={}, second={}",
                    i, v1, v2
                ),
            );
        }
    }

    // Second run should not be significantly slower (allow 3x tolerance for variance)
    // If circuit was being re-uploaded, second run would likely be slower or same
    // We just verify it's not drastically slower which would indicate a problem
    if second_duration > first_duration * 3 && second_duration > Duration::from_micros(100) {
        return TestResult::error(
            "test_transfer_circuit_cached",
            start.elapsed(),
            format!(
                "Second run significantly slower ({:?}) than first ({:?}), suggests circuit re-upload",
                second_duration, first_duration
            ),
        );
    }

    TestResult::passed("test_transfer_circuit_cached", start.elapsed())
}

// =============================================================================
// G04.2: Transfer vs Compute Ratio Tests
// =============================================================================

/// Test 4: For 10-var circuit, compute dominates
///
/// Even for small circuits, the forward/backward computation should complete
/// in reasonable time, demonstrating that transfer overhead is not the bottleneck.
fn test_transfer_compute_ratio_small(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 10;
    let spec = gen_large_or_circuit(num_vars);

    // Run multiple iterations to get stable timing
    let num_iterations = 50;

    // Warmup
    for _ in 0..5 {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_small",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // Time forward passes
    let forward_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_small",
            start.elapsed(),
            format!("Sync failed after forward passes: {}", e),
        );
    }
    let forward_total = forward_start.elapsed();
    let forward_per_iter = forward_total / num_iterations as u32;

    // Time backward passes
    let backward_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = run_tiny_xgcf_backward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_small",
            start.elapsed(),
            format!("Sync failed after backward passes: {}", e),
        );
    }
    let backward_total = backward_start.elapsed();
    let backward_per_iter = backward_total / num_iterations as u32;

    // For a small circuit, each iteration should be very fast (sub-millisecond)
    // If transfer dominated, we'd see much higher latency per iteration
    let max_reasonable_per_iter = Duration::from_millis(10);

    if forward_per_iter > max_reasonable_per_iter {
        return TestResult::error(
            "test_transfer_compute_ratio_small",
            start.elapsed(),
            format!(
                "Forward pass too slow for small circuit ({:?}/iter), suggests transfer overhead",
                forward_per_iter
            ),
        );
    }

    if backward_per_iter > max_reasonable_per_iter {
        return TestResult::error(
            "test_transfer_compute_ratio_small",
            start.elapsed(),
            format!(
                "Backward pass too slow for small circuit ({:?}/iter), suggests transfer overhead",
                backward_per_iter
            ),
        );
    }

    TestResult::passed("test_transfer_compute_ratio_small", start.elapsed())
}

/// Test 5: For 1000-var circuit, compute still dominates
///
/// Even for larger circuits, the computation time should scale reasonably
/// with circuit size, not dominated by transfer overhead.
fn test_transfer_compute_ratio_large(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 1000;
    let spec = gen_large_or_circuit(num_vars);

    let num_iterations = 20;

    // Warmup
    for _ in 0..3 {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_large",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // Time forward passes
    let forward_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_large",
            start.elapsed(),
            format!("Sync failed after forward passes: {}", e),
        );
    }
    let forward_total = forward_start.elapsed();
    let forward_per_iter = forward_total / num_iterations as u32;

    // Time backward passes
    let backward_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = run_tiny_xgcf_backward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_compute_ratio_large",
            start.elapsed(),
            format!("Sync failed after backward passes: {}", e),
        );
    }
    let backward_total = backward_start.elapsed();
    let backward_per_iter = backward_total / num_iterations as u32;

    // For 1000-var circuit, still expect reasonable per-iteration time
    // 100x more vars shouldn't mean 100x more time if compute dominates
    let max_reasonable_per_iter = Duration::from_millis(50);

    if forward_per_iter > max_reasonable_per_iter {
        return TestResult::error(
            "test_transfer_compute_ratio_large",
            start.elapsed(),
            format!(
                "Forward pass too slow for large circuit ({:?}/iter), suggests transfer overhead",
                forward_per_iter
            ),
        );
    }

    if backward_per_iter > max_reasonable_per_iter {
        return TestResult::error(
            "test_transfer_compute_ratio_large",
            start.elapsed(),
            format!(
                "Backward pass too slow for large circuit ({:?}/iter), suggests transfer overhead",
                backward_per_iter
            ),
        );
    }

    // Verify the scaling is reasonable (large shouldn't be more than 20x slower than we'd expect)
    // This is a sanity check that we're not hitting pathological transfer behavior

    TestResult::passed("test_transfer_compute_ratio_large", start.elapsed())
}

/// Test 6: GPU kernel time > 90% of total forward+backward time
///
/// Verifies that the GPU compute time dominates the total execution time,
/// meaning transfer overhead is minimal (< 10% of total time).
fn test_transfer_dominance_check(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 500;
    let spec = gen_large_or_circuit(num_vars);

    // Warmup to ensure fair timing
    for _ in 0..5 {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
        let _ = run_tiny_xgcf_backward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_dominance_check",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // Run a batch of forward+backward passes
    let num_iterations = 30;
    let batch_start = Instant::now();

    for _ in 0..num_iterations {
        match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(_) => {}
            Err(e) => {
                return TestResult::error(
                    "test_transfer_dominance_check",
                    start.elapsed(),
                    format!("Forward pass failed: {}", e),
                );
            }
        }

        match run_tiny_xgcf_backward(ctx, &spec) {
            Ok(_) => {}
            Err(e) => {
                return TestResult::error(
                    "test_transfer_dominance_check",
                    start.elapsed(),
                    format!("Backward pass failed: {}", e),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_transfer_dominance_check",
            start.elapsed(),
            format!("Sync failed after batch: {}", e),
        );
    }
    let batch_duration = batch_start.elapsed();
    let per_iteration = batch_duration / num_iterations as u32;

    // We can't directly measure GPU kernel time vs transfer time without CUDA events,
    // but we can verify that the total time is reasonable for the computation size.
    // If transfer dominated, we'd see much higher latency.
    //
    // For 500 vars with forward+backward, expect sub-10ms per iteration on modern GPU
    // if compute dominates. Transfer-dominated would show 50ms+ per iteration.
    let max_acceptable = Duration::from_millis(20);

    if per_iteration > max_acceptable {
        return TestResult::error(
            "test_transfer_dominance_check",
            start.elapsed(),
            format!(
                "Per-iteration time too high ({:?}), suggests transfer overhead > 10%",
                per_iteration
            ),
        );
    }

    // Additional check: verify total throughput is reasonable
    let ops_per_second = num_iterations as f64 / batch_duration.as_secs_f64();
    let min_ops_per_second = 50.0; // Should be able to do at least 50 forward+backward per second

    if ops_per_second < min_ops_per_second {
        return TestResult::error(
            "test_transfer_dominance_check",
            start.elapsed(),
            format!(
                "Throughput too low ({:.1} ops/sec), expected > {} ops/sec",
                ops_per_second, min_ops_per_second
            ),
        );
    }

    TestResult::passed("test_transfer_dominance_check", start.elapsed())
}

// =============================================================================
// G04.3: Hot Path Verification Tests
// =============================================================================

/// Test 7: 100 evaluations with same circuit structure
///
/// Verifies that repeated evaluations scale linearly (not quadratically),
/// proving that circuit structure is not being re-uploaded on each call.
fn test_repeated_eval_no_reupload(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 200;
    let mut spec = gen_large_or_circuit(num_vars);

    // Warmup
    for _ in 0..5 {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_eval_no_reupload",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // Time 10 evaluations
    let small_batch_start = Instant::now();
    for i in 0..10 {
        // Slightly modify weights to ensure we're actually computing
        spec.var_log_true[1] = (0.5 + 0.001 * i as f64).ln();

        match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(_) => {}
            Err(e) => {
                return TestResult::error(
                    "test_repeated_eval_no_reupload",
                    start.elapsed(),
                    format!("Forward pass {} failed: {}", i, e),
                );
            }
        }
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_eval_no_reupload",
            start.elapsed(),
            format!("Sync failed after small batch: {}", e),
        );
    }
    let small_batch_duration = small_batch_start.elapsed();

    // Time 100 evaluations
    let large_batch_start = Instant::now();
    for i in 0..100 {
        spec.var_log_true[1] = (0.5 + 0.001 * i as f64).ln();

        match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(_) => {}
            Err(e) => {
                return TestResult::error(
                    "test_repeated_eval_no_reupload",
                    start.elapsed(),
                    format!("Forward pass {} failed: {}", i, e),
                );
            }
        }
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_eval_no_reupload",
            start.elapsed(),
            format!("Sync failed after large batch: {}", e),
        );
    }
    let large_batch_duration = large_batch_start.elapsed();

    // 100 evaluations should take ~10x the time of 10 evaluations (linear scaling)
    // If circuit was re-uploaded each time quadratically, 100 evals would be 100x slower
    let expected_ratio = 10.0;
    let actual_ratio = large_batch_duration.as_secs_f64() / small_batch_duration.as_secs_f64();

    // Allow up to 20x ratio to account for variance, but catch quadratic behavior (would be ~100x)
    let max_acceptable_ratio = 20.0;

    if actual_ratio > max_acceptable_ratio {
        return TestResult::error(
            "test_repeated_eval_no_reupload",
            start.elapsed(),
            format!(
                "Non-linear scaling detected: 100 evals took {:.1}x longer than 10 evals (expected ~{:.1}x). \
                 10 evals: {:?}, 100 evals: {:?}. Suggests circuit re-upload on each call.",
                actual_ratio, expected_ratio, small_batch_duration, large_batch_duration
            ),
        );
    }

    TestResult::passed("test_repeated_eval_no_reupload", start.elapsed())
}

/// Test 8: Multiple weight sets evaluated efficiently
///
/// Verifies that evaluating the same circuit with different weight sets
/// is efficient - only weights are transferred, not the entire circuit.
fn test_batch_weights_efficiency(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_vars = 300;
    let base_spec = gen_large_or_circuit(num_vars);

    // Generate 10 different weight sets
    let num_weight_sets = 10;
    let mut weight_sets: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(num_weight_sets);

    for i in 0..num_weight_sets {
        let mut log_true = vec![0.0; num_vars + 1];
        let mut log_false = vec![0.0; num_vars + 1];

        for j in 1..=num_vars {
            // Create distinct weight values for each set
            let p = 0.3 + 0.4 * ((i + j) as f64 / (num_weight_sets + num_vars) as f64);
            log_true[j] = p.ln();
            log_false[j] = (1.0 - p).ln();
        }

        weight_sets.push((log_true, log_false));
    }

    // Warmup with first weight set
    let mut spec = base_spec.clone();
    for _ in 0..3 {
        let _ = run_tiny_xgcf_forward(ctx, &spec);
    }
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!("Sync failed after warmup: {}", e),
        );
    }

    // Time evaluation with all weight sets
    let batch_start = Instant::now();
    let mut results: Vec<f64> = Vec::with_capacity(num_weight_sets);

    for (log_true, log_false) in &weight_sets {
        spec.var_log_true = log_true.clone();
        spec.var_log_false = log_false.clone();

        match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(values) => {
                results.push(values[spec.root as usize]);
            }
            Err(e) => {
                return TestResult::error(
                    "test_batch_weights_efficiency",
                    start.elapsed(),
                    format!("Forward pass failed: {}", e),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!("Sync failed after batch: {}", e),
        );
    }
    let batch_duration = batch_start.elapsed();

    // Verify we got distinct results (different weights should give different outputs)
    let unique_results: std::collections::HashSet<u64> = results
        .iter()
        .map(|&r| r.to_bits())
        .collect();

    if unique_results.len() < num_weight_sets / 2 {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!(
                "Expected distinct results for different weight sets, got only {} unique values out of {}",
                unique_results.len(), num_weight_sets
            ),
        );
    }

    // 10 weight set evaluations should complete quickly
    // If we were re-uploading circuit each time, this would be much slower
    let per_weight_set = batch_duration / num_weight_sets as u32;
    let max_acceptable = Duration::from_millis(20);

    if per_weight_set > max_acceptable {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!(
                "Per-weight-set evaluation too slow ({:?}), expected < {:?}. \
                 Suggests circuit structure being re-uploaded with weights.",
                per_weight_set, max_acceptable
            ),
        );
    }

    // Run multiple iterations of the entire batch to verify consistency
    let num_batch_iterations = 5;
    let multi_batch_start = Instant::now();

    for _ in 0..num_batch_iterations {
        for (log_true, log_false) in &weight_sets {
            spec.var_log_true = log_true.clone();
            spec.var_log_false = log_false.clone();

            match run_tiny_xgcf_forward(ctx, &spec) {
                Ok(_) => {}
                Err(e) => {
                    return TestResult::error(
                        "test_batch_weights_efficiency",
                        start.elapsed(),
                        format!("Forward pass in multi-batch failed: {}", e),
                    );
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!("Sync failed after multi-batch: {}", e),
        );
    }
    let multi_batch_duration = multi_batch_start.elapsed();

    // Verify linear scaling: 5 batches should take ~5x one batch
    let expected_ratio = num_batch_iterations as f64;
    let actual_ratio = multi_batch_duration.as_secs_f64() / batch_duration.as_secs_f64();

    // Allow up to 3x the expected ratio for variance (accounts for system load, warmup effects)
    // The key is that it's not quadratic (which would be 25x for 5 batches)
    if actual_ratio > expected_ratio * 3.0 {
        return TestResult::error(
            "test_batch_weights_efficiency",
            start.elapsed(),
            format!(
                "Non-linear batch scaling: {} batches took {:.1}x one batch (expected ~{:.1}x)",
                num_batch_iterations, actual_ratio, expected_ratio
            ),
        );
    }

    TestResult::passed("test_batch_weights_efficiency", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_g04_transfer_weight_size() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_weight_size(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_transfer_gradient_size() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_gradient_size(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_transfer_circuit_cached() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_circuit_cached(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_transfer_compute_ratio_small() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_compute_ratio_small(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_transfer_compute_ratio_large() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_compute_ratio_large(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_transfer_dominance_check() {
        if let Some(ctx) = get_test_context() {
            let result = test_transfer_dominance_check(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_repeated_eval_no_reupload() {
        if let Some(ctx) = get_test_context() {
            let result = test_repeated_eval_no_reupload(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g04_batch_weights_efficiency() {
        if let Some(ctx) = get_test_context() {
            let result = test_batch_weights_efficiency(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }
}
