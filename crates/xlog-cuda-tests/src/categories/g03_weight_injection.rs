//! Weight injection pattern tests
//!
//! Tests the weight buffer operations and value handling for XGCF circuits.
//! These tests verify that weights are correctly uploaded, reused, mapped to
//! variable indices, and handle extreme values without numerical issues.

use crate::harness::xgcf::{
    gen_large_or_circuit, gen_or_circuit, gen_single_lit_circuit, run_tiny_xgcf_forward,
    TinyXgcfSpec,
};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("weight_injection_patterns");
    let start = Instant::now();

    // Weight buffer operation checks.
    results.add_result(test_weight_upload_integrity(ctx));
    results.add_result(test_weight_buffer_reuse(ctx));
    results.add_result(test_weight_variable_mapping(ctx));

    // Weight value handling checks.
    results.add_result(test_weight_extreme_values(ctx));
    results.add_result(test_weight_uniform_distribution(ctx));
    results.add_result(test_weight_sparse_nonzero(ctx));

    results.set_duration(start.elapsed());
    results
}

// =============================================================================
// Weight buffer operation tests
// =============================================================================

/// Test 1: Weight upload integrity
///
/// Upload weights, verify exact values preserved after forward pass.
/// This test creates a circuit with known weights, runs forward pass,
/// and verifies the output matches expected values (proving weights uploaded correctly).
fn test_weight_upload_integrity(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create a simple literal circuit with specific weight values
    let mut spec = gen_single_lit_circuit(1);

    // Set very specific weight values that are easy to verify
    let p_true = 0.73_f64;
    let p_false = 0.27_f64;
    spec.var_log_true[1] = p_true.ln();
    spec.var_log_false[1] = p_false.ln();

    // Expected value for positive literal is var_log_true[1]
    let expected_output = spec.var_log_true[1];

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_upload_integrity",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify the output matches the expected weight value exactly
    let actual_output = values[spec.root as usize];
    let diff = (actual_output - expected_output).abs();

    if diff > 1e-14 {
        return TestResult::error(
            "test_weight_upload_integrity",
            start.elapsed(),
            format!(
                "Weight upload integrity failed: expected {} (log({:.6})), got {} (diff={})",
                expected_output, p_true, actual_output, diff
            ),
        );
    }

    // Test with a second set of weights to ensure upload is working
    let p_true_2 = 0.123456789_f64;
    spec.var_log_true[1] = p_true_2.ln();

    let values_2 = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_upload_integrity",
                start.elapsed(),
                format!("Failed to run second forward pass: {}", e),
            );
        }
    };

    let expected_2 = p_true_2.ln();
    let actual_2 = values_2[spec.root as usize];
    let diff_2 = (actual_2 - expected_2).abs();

    if diff_2 > 1e-14 {
        return TestResult::error(
            "test_weight_upload_integrity",
            start.elapsed(),
            format!(
                "Weight upload integrity (second pass) failed: expected {}, got {} (diff={})",
                expected_2, actual_2, diff_2
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_upload_integrity",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_upload_integrity", start.elapsed())
}

/// Test 2: Weight buffer reuse
///
/// Run multiple evaluations with different weights on same circuit structure.
/// Verifies that the weight buffer is properly updated between runs.
fn test_weight_buffer_reuse(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create an OR circuit: root = OR(Lit(+1), Lit(+2))
    let mut spec = gen_or_circuit();

    // First set of weights (sum != 1 to get distinct results)
    let p1_a = 0.7_f64;
    let p2_a = 0.6_f64;
    spec.var_log_true[1] = p1_a.ln();
    spec.var_log_true[2] = p2_a.ln();
    spec.var_log_false[1] = (1.0 - p1_a).ln();
    spec.var_log_false[2] = (1.0 - p2_a).ln();

    let values_a = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_buffer_reuse",
                start.elapsed(),
                format!("Failed to run first forward pass: {}", e),
            );
        }
    };

    // Compute expected OR result: logsumexp(log(p1_a), log(p2_a))
    let v1_a = p1_a.ln();
    let v2_a = p2_a.ln();
    let max_a = v1_a.max(v2_a);
    let expected_a = max_a + ((v1_a - max_a).exp() + (v2_a - max_a).exp()).ln();
    let actual_a = values_a[spec.root as usize];

    if (actual_a - expected_a).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_buffer_reuse",
            start.elapsed(),
            format!(
                "First evaluation failed: expected {}, got {} (diff={})",
                expected_a,
                actual_a,
                (actual_a - expected_a).abs()
            ),
        );
    }

    // Second set of weights (significantly different, sum != 1)
    let p1_b = 0.2_f64;
    let p2_b = 0.3_f64;
    spec.var_log_true[1] = p1_b.ln();
    spec.var_log_true[2] = p2_b.ln();
    spec.var_log_false[1] = (1.0 - p1_b).ln();
    spec.var_log_false[2] = (1.0 - p2_b).ln();

    let values_b = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_buffer_reuse",
                start.elapsed(),
                format!("Failed to run second forward pass: {}", e),
            );
        }
    };

    // Compute expected OR result for second weights
    let v1_b = p1_b.ln();
    let v2_b = p2_b.ln();
    let max_b = v1_b.max(v2_b);
    let expected_b = max_b + ((v1_b - max_b).exp() + (v2_b - max_b).exp()).ln();
    let actual_b = values_b[spec.root as usize];

    if (actual_b - expected_b).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_buffer_reuse",
            start.elapsed(),
            format!(
                "Second evaluation failed: expected {}, got {} (diff={})",
                expected_b,
                actual_b,
                (actual_b - expected_b).abs()
            ),
        );
    }

    // Verify the results are different (proves buffer was updated)
    // First: log(0.7+0.6) = log(1.3), Second: log(0.2+0.3) = log(0.5)
    // These should be clearly different
    if (actual_a - actual_b).abs() < 0.1 {
        return TestResult::error(
            "test_weight_buffer_reuse",
            start.elapsed(),
            format!(
                "Results should differ between weight sets: first={}, second={} (diff={})",
                actual_a,
                actual_b,
                (actual_a - actual_b).abs()
            ),
        );
    }

    // Third evaluation: revert to first weights to confirm proper switching
    spec.var_log_true[1] = p1_a.ln();
    spec.var_log_true[2] = p2_a.ln();
    spec.var_log_false[1] = (1.0 - p1_a).ln();
    spec.var_log_false[2] = (1.0 - p2_a).ln();

    let values_c = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_buffer_reuse",
                start.elapsed(),
                format!("Failed to run third forward pass: {}", e),
            );
        }
    };

    let actual_c = values_c[spec.root as usize];
    if (actual_c - expected_a).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_buffer_reuse",
            start.elapsed(),
            format!(
                "Third evaluation (revert to first) failed: expected {}, got {} (diff={})",
                expected_a,
                actual_c,
                (actual_c - expected_a).abs()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_buffer_reuse",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_buffer_reuse", start.elapsed())
}

/// Test 3: Weight variable mapping
///
/// Verify DIMACS variable indices map correctly to buffer positions.
/// Creates a circuit with sparse variable indices [1, 3, 5] and verifies
/// that weights at those specific positions are used correctly.
fn test_weight_variable_mapping(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const AND: u8 = 3;

    // Create a circuit with variables [1, 3, 5] (sparse indices)
    // Circuit: AND(Lit(+1), Lit(+3), Lit(+5))
    let num_nodes = 4;
    let num_vars = 5; // Variables 1-5, but we only use 1, 3, 5
    let root = 3;

    let node_type = vec![LIT, LIT, LIT, AND];
    let child_offsets = vec![0, 0, 0, 0, 3];
    let child_indices = vec![0, 1, 2];
    let lit = vec![1, 3, 5, 0]; // Literals for vars 1, 3, 5
    let decision_var = vec![0, 0, 0, 0];
    let decision_child_false = vec![0, 0, 0, 0];
    let decision_child_true = vec![0, 0, 0, 0];
    let level_nodes = vec![0, 1, 2, 3];
    let levels = vec![(0, 3), (3, 1)];

    // Set distinct weights for each variable
    let p1 = 0.8_f64;
    let p2 = 0.5_f64; // var 2 - should NOT be used
    let p3 = 0.6_f64;
    let p4 = 0.5_f64; // var 4 - should NOT be used
    let p5 = 0.4_f64;

    let var_log_true = vec![0.0, p1.ln(), p2.ln(), p3.ln(), p4.ln(), p5.ln()];
    let var_log_false = vec![
        0.0,
        (1.0 - p1).ln(),
        (1.0 - p2).ln(),
        (1.0 - p3).ln(),
        (1.0 - p4).ln(),
        (1.0 - p5).ln(),
    ];

    // Expected: AND in log space = sum of children = log(p1) + log(p3) + log(p5)
    let expected_root = p1.ln() + p3.ln() + p5.ln();

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values: vec![p1.ln(), p3.ln(), p5.ln(), expected_root],
        expected_grad_true: vec![0.0; num_vars + 1],
        expected_grad_false: vec![0.0; num_vars + 1],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_variable_mapping",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify each literal uses the correct variable's weight
    let lit_1_expected = p1.ln();
    let lit_3_expected = p3.ln();
    let lit_5_expected = p5.ln();

    if (values[0] - lit_1_expected).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!(
                "Lit(+1) used wrong weight: expected {} (log({:.2})), got {}",
                lit_1_expected, p1, values[0]
            ),
        );
    }

    if (values[1] - lit_3_expected).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!(
                "Lit(+3) used wrong weight: expected {} (log({:.2})), got {}",
                lit_3_expected, p3, values[1]
            ),
        );
    }

    if (values[2] - lit_5_expected).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!(
                "Lit(+5) used wrong weight: expected {} (log({:.2})), got {}",
                lit_5_expected, p5, values[2]
            ),
        );
    }

    // Verify the AND root
    let actual_root = values[root as usize];
    if (actual_root - expected_root).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!(
                "AND root with sparse vars [1,3,5] failed: expected {}, got {} (diff={})",
                expected_root,
                actual_root,
                (actual_root - expected_root).abs()
            ),
        );
    }

    // Verify that changing an unused variable (e.g., var 2) doesn't affect the result
    let mut spec_modified = spec.clone();
    spec_modified.var_log_true[2] = 0.001_f64.ln(); // Drastically change var 2
    spec_modified.var_log_true[4] = 0.999_f64.ln(); // Drastically change var 4

    let values_modified = match run_tiny_xgcf_forward(ctx, &spec_modified) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_variable_mapping",
                start.elapsed(),
                format!("Failed to run modified forward pass: {}", e),
            );
        }
    };

    // Result should be identical since vars 2 and 4 are not used
    let actual_modified = values_modified[root as usize];
    if (actual_modified - expected_root).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!(
                "Changing unused vars affected result: expected {}, got {} (diff={})",
                expected_root,
                actual_modified,
                (actual_modified - expected_root).abs()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_variable_mapping",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_variable_mapping", start.elapsed())
}

// =============================================================================
// Weight value handling tests
// =============================================================================

/// Test 4: Weight extreme values
///
/// Tests that log(1e-308) and log(0.9999...) don't cause overflow/underflow.
/// Verifies numerical stability with extreme weight values.
fn test_weight_extreme_values(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const AND: u8 = 3;

    // Test 1: Very small probability (near floating point minimum)
    // log(1e-308) ~ -709
    let mut spec = gen_single_lit_circuit(1);
    let tiny_prob = 1e-308_f64;
    spec.var_log_true[1] = tiny_prob.ln();
    spec.var_log_false[1] = (1.0 - tiny_prob).ln(); // Effectively 0.0 (log(1))

    let values_tiny = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_extreme_values",
                start.elapsed(),
                format!("Failed to run forward pass with tiny weight: {}", e),
            );
        }
    };

    // Verify result is finite (not NaN or unexpected Inf)
    let result_tiny = values_tiny[spec.root as usize];
    if result_tiny.is_nan() {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            "Result is NaN with tiny weight (1e-308)".to_string(),
        );
    }

    // Should be a large negative number, not positive infinity
    if result_tiny.is_infinite() && result_tiny.is_sign_positive() {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            "Result is +Inf with tiny weight (1e-308)".to_string(),
        );
    }

    // Verify the actual value is close to expected
    let expected_tiny = tiny_prob.ln();
    if (result_tiny - expected_tiny).abs() > 1e-10 {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            format!(
                "Tiny weight result mismatch: expected {}, got {} (diff={})",
                expected_tiny,
                result_tiny,
                (result_tiny - expected_tiny).abs()
            ),
        );
    }

    // Test 2: Very large probability (near 1.0)
    let near_one_prob = 1.0 - 1e-15_f64;
    spec.var_log_true[1] = near_one_prob.ln(); // Very close to 0
    spec.var_log_false[1] = 1e-15_f64.ln(); // Very negative

    let values_near_one = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_extreme_values",
                start.elapsed(),
                format!("Failed to run forward pass with near-one weight: {}", e),
            );
        }
    };

    let result_near_one = values_near_one[spec.root as usize];
    if result_near_one.is_nan() {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            "Result is NaN with near-one weight (1-1e-15)".to_string(),
        );
    }

    if result_near_one.is_infinite() {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            format!("Result is Inf with near-one weight: {}", result_near_one),
        );
    }

    let expected_near_one = near_one_prob.ln();
    if (result_near_one - expected_near_one).abs() > 1e-14 {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            format!(
                "Near-one weight result mismatch: expected {}, got {} (diff={})",
                expected_near_one,
                result_near_one,
                (result_near_one - expected_near_one).abs()
            ),
        );
    }

    // Test 3: Combination of extremes in an AND circuit
    let spec_extreme = TinyXgcfSpec {
        num_nodes: 3,
        num_vars: 2,
        root: 2,
        node_type: vec![LIT, LIT, AND],
        child_offsets: vec![0, 0, 0, 2],
        child_indices: vec![0, 1],
        lit: vec![1, 2, 0],
        decision_var: vec![0, 0, 0],
        decision_child_false: vec![0, 0, 0],
        decision_child_true: vec![0, 0, 0],
        level_nodes: vec![0, 1, 2],
        levels: vec![(0, 2), (2, 1)],
        var_log_true: vec![0.0, 1e-200_f64.ln(), (1.0 - 1e-14_f64).ln()],
        var_log_false: vec![0.0, (1.0 - 1e-200_f64).ln(), 1e-14_f64.ln()],
        expected_values: vec![
            1e-200_f64.ln(),
            (1.0 - 1e-14_f64).ln(),
            1e-200_f64.ln() + (1.0 - 1e-14_f64).ln(),
        ],
        expected_grad_true: vec![0.0, 0.0, 0.0],
        expected_grad_false: vec![0.0, 0.0, 0.0],
    };

    let values_extreme = match run_tiny_xgcf_forward(ctx, &spec_extreme) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_extreme_values",
                start.elapsed(),
                format!("Failed to run forward pass with extreme AND: {}", e),
            );
        }
    };

    for (i, &val) in values_extreme.iter().enumerate() {
        if val.is_nan() {
            return TestResult::error(
                "test_weight_extreme_values",
                start.elapsed(),
                format!("Node {} is NaN with extreme values", i),
            );
        }
        // Positive infinity is invalid; negative infinity may be valid for log(0)
        if val.is_infinite() && val.is_sign_positive() {
            return TestResult::error(
                "test_weight_extreme_values",
                start.elapsed(),
                format!("Node {} is +Inf with extreme values", i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_extreme_values",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_extreme_values", start.elapsed())
}

/// Test 5: Weight uniform distribution
///
/// All weights equal produces correct uniform probability.
/// For an OR of N equally-weighted children, result is log(p) + log(N).
fn test_weight_uniform_distribution(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test with 2-way OR with p=0.5 for all variables
    let mut spec_2 = gen_or_circuit();
    let p = 0.5_f64;
    spec_2.var_log_true[1] = p.ln();
    spec_2.var_log_true[2] = p.ln();
    spec_2.var_log_false[1] = (1.0 - p).ln();
    spec_2.var_log_false[2] = (1.0 - p).ln();

    let values_2 = match run_tiny_xgcf_forward(ctx, &spec_2) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_uniform_distribution",
                start.elapsed(),
                format!("Failed to run forward pass for 2-way OR: {}", e),
            );
        }
    };

    // Expected: log(0.5) + ln(2) = log(0.5 * 2) = log(1) = 0
    // More precisely: logsumexp(log(0.5), log(0.5)) = log(0.5) + ln(2)
    let expected_2 = p.ln() + 2.0_f64.ln();
    let actual_2 = values_2[spec_2.root as usize];

    if (actual_2 - expected_2).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_uniform_distribution",
            start.elapsed(),
            format!(
                "2-way uniform OR failed: expected {} (log(0.5) + ln(2)), got {} (diff={})",
                expected_2,
                actual_2,
                (actual_2 - expected_2).abs()
            ),
        );
    }

    // Test with larger OR (10-way) with p=0.5
    let spec_10 = gen_large_or_circuit(10);
    // gen_large_or_circuit already uses p=0.5

    let values_10 = match run_tiny_xgcf_forward(ctx, &spec_10) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_uniform_distribution",
                start.elapsed(),
                format!("Failed to run forward pass for 10-way OR: {}", e),
            );
        }
    };

    // Expected: log(0.5) + ln(10)
    let expected_10 = p.ln() + 10.0_f64.ln();
    let actual_10 = values_10[spec_10.root as usize];

    if (actual_10 - expected_10).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_uniform_distribution",
            start.elapsed(),
            format!(
                "10-way uniform OR failed: expected {} (log(0.5) + ln(10)), got {} (diff={})",
                expected_10,
                actual_10,
                (actual_10 - expected_10).abs()
            ),
        );
    }

    // Test with 100-way OR with p=0.5
    let spec_100 = gen_large_or_circuit(100);

    let values_100 = match run_tiny_xgcf_forward(ctx, &spec_100) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_uniform_distribution",
                start.elapsed(),
                format!("Failed to run forward pass for 100-way OR: {}", e),
            );
        }
    };

    // Expected: log(0.5) + ln(100)
    let expected_100 = p.ln() + 100.0_f64.ln();
    let actual_100 = values_100[spec_100.root as usize];

    if (actual_100 - expected_100).abs() > 1e-11 {
        return TestResult::error(
            "test_weight_uniform_distribution",
            start.elapsed(),
            format!(
                "100-way uniform OR failed: expected {} (log(0.5) + ln(100)), got {} (diff={})",
                expected_100,
                actual_100,
                (actual_100 - expected_100).abs()
            ),
        );
    }

    // Verify all children have equal values in the 10-way case
    for i in 0..10 {
        let child_val = values_10[i];
        let expected_child = p.ln();
        if (child_val - expected_child).abs() > 1e-14 {
            return TestResult::error(
                "test_weight_uniform_distribution",
                start.elapsed(),
                format!(
                    "Child {} of uniform OR has wrong value: expected {}, got {}",
                    i, expected_child, child_val
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_uniform_distribution",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_uniform_distribution", start.elapsed())
}

/// Test 6: Weight sparse nonzero
///
/// Most weights log(0) = -inf, only few non-zero.
/// Verifies that only the non-zero weight contributes to the result.
fn test_weight_sparse_nonzero(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const OR: u8 = 4;

    // Create an OR circuit with 10 children, but only one has non-zero weight
    let num_vars = 10;
    let num_nodes = num_vars + 1;
    let root = num_vars as u32;

    let mut node_type = vec![LIT; num_vars];
    node_type.push(OR);

    let mut child_offsets: Vec<u32> = (0..=num_vars).map(|_| 0).collect();
    child_offsets.push(num_vars as u32);

    let child_indices: Vec<u32> = (0..num_vars as u32).collect();

    let mut lit: Vec<i32> = (1..=num_vars as i32).collect();
    lit.push(0);

    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];

    let mut level_nodes: Vec<u32> = (0..num_vars as u32).collect();
    level_nodes.push(root);
    let levels = vec![(0, num_vars as u32), (num_vars as u32, 1)];

    // Set all weights to -inf (log(0)) except variable 5 which has log(0.7)
    let p_nonzero = 0.7_f64;
    let nonzero_idx = 5;
    let mut var_log_true = vec![f64::NEG_INFINITY; num_vars + 1];
    var_log_true[0] = 0.0; // Placeholder for index 0
    var_log_true[nonzero_idx] = p_nonzero.ln();

    let var_log_false = vec![0.0; num_vars + 1]; // Not used by positive literals

    // Expected: OR of mostly -inf + one log(0.7) = log(0.7)
    let mut expected_values = vec![f64::NEG_INFINITY; num_vars];
    expected_values[nonzero_idx - 1] = p_nonzero.ln(); // Lit index is var - 1
    let expected_root = p_nonzero.ln(); // logsumexp with all -inf except one = that one value
    expected_values.push(expected_root);

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true: vec![0.0; num_vars + 1],
        expected_grad_false: vec![0.0; num_vars + 1],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_sparse_nonzero",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify all -inf children are -inf
    for i in 0..num_vars {
        if i == nonzero_idx - 1 {
            continue; // Skip the non-zero one
        }
        let child_val = values[i];
        if !child_val.is_infinite() || !child_val.is_sign_negative() {
            return TestResult::error(
                "test_weight_sparse_nonzero",
                start.elapsed(),
                format!("Child {} should be -inf (log(0)), got {}", i, child_val),
            );
        }
    }

    // Verify the non-zero child has correct value
    let nonzero_child_val = values[nonzero_idx - 1];
    if (nonzero_child_val - p_nonzero.ln()).abs() > 1e-14 {
        return TestResult::error(
            "test_weight_sparse_nonzero",
            start.elapsed(),
            format!(
                "Non-zero child has wrong value: expected {}, got {}",
                p_nonzero.ln(),
                nonzero_child_val
            ),
        );
    }

    // Verify the OR root equals the only non-zero contribution
    let actual_root = values[root as usize];
    if (actual_root - expected_root).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_sparse_nonzero",
            start.elapsed(),
            format!(
                "Sparse OR root should equal single non-zero: expected {}, got {} (diff={})",
                expected_root,
                actual_root,
                (actual_root - expected_root).abs()
            ),
        );
    }

    // Test with two non-zero values to verify proper logsumexp
    let mut spec_two = spec.clone();
    let p_second = 0.3_f64;
    let second_idx = 8;
    spec_two.var_log_true[second_idx] = p_second.ln();

    let values_two = match run_tiny_xgcf_forward(ctx, &spec_two) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_weight_sparse_nonzero",
                start.elapsed(),
                format!("Failed to run forward pass with two non-zero: {}", e),
            );
        }
    };

    // Expected: logsumexp(log(0.7), log(0.3)) = log(0.7 + 0.3) = log(1.0) = 0
    let v1 = p_nonzero.ln();
    let v2 = p_second.ln();
    let max_v = v1.max(v2);
    let expected_two = max_v + ((v1 - max_v).exp() + (v2 - max_v).exp()).ln();
    let actual_two = values_two[spec_two.root as usize];

    if (actual_two - expected_two).abs() > 1e-12 {
        return TestResult::error(
            "test_weight_sparse_nonzero",
            start.elapsed(),
            format!(
                "Two non-zero sparse OR failed: expected {}, got {} (diff={})",
                expected_two,
                actual_two,
                (actual_two - expected_two).abs()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_weight_sparse_nonzero",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_weight_sparse_nonzero", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_weight_upload_integrity_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_upload_integrity(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_weight_buffer_reuse_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_buffer_reuse(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_weight_variable_mapping_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_variable_mapping(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_weight_extreme_values_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_extreme_values(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_weight_uniform_distribution_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_uniform_distribution(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_weight_sparse_nonzero_wrapper() {
        if let Some(ctx) = get_test_context() {
            let result = test_weight_sparse_nonzero(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }
}
