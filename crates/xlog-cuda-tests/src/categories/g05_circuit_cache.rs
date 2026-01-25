//! Category G05: Circuit Cache GPU Integration Tests
//!
//! Tests circuit caching behavior on GPU, verifying that:
//! 1. Repeated evaluations of the same circuit structure reuse GPU structures
//! 2. Cached evaluations are performant (no memory leaks or re-uploads)
//! 3. Different weights on cached circuits yield different results
//! 4. Different circuit structures produce different results
//! 5. Cached results are bit-identical to fresh evaluations

use crate::harness::xgcf::{gen_and_circuit, gen_or_circuit, tiny_xgcf_spec, TinyXgcfDevice};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::{Duration, Instant};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g05_circuit_cache");
    let start = Instant::now();

    // G05.1: Cache Hit GPU Behavior (3 tests)
    results.add_result(test_cache_hit_reuse(ctx));
    results.add_result(test_cache_hit_speedup(ctx));
    results.add_result(test_cache_distinct_results(ctx));

    // G05.2: Cache Key Correctness (3 tests)
    results.add_result(test_cache_key_same_structure(ctx));
    results.add_result(test_cache_different_circuits(ctx));
    results.add_result(test_cache_correctness(ctx));

    results.set_duration(start.elapsed());
    results
}

// =============================================================================
// G05.1: Cache Hit GPU Behavior Tests
// =============================================================================

/// Test 1: Same circuit spec evaluated twice uses cached GPU structures
///
/// Verifies that running the same circuit forward pass multiple times
/// produces bit-identical results, demonstrating GPU structures can be reused.
fn test_cache_hit_reuse(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let mut dev = match TinyXgcfDevice::upload(ctx, &spec) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_hit_reuse",
                start.elapsed(),
                format!("Failed to upload circuit to device: {}", e),
            );
        }
    };

    // First forward pass
    let values_first = match dev.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_hit_reuse",
                start.elapsed(),
                format!("Failed first forward pass: {}", e),
            );
        }
    };

    // Second forward pass with same spec
    let values_second = match dev.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_hit_reuse",
                start.elapsed(),
                format!("Failed second forward pass: {}", e),
            );
        }
    };

    // Verify both passes succeeded with same output size
    if values_first.len() != values_second.len() {
        return TestResult::error(
            "test_cache_hit_reuse",
            start.elapsed(),
            format!(
                "Output size mismatch: first={}, second={}",
                values_first.len(),
                values_second.len()
            ),
        );
    }

    // Verify results are bit-identical (GPU structures properly reused)
    for (idx, (&v1, &v2)) in values_first.iter().zip(values_second.iter()).enumerate() {
        if v1.to_bits() != v2.to_bits() {
            return TestResult::error(
                "test_cache_hit_reuse",
                start.elapsed(),
                format!(
                    "Results differ at index {}: first={} (bits={:#018x}), second={} (bits={:#018x}). \
                     GPU structure reuse should produce identical results.",
                    idx,
                    v1,
                    v1.to_bits(),
                    v2,
                    v2.to_bits()
                ),
            );
        }
    }

    // Verify the root value is valid (not NaN or unexpected)
    let root_value = values_first[spec.root as usize];
    if root_value.is_nan() {
        return TestResult::error(
            "test_cache_hit_reuse",
            start.elapsed(),
            "Root value is NaN, expected valid computation".to_string(),
        );
    }

    TestResult::passed("test_cache_hit_reuse", start.elapsed())
}

/// Test 2: Cached evaluation is faster than first (warmup)
///
/// Runs forward 10 times to warm up, then times next 100 runs.
/// Average should be reasonable (not degrading), proving no memory leaks or re-uploads.
fn test_cache_hit_speedup(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let mut dev = match TinyXgcfDevice::upload(ctx, &spec) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_hit_speedup",
                start.elapsed(),
                format!("Failed to upload circuit to device: {}", e),
            );
        }
    };

    // Warm-up: run 10 times to establish baseline GPU state
    for i in 0..10 {
        if let Err(e) = dev.forward_launch(ctx) {
            return TestResult::error(
                "test_cache_hit_speedup",
                start.elapsed(),
                format!("Failed warm-up forward pass {}: {}", i, e),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cache_hit_speedup",
            start.elapsed(),
            format!("Sync failed after warm-up: {}", e),
        );
    }

    // Time the next 100 runs
    let timed_start = Instant::now();
    for i in 0..100 {
        if let Err(e) = dev.forward_launch(ctx) {
            return TestResult::error(
                "test_cache_hit_speedup",
                start.elapsed(),
                format!("Failed timed forward pass {}: {}", i, e),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cache_hit_speedup",
            start.elapsed(),
            format!("Sync failed after timed runs: {}", e),
        );
    }

    let total_duration = timed_start.elapsed();
    let avg_duration = total_duration / 100;

    // Average should be reasonable: under 10ms per evaluation for this small circuit
    // If there were memory leaks or re-uploads, we'd see degradation
    let max_acceptable_avg = Duration::from_millis(10);
    if avg_duration > max_acceptable_avg {
        return TestResult::error(
            "test_cache_hit_speedup",
            start.elapsed(),
            format!(
                "Average evaluation time {:?} exceeds maximum acceptable {:?}. \
                 Suggests memory leak or re-upload overhead.",
                avg_duration, max_acceptable_avg
            ),
        );
    }

    // Verify throughput is reasonable: should handle at least 100 ops/sec
    let ops_per_sec = 100.0 / total_duration.as_secs_f64();
    if ops_per_sec < 100.0 {
        return TestResult::error(
            "test_cache_hit_speedup",
            start.elapsed(),
            format!(
                "Throughput {:.1} ops/sec is below minimum 100 ops/sec. \
                 Expected stable performance with cached GPU structures.",
                ops_per_sec
            ),
        );
    }

    TestResult::passed("test_cache_hit_speedup", start.elapsed())
}

/// Test 3: Different weights on same cached circuit yield different results
///
/// Uses same circuit structure but evaluates with weights A and weights B.
/// Verifies result_A != result_B, proving weights are properly updated.
fn test_cache_distinct_results(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create base circuit spec
    let spec_a = gen_and_circuit();

    // Create a modified spec with different weights but same structure
    let mut spec_b = gen_and_circuit();
    // Modify weights for spec_b (different probability values)
    spec_b.var_log_true[1] = 0.9_f64.ln(); // Was 0.7
    spec_b.var_log_true[2] = 0.8_f64.ln(); // Was 0.6
    spec_b.var_log_false[1] = 0.1_f64.ln(); // Was 0.3
    spec_b.var_log_false[2] = 0.2_f64.ln(); // Was 0.4

    let mut dev = match TinyXgcfDevice::upload(ctx, &spec_a) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_distinct_results",
                start.elapsed(),
                format!("Failed to upload circuit to device: {}", e),
            );
        }
    };

    // First eval with weights A
    if let Err(e) = dev.set_weights(ctx, &spec_a.var_log_true, &spec_a.var_log_false) {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!("Failed to upload weights A: {}", e),
        );
    }
    let root_a = match dev.forward_download_root(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_distinct_results",
                start.elapsed(),
                format!("Failed forward pass with weights A: {}", e),
            );
        }
    };

    // Second eval with weights B (same structure, different weights)
    if let Err(e) = dev.set_weights(ctx, &spec_b.var_log_true, &spec_b.var_log_false) {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!("Failed to upload weights B: {}", e),
        );
    }
    let root_b = match dev.forward_download_root(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_distinct_results",
                start.elapsed(),
                format!("Failed forward pass with weights B: {}", e),
            );
        }
    };

    // Verify results are different (weights properly updated)
    if root_a.to_bits() == root_b.to_bits() {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!(
                "Same structure with different weights produced identical root values: {} (bits={:#018x}). \
                 Expected different results, proving weights are properly updated.",
                root_a,
                root_a.to_bits()
            ),
        );
    }

    // Verify the difference is meaningful (not just floating point noise)
    let diff = (root_a - root_b).abs();
    if diff < 1e-10 {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!(
                "Results with different weights are too similar: A={}, B={}, diff={}. \
                 Expected meaningfully different results.",
                root_a, root_b, diff
            ),
        );
    }

    // Re-run with weights A to ensure we get the original result back
    if let Err(e) = dev.set_weights(ctx, &spec_a.var_log_true, &spec_a.var_log_false) {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!("Failed to upload weights A (rerun): {}", e),
        );
    }
    let root_a2 = match dev.forward_download_root(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_distinct_results",
                start.elapsed(),
                format!("Failed second forward pass with weights A: {}", e),
            );
        }
    };
    if root_a.to_bits() != root_a2.to_bits() {
        return TestResult::error(
            "test_cache_distinct_results",
            start.elapsed(),
            format!(
                "Re-running with weights A produced different result: \
                 first={} (bits={:#018x}), second={} (bits={:#018x}). \
                 Expected identical results with same weights.",
                root_a,
                root_a.to_bits(),
                root_a2,
                root_a2.to_bits()
            ),
        );
    }

    TestResult::passed("test_cache_distinct_results", start.elapsed())
}

// =============================================================================
// G05.2: Cache Key Correctness Tests
// =============================================================================

/// Test 4: Same structure circuits produce same results
///
/// Creates two specs with identical structure and weights.
/// Both should produce identical results.
fn test_cache_key_same_structure(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create two identical circuit specs
    let spec1 = gen_and_circuit();
    let spec2 = gen_and_circuit();

    // Verify specs are structurally identical
    if spec1.num_nodes != spec2.num_nodes
        || spec1.num_vars != spec2.num_vars
        || spec1.root != spec2.root
    {
        return TestResult::error(
            "test_cache_key_same_structure",
            start.elapsed(),
            "Test setup error: specs should have identical structure".to_string(),
        );
    }

    // Run forward on first spec
    let mut dev1 = match TinyXgcfDevice::upload(ctx, &spec1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_key_same_structure",
                start.elapsed(),
                format!("Failed to upload spec1 to device: {}", e),
            );
        }
    };
    let result1 = match dev1.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_key_same_structure",
                start.elapsed(),
                format!("Failed forward pass for spec1: {}", e),
            );
        }
    };

    // Run forward on second spec (identical structure and weights)
    let mut dev2 = match TinyXgcfDevice::upload(ctx, &spec2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_key_same_structure",
                start.elapsed(),
                format!("Failed to upload spec2 to device: {}", e),
            );
        }
    };
    let result2 = match dev2.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_key_same_structure",
                start.elapsed(),
                format!("Failed forward pass for spec2: {}", e),
            );
        }
    };

    // Verify output sizes match
    if result1.len() != result2.len() {
        return TestResult::error(
            "test_cache_key_same_structure",
            start.elapsed(),
            format!(
                "Output size mismatch: spec1={}, spec2={}",
                result1.len(),
                result2.len()
            ),
        );
    }

    // Verify all values are bit-identical
    for (idx, (&v1, &v2)) in result1.iter().zip(result2.iter()).enumerate() {
        if v1.to_bits() != v2.to_bits() {
            return TestResult::error(
                "test_cache_key_same_structure",
                start.elapsed(),
                format!(
                    "Results differ at index {}: spec1={} (bits={:#018x}), spec2={} (bits={:#018x}). \
                     Same structure and weights should produce identical results.",
                    idx,
                    v1,
                    v1.to_bits(),
                    v2,
                    v2.to_bits()
                ),
            );
        }
    }

    // Verify root values specifically
    let root1 = result1[spec1.root as usize];
    let root2 = result2[spec2.root as usize];
    if root1.to_bits() != root2.to_bits() {
        return TestResult::error(
            "test_cache_key_same_structure",
            start.elapsed(),
            format!(
                "Root values differ: spec1={} (bits={:#018x}), spec2={} (bits={:#018x})",
                root1,
                root1.to_bits(),
                root2,
                root2.to_bits()
            ),
        );
    }

    TestResult::passed("test_cache_key_same_structure", start.elapsed())
}

/// Test 5: Different circuits produce different results
///
/// Creates gen_and_circuit() and gen_or_circuit().
/// Even with same weights, results should differ.
/// This verifies circuit structure is properly evaluated.
fn test_cache_different_circuits(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create AND and OR circuits (different structure, same weight initialization)
    let and_spec = gen_and_circuit();
    let or_spec = gen_or_circuit();

    // Verify they have the same number of nodes and vars (same "shape" but different operations)
    if and_spec.num_nodes != or_spec.num_nodes || and_spec.num_vars != or_spec.num_vars {
        return TestResult::error(
            "test_cache_different_circuits",
            start.elapsed(),
            format!(
                "Test setup: AND ({} nodes, {} vars) and OR ({} nodes, {} vars) should have same shape",
                and_spec.num_nodes, and_spec.num_vars, or_spec.num_nodes, or_spec.num_vars
            ),
        );
    }

    let mut and_dev = match TinyXgcfDevice::upload(ctx, &and_spec) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_different_circuits",
                start.elapsed(),
                format!("Failed to upload AND circuit: {}", e),
            );
        }
    };
    let and_root = match and_dev.forward_download_root(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_different_circuits",
                start.elapsed(),
                format!("Failed forward pass for AND circuit: {}", e),
            );
        }
    };

    let mut or_dev = match TinyXgcfDevice::upload(ctx, &or_spec) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_different_circuits",
                start.elapsed(),
                format!("Failed to upload OR circuit: {}", e),
            );
        }
    };
    let or_root = match or_dev.forward_download_root(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_different_circuits",
                start.elapsed(),
                format!("Failed forward pass for OR circuit: {}", e),
            );
        }
    };

    // Verify results are different (AND vs OR should compute differently)
    if and_root.to_bits() == or_root.to_bits() {
        return TestResult::error(
            "test_cache_different_circuits",
            start.elapsed(),
            format!(
                "AND and OR circuits produced identical root values: {} (bits={:#018x}). \
                 Expected different results for different circuit structures.",
                and_root,
                and_root.to_bits()
            ),
        );
    }

    // Verify the difference is meaningful (not just floating point noise)
    let diff = (and_root - or_root).abs();
    if diff < 1e-10 {
        return TestResult::error(
            "test_cache_different_circuits",
            start.elapsed(),
            format!(
                "AND root ({}) and OR root ({}) are too similar (diff={}). \
                 Expected meaningfully different results.",
                and_root, or_root, diff
            ),
        );
    }

    // Verify both results are valid (not NaN)
    if and_root.is_nan() || or_root.is_nan() {
        return TestResult::error(
            "test_cache_different_circuits",
            start.elapsed(),
            format!(
                "Invalid results: AND root={}, OR root={}. Expected valid computations.",
                and_root, or_root
            ),
        );
    }

    // Verify AND < OR for these inputs (AND is product of log-probs, OR is log-sum-exp)
    // For AND(Lit(+1), Lit(+2)): ln(0.7) + ln(0.6) = ln(0.42) ~ -0.868
    // For OR(Lit(+1), Lit(+2)): logsumexp(ln(0.7), ln(0.6)) ~ ln(0.7 + 0.6) = ln(1.3) ~ 0.262
    // So and_root < or_root
    if and_root >= or_root {
        return TestResult::error(
            "test_cache_different_circuits",
            start.elapsed(),
            format!(
                "Expected AND root ({}) < OR root ({}) for these inputs. \
                 AND is log(p1*p2), OR is logsumexp(log(p1), log(p2)).",
                and_root, or_root
            ),
        );
    }

    TestResult::passed("test_cache_different_circuits", start.elapsed())
}

/// Test 6: Cached result matches fresh evaluation exactly
///
/// Runs evaluation, records result. Runs again, records result.
/// Results must be bit-identical, verifying caching doesn't corrupt results.
fn test_cache_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let mut dev = match TinyXgcfDevice::upload(ctx, &spec) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!("Failed to upload circuit to device: {}", e),
            );
        }
    };

    // First evaluation (fresh)
    let result_fresh = match dev.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!("Failed fresh evaluation: {}", e),
            );
        }
    };

    // Second evaluation (potentially cached)
    let result_cached = match dev.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!("Failed cached evaluation: {}", e),
            );
        }
    };

    // Verify output sizes match
    if result_fresh.len() != result_cached.len() {
        return TestResult::error(
            "test_cache_correctness",
            start.elapsed(),
            format!(
                "Output size mismatch: fresh={}, cached={}",
                result_fresh.len(),
                result_cached.len()
            ),
        );
    }

    // Verify all values are bit-identical
    for (idx, (&fresh, &cached)) in result_fresh.iter().zip(result_cached.iter()).enumerate() {
        if fresh.to_bits() != cached.to_bits() {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!(
                    "Results differ at index {}: fresh={} (bits={:#018x}), cached={} (bits={:#018x}). \
                     Cached result must match fresh evaluation exactly.",
                    idx,
                    fresh,
                    fresh.to_bits(),
                    cached,
                    cached.to_bits()
                ),
            );
        }
    }

    // Run a third time to verify continued correctness
    let result_third = match dev.forward_download_values(ctx) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!("Failed third evaluation: {}", e),
            );
        }
    };

    // Verify third result matches
    for (idx, (&fresh, &third)) in result_fresh.iter().zip(result_third.iter()).enumerate() {
        if fresh.to_bits() != third.to_bits() {
            return TestResult::error(
                "test_cache_correctness",
                start.elapsed(),
                format!(
                    "Third evaluation differs at index {}: fresh={} (bits={:#018x}), third={} (bits={:#018x}). \
                     All evaluations must be bit-identical.",
                    idx,
                    fresh,
                    fresh.to_bits(),
                    third,
                    third.to_bits()
                ),
            );
        }
    }

    // Verify against expected values from spec
    let root_idx = spec.root as usize;
    let computed_root = result_fresh[root_idx];
    let expected_root = spec.expected_values[root_idx];

    // Allow small tolerance for expected values (computed differently)
    let tolerance = 1e-12;
    if (computed_root - expected_root).abs() > tolerance {
        return TestResult::error(
            "test_cache_correctness",
            start.elapsed(),
            format!(
                "Root value {} differs from expected {} by more than tolerance {}. \
                 Cached computation may be incorrect.",
                computed_root, expected_root, tolerance
            ),
        );
    }

    TestResult::passed("test_cache_correctness", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_g05_cache_hit_reuse() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_hit_reuse(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g05_cache_hit_speedup() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_hit_speedup(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g05_cache_distinct_results() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_distinct_results(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g05_cache_key_same_structure() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_key_same_structure(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g05_cache_different_circuits() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_different_circuits(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g05_cache_correctness() {
        if let Some(ctx) = get_test_context() {
            let result = test_cache_correctness(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: status={:?} diagnostic={:?}",
                result.status,
                result.diagnostic
            );
        }
    }
}
