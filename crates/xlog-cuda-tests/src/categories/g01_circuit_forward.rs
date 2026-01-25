//! Category G01: Circuit Forward Kernel Tests
//!
//! Tests the forward evaluation kernels for XGCF circuits, covering all node types:
//! CONST0, CONST1, LIT, AND, OR, and DECISION nodes. These tests verify correct
//! log-space probability computation for neural-symbolic integration.

use crate::harness::xgcf::{
    gen_and_circuit, gen_decision_circuit, gen_deep_chain_circuit, gen_large_or_circuit,
    gen_or_circuit, gen_single_lit_circuit, run_tiny_xgcf_forward, tiny_xgcf_spec, TinyXgcfSpec,
};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g01_circuit_forward");
    let start = Instant::now();

    results.add_result(test_forward_const_nodes(ctx));
    results.add_result(test_forward_lit_nodes(ctx));
    results.add_result(test_forward_and_node(ctx));
    results.add_result(test_forward_or_node(ctx));
    results.add_result(test_forward_decision_basic(ctx));
    results.add_result(test_forward_decision_chain(ctx));
    results.add_result(test_forward_multi_level(ctx));
    results.add_result(test_forward_parallel_same_level(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test CONST0/CONST1 evaluation (-inf and 0.0)
///
/// Verifies that CONST0 nodes evaluate to -infinity (log(0) = -inf)
/// and CONST1 nodes evaluate to 0.0 (log(1) = 0).
fn test_forward_const_nodes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const CONST0: u8 = 0;
    const CONST1: u8 = 1;

    // Create a spec with 2 nodes: CONST0 (type 0) and CONST1 (type 1)
    let spec = TinyXgcfSpec {
        num_nodes: 2,
        num_vars: 0,
        root: 1, // CONST1 as root (valid node)
        node_type: vec![CONST0, CONST1],
        child_offsets: vec![0, 0, 0],
        child_indices: vec![],
        lit: vec![0, 0],
        decision_var: vec![0, 0],
        decision_child_false: vec![0, 0],
        decision_child_true: vec![0, 0],
        level_nodes: vec![0, 1],
        levels: vec![(0, 2)],
        var_log_true: vec![0.0],
        var_log_false: vec![0.0],
        expected_values: vec![f64::NEG_INFINITY, 0.0],
        expected_grad_true: vec![0.0],
        expected_grad_false: vec![0.0],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_const_nodes",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify CONST0 evaluates to -inf
    if !values[0].is_infinite() || !values[0].is_sign_negative() {
        return TestResult::error(
            "test_forward_const_nodes",
            start.elapsed(),
            format!("CONST0 should evaluate to -inf, got {}", values[0]),
        );
    }

    // Verify CONST1 evaluates to 0.0
    if (values[1] - 0.0).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_const_nodes",
            start.elapsed(),
            format!("CONST1 should evaluate to 0.0, got {}", values[1]),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_const_nodes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_const_nodes", start.elapsed())
}

/// Test 2: Test positive and negative literal evaluation
///
/// Verifies that positive literals return var_log_true and
/// negative literals return var_log_false for their respective variables.
fn test_forward_lit_nodes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test positive literal using gen_single_lit_circuit(1)
    let pos_spec = gen_single_lit_circuit(1);
    let pos_values = match run_tiny_xgcf_forward(ctx, &pos_spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_lit_nodes",
                start.elapsed(),
                format!("Failed to run forward pass for positive literal: {}", e),
            );
        }
    };

    // Verify positive literal returns var_log_true
    let expected_pos = pos_spec.var_log_true[1];
    if (pos_values[0] - expected_pos).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_lit_nodes",
            start.elapsed(),
            format!(
                "Positive literal should return var_log_true[1]={}, got {}",
                expected_pos, pos_values[0]
            ),
        );
    }

    // Create inline spec for negative literal (lit = [-1])
    const LIT: u8 = 2;
    let neg_spec = TinyXgcfSpec {
        num_nodes: 1,
        num_vars: 1,
        root: 0,
        node_type: vec![LIT],
        child_offsets: vec![0, 0],
        child_indices: vec![],
        lit: vec![-1], // Negative literal
        decision_var: vec![0],
        decision_child_false: vec![0],
        decision_child_true: vec![0],
        level_nodes: vec![0],
        levels: vec![(0, 1)],
        var_log_true: vec![0.0, 0.7_f64.ln()],
        var_log_false: vec![0.0, 0.3_f64.ln()],
        expected_values: vec![0.3_f64.ln()],
        expected_grad_true: vec![0.0, 0.0],
        expected_grad_false: vec![0.0, 1.0],
    };

    let neg_values = match run_tiny_xgcf_forward(ctx, &neg_spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_lit_nodes",
                start.elapsed(),
                format!("Failed to run forward pass for negative literal: {}", e),
            );
        }
    };

    // Verify negative literal returns var_log_false
    let expected_neg = neg_spec.var_log_false[1];
    if (neg_values[0] - expected_neg).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_lit_nodes",
            start.elapsed(),
            format!(
                "Negative literal should return var_log_false[1]={}, got {}",
                expected_neg, neg_values[0]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_lit_nodes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_lit_nodes", start.elapsed())
}

/// Test 3: Test AND node evaluation (sum in log space)
///
/// Verifies that AND nodes correctly compute the sum of child values
/// in log space, corresponding to multiplication of probabilities.
fn test_forward_and_node(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_and_circuit();
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_and_node",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify all values match expected with tolerance
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_and_node",
                start.elapsed(),
                format!(
                    "Node {} value mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify AND semantics: root = sum of children in log space
    let child0_val = values[0];
    let child1_val = values[1];
    let and_val = values[2];
    let expected_and = child0_val + child1_val;
    if (and_val - expected_and).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_and_node",
            start.elapsed(),
            format!(
                "AND node should compute sum in log space: {} + {} = {}, got {}",
                child0_val, child1_val, expected_and, and_val
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_and_node",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_and_node", start.elapsed())
}

/// Test 4: Test OR node evaluation (logsumexp)
///
/// Verifies that OR nodes correctly compute logsumexp of child values,
/// corresponding to addition of probabilities in log space.
fn test_forward_or_node(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_or_circuit();
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_or_node",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify all values match expected with tolerance
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_or_node",
                start.elapsed(),
                format!(
                    "Node {} value mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify OR semantics: root = logsumexp(children)
    let child0_val = values[0];
    let child1_val = values[1];
    let or_val = values[2];
    let max_val = child0_val.max(child1_val);
    let expected_or = max_val + ((child0_val - max_val).exp() + (child1_val - max_val).exp()).ln();
    if (or_val - expected_or).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_or_node",
            start.elapsed(),
            format!(
                "OR node should compute logsumexp: logsumexp({}, {}) = {}, got {}",
                child0_val, child1_val, expected_or, or_val
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_or_node",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_or_node", start.elapsed())
}

/// Test 5: Test basic decision node evaluation
///
/// Verifies that decision nodes correctly compute the weighted combination
/// of their true and false children based on the decision variable.
fn test_forward_decision_basic(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_decision_basic",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify all values match expected with tolerance
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_decision_basic",
                start.elapsed(),
                format!(
                    "Node {} value mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify decision node semantics: logsumexp(log_false + val_false, log_true + val_true)
    let val_false_child = values[0]; // CONST1 = 0.0
    let val_true_child = values[1]; // LIT(+1)
    let decision_val = values[2];
    let log_false = spec.var_log_false[2];
    let log_true = spec.var_log_true[2];
    let branch_false = log_false + val_false_child;
    let branch_true = log_true + val_true_child;
    let max_branch = branch_false.max(branch_true);
    let expected_decision =
        max_branch + ((branch_false - max_branch).exp() + (branch_true - max_branch).exp()).ln();

    if (decision_val - expected_decision).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_decision_basic",
            start.elapsed(),
            format!(
                "Decision node value mismatch: expected {}, got {}",
                expected_decision, decision_val
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_decision_basic",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_decision_basic", start.elapsed())
}

/// Test 6: Test chain of decision nodes (use tiny_xgcf_spec)
///
/// Verifies correct evaluation of a complex circuit with multiple
/// node types including decision nodes in a chain configuration.
fn test_forward_decision_chain(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_decision_chain",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify all values match expected with tolerance
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        // Handle -inf comparison specially
        if expected.is_infinite() && expected.is_sign_negative() {
            if !actual.is_infinite() || !actual.is_sign_negative() {
                return TestResult::error(
                    "test_forward_decision_chain",
                    start.elapsed(),
                    format!("Node {} should be -inf, got {}", i, actual),
                );
            }
            continue;
        }

        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_decision_chain",
                start.elapsed(),
                format!(
                    "Node {} value mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify correct number of nodes
    if values.len() != spec.num_nodes {
        return TestResult::error(
            "test_forward_decision_chain",
            start.elapsed(),
            format!("Expected {} nodes, got {}", spec.num_nodes, values.len()),
        );
    }

    // Verify root value specifically (node 6 is OR of decision and CONST0)
    let root_val = values[spec.root as usize];
    let expected_root = spec.expected_values[spec.root as usize];
    if (root_val - expected_root).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_decision_chain",
            start.elapsed(),
            format!(
                "Root node value mismatch: expected {}, got {}",
                expected_root, root_val
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_decision_chain",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_decision_chain", start.elapsed())
}

/// Test 7: Test multi-level circuit (use gen_deep_chain_circuit(10))
///
/// Verifies correct evaluation of a deep chain of AND nodes, where each
/// AND has a single child, propagating the literal value through all levels.
fn test_forward_multi_level(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_deep_chain_circuit(10);
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_multi_level",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify correct number of nodes (1 LIT + 10 ANDs = 11 nodes)
    if values.len() != spec.num_nodes {
        return TestResult::error(
            "test_forward_multi_level",
            start.elapsed(),
            format!("Expected {} nodes, got {}", spec.num_nodes, values.len()),
        );
    }

    // All nodes should have the same value (lit value propagates through single-child ANDs)
    let lit_val = spec.var_log_true[1]; // The value of Lit(+1)
    for (i, &val) in values.iter().enumerate() {
        let diff = (val - lit_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_multi_level",
                start.elapsed(),
                format!(
                    "Node {} should have value {} (same as lit), got {} (diff={})",
                    i, lit_val, val, diff
                ),
            );
        }
    }

    // Verify against expected values from spec
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_multi_level",
                start.elapsed(),
                format!(
                    "Node {} value mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_multi_level",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_multi_level", start.elapsed())
}

/// Test 8: Test parallel execution (use gen_large_or_circuit(1000))
///
/// Verifies correct parallel evaluation of 1000 literals at the same level,
/// all feeding into a single OR node. Tests GPU parallelism correctness.
fn test_forward_parallel_same_level(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let num_lits = 1000;
    let spec = gen_large_or_circuit(num_lits);
    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_forward_parallel_same_level",
                start.elapsed(),
                format!("Failed to run forward pass: {}", e),
            );
        }
    };

    // Verify correct number of nodes (1000 LITs + 1 OR = 1001 nodes)
    if values.len() != spec.num_nodes {
        return TestResult::error(
            "test_forward_parallel_same_level",
            start.elapsed(),
            format!("Expected {} nodes, got {}", spec.num_nodes, values.len()),
        );
    }

    // Verify all 1000 literals have the same value (0.5.ln())
    let expected_lit_val = 0.5_f64.ln();
    for i in 0..num_lits {
        let diff = (values[i] - expected_lit_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_parallel_same_level",
                start.elapsed(),
                format!(
                    "Literal {} should have value {}, got {} (diff={})",
                    i, expected_lit_val, values[i], diff
                ),
            );
        }
    }

    // Verify OR root value matches expected: lit_val + ln(num_lits)
    let or_idx = num_lits;
    let expected_or_val = expected_lit_val + (num_lits as f64).ln();
    let or_val = values[or_idx];
    let diff = (or_val - expected_or_val).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_forward_parallel_same_level",
            start.elapsed(),
            format!(
                "OR root should have value {} (lit_val + ln({})), got {} (diff={})",
                expected_or_val, num_lits, or_val, diff
            ),
        );
    }

    // Cross-check with spec expected values
    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_forward_parallel_same_level",
                start.elapsed(),
                format!(
                    "Node {} value mismatch vs spec: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_forward_parallel_same_level",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_forward_parallel_same_level", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_g01_forward_const_nodes() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_const_nodes(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_lit_nodes() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_lit_nodes(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_and_node() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_and_node(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_or_node() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_or_node(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_decision_basic() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_decision_basic(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_decision_chain() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_decision_chain(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_multi_level() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_multi_level(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g01_forward_parallel_same_level() {
        if let Some(ctx) = get_test_context() {
            let result = test_forward_parallel_same_level(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }
}
