//! Category G06: PTX Kernel Robustness Tests
//!
//! Tests the robustness of PTX kernels under various edge conditions:
//! - Circuit size edge cases (single node, single level, deep circuits, large circuits)
//! - Variable count limits (max variables, sparse variables, many literals)
//! - Numerical stability (log underflow, logsumexp stability, determinism)

use crate::harness::xgcf::{
    gen_deep_chain_circuit, gen_large_or_circuit, run_tiny_xgcf_forward, TinyXgcfSpec,
};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g06_ptx_robustness");
    let start = Instant::now();

    // G06.1: Circuit Size Edge Cases (4 tests)
    results.add_result(test_kernel_single_node(ctx));
    results.add_result(test_kernel_single_level(ctx));
    results.add_result(test_kernel_deep_circuit(ctx));
    results.add_result(test_kernel_large_circuit(ctx));

    // G06.2: Variable Count Limits (3 tests)
    results.add_result(test_kernel_max_variables(ctx));
    results.add_result(test_kernel_sparse_variables(ctx));
    results.add_result(test_kernel_many_literals(ctx));

    // G06.3: Numerical Stability (3 tests)
    results.add_result(test_kernel_log_underflow(ctx));
    results.add_result(test_kernel_logsumexp_stability(ctx));
    results.add_result(test_kernel_determinism(ctx));

    results.set_duration(start.elapsed());
    results
}

// =============================================================================
// G06.1: Circuit Size Edge Cases (4 tests)
// =============================================================================

/// Test 1: Single-node circuit (just CONST1)
///
/// Creates a circuit with only 1 node (CONST1 which evaluates to 0.0).
/// Verifies the forward pass handles minimal circuit size correctly.
fn test_kernel_single_node(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const CONST1: u8 = 1;

    // Minimal circuit: just CONST1 (evaluates to 0.0 = log(1))
    let spec = TinyXgcfSpec {
        num_nodes: 1,
        num_vars: 0,
        root: 0,
        node_type: vec![CONST1],
        child_offsets: vec![0, 0],
        child_indices: vec![],
        lit: vec![0],
        decision_var: vec![0],
        decision_child_false: vec![0],
        decision_child_true: vec![0],
        level_nodes: vec![0],
        levels: vec![(0, 1)],
        var_log_true: vec![0.0],
        var_log_false: vec![0.0],
        expected_values: vec![0.0],
        expected_grad_true: vec![0.0],
        expected_grad_false: vec![0.0],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_single_node",
                start.elapsed(),
                format!("Failed to run forward pass on single-node circuit: {}", e),
            );
        }
    };

    // Verify we got exactly 1 value
    if values.len() != 1 {
        return TestResult::error(
            "test_kernel_single_node",
            start.elapsed(),
            format!("Expected 1 value, got {}", values.len()),
        );
    }

    // CONST1 should evaluate to 0.0 (log(1) = 0)
    if (values[0] - 0.0).abs() > 1e-10 {
        return TestResult::error(
            "test_kernel_single_node",
            start.elapsed(),
            format!("CONST1 should evaluate to 0.0, got {}", values[0]),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_single_node",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_single_node", start.elapsed())
}

/// Test 2: Single-level circuit (all nodes at same level)
///
/// Creates a large OR circuit with 100 literals at level 0, OR at level 1.
/// Verifies parallel execution at same level handles high parallelism correctly.
fn test_kernel_single_level(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // 100 literals at level 0, all feeding into OR at level 1
    let spec = gen_large_or_circuit(100);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_single_level",
                start.elapsed(),
                format!("Failed to run forward pass on single-level circuit: {}", e),
            );
        }
    };

    // Verify we got 101 values (100 literals + 1 OR)
    if values.len() != 101 {
        return TestResult::error(
            "test_kernel_single_level",
            start.elapsed(),
            format!("Expected 101 values (100 lits + 1 OR), got {}", values.len()),
        );
    }

    // All 100 literals should have the same value: ln(0.5)
    let expected_lit_val = 0.5_f64.ln();
    for i in 0..100 {
        let diff = (values[i] - expected_lit_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_single_level",
                start.elapsed(),
                format!(
                    "Literal {} should have value {}, got {} (diff={})",
                    i, expected_lit_val, values[i], diff
                ),
            );
        }
    }

    // OR root should be lit_val + ln(100): logsumexp of 100 identical values
    let expected_or_val = expected_lit_val + (100.0_f64).ln();
    let or_val = values[100];
    let diff = (or_val - expected_or_val).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_kernel_single_level",
            start.elapsed(),
            format!(
                "OR root should have value {} (lit_val + ln(100)), got {} (diff={})",
                expected_or_val, or_val, diff
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_single_level",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_single_level", start.elapsed())
}

/// Test 3: Deep circuit (100 levels)
///
/// Creates a chain of 100 AND nodes, verifying sequential level-by-level execution.
/// Tests that deep recursion/iteration doesn't cause stack overflow or other issues.
fn test_kernel_deep_circuit(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // 100 levels deep: Lit -> AND -> AND -> ... -> AND (100 ANDs)
    let spec = gen_deep_chain_circuit(100);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_deep_circuit",
                start.elapsed(),
                format!("Failed to run forward pass on 100-level deep circuit: {}", e),
            );
        }
    };

    // Verify we got 101 values (1 Lit + 100 ANDs)
    if values.len() != 101 {
        return TestResult::error(
            "test_kernel_deep_circuit",
            start.elapsed(),
            format!("Expected 101 values (1 Lit + 100 ANDs), got {}", values.len()),
        );
    }

    // All nodes should propagate the literal value: ln(0.7)
    let expected_val = 0.7_f64.ln();
    for (i, &val) in values.iter().enumerate() {
        let diff = (val - expected_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_deep_circuit",
                start.elapsed(),
                format!(
                    "Node {} should have value {} (propagated literal), got {} (diff={})",
                    i, expected_val, val, diff
                ),
            );
        }
    }

    // Verify root specifically
    let root_val = values[spec.root as usize];
    if (root_val - expected_val).abs() > 1e-10 {
        return TestResult::error(
            "test_kernel_deep_circuit",
            start.elapsed(),
            format!(
                "Root (level 100) should have value {}, got {}",
                expected_val, root_val
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_deep_circuit",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_deep_circuit", start.elapsed())
}

/// Test 4: Large circuit (10000 nodes)
///
/// Creates a circuit with 10000 literals + 1 OR = 10001 nodes.
/// Stress test for memory allocation and parallel throughput.
fn test_kernel_large_circuit(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // 10000 literals + 1 OR = 10001 nodes
    let spec = gen_large_or_circuit(10000);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_large_circuit",
                start.elapsed(),
                format!("Failed to run forward pass on 10001-node circuit: {}", e),
            );
        }
    };

    // Verify we got 10001 values
    if values.len() != 10001 {
        return TestResult::error(
            "test_kernel_large_circuit",
            start.elapsed(),
            format!("Expected 10001 values, got {}", values.len()),
        );
    }

    // Spot check: all literals should have ln(0.5)
    let expected_lit_val = 0.5_f64.ln();
    for &idx in &[0, 100, 1000, 5000, 9999] {
        let diff = (values[idx] - expected_lit_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_large_circuit",
                start.elapsed(),
                format!(
                    "Literal {} should have value {}, got {} (diff={})",
                    idx, expected_lit_val, values[idx], diff
                ),
            );
        }
    }

    // OR root should be lit_val + ln(10000)
    let expected_or_val = expected_lit_val + (10000.0_f64).ln();
    let or_val = values[10000];
    let diff = (or_val - expected_or_val).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_kernel_large_circuit",
            start.elapsed(),
            format!(
                "OR root should have value {} (lit_val + ln(10000)), got {} (diff={})",
                expected_or_val, or_val, diff
            ),
        );
    }

    // Verify no NaN or unexpected infinity in output
    for (i, &val) in values.iter().enumerate() {
        if val.is_nan() {
            return TestResult::error(
                "test_kernel_large_circuit",
                start.elapsed(),
                format!("Node {} produced NaN, expected valid value", i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_large_circuit",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_large_circuit", start.elapsed())
}

// =============================================================================
// G06.2: Variable Count Limits (3 tests)
// =============================================================================

/// Test 5: Circuit with 1000 variables
///
/// Creates a circuit referencing 1000 distinct variables.
/// Verifies weight buffers handle large variable counts correctly.
fn test_kernel_max_variables(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const OR: u8 = 4;

    let num_vars = 1000;
    let num_nodes = num_vars + 1; // 1000 literals + 1 OR

    // Create a circuit with 1000 distinct variables, each in a positive literal
    let mut node_type = vec![LIT; num_vars];
    node_type.push(OR);

    let mut child_offsets: Vec<u32> = (0..=num_vars).map(|_| 0).collect();
    child_offsets.push(num_vars as u32);

    let child_indices: Vec<u32> = (0..num_vars as u32).collect();

    // Each literal references a unique variable: Lit(1), Lit(2), ..., Lit(1000)
    let mut lit: Vec<i32> = (1..=num_vars as i32).collect();
    lit.push(0);

    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];

    let mut level_nodes: Vec<u32> = (0..num_vars as u32).collect();
    level_nodes.push(num_vars as u32);
    let levels = vec![(0, num_vars as u32), (num_vars as u32, 1)];

    // Each variable has different probability: p_i = 0.5 + 0.0004*(i-1)
    // This ensures each variable's weight is unique
    let mut var_log_true = vec![0.0; num_vars + 1];
    let mut var_log_false = vec![0.0; num_vars + 1];
    for i in 1..=num_vars {
        let p = 0.5 + 0.0004 * (i - 1) as f64; // 0.5 to 0.8996
        var_log_true[i] = p.ln();
        var_log_false[i] = (1.0 - p).ln();
    }

    // Expected values: each literal gets its var_log_true
    let mut expected_values: Vec<f64> = (1..=num_vars).map(|i| var_log_true[i]).collect();

    // OR node: logsumexp of all literals
    let max_val = expected_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let or_val = max_val + expected_values.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
    expected_values.push(or_val);

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars,
        root: num_vars as u32,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true: var_log_true.clone(),
        var_log_false,
        expected_values: expected_values.clone(),
        expected_grad_true: vec![0.0; num_vars + 1],
        expected_grad_false: vec![0.0; num_vars + 1],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_max_variables",
                start.elapsed(),
                format!("Failed to run forward pass with 1000 variables: {}", e),
            );
        }
    };

    // Verify we got correct number of values
    if values.len() != num_nodes {
        return TestResult::error(
            "test_kernel_max_variables",
            start.elapsed(),
            format!("Expected {} values, got {}", num_nodes, values.len()),
        );
    }

    // Spot check: verify some literals have correct values
    for &i in &[0, 100, 500, 999] {
        let expected = var_log_true[i + 1];
        let diff = (values[i] - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_max_variables",
                start.elapsed(),
                format!(
                    "Literal {} (var {}) should have value {}, got {} (diff={})",
                    i,
                    i + 1,
                    expected,
                    values[i],
                    diff
                ),
            );
        }
    }

    // Verify OR root
    let computed_or = values[num_vars];
    let diff = (computed_or - or_val).abs();
    if diff > 1e-8 {
        return TestResult::error(
            "test_kernel_max_variables",
            start.elapsed(),
            format!(
                "OR root should have value {}, got {} (diff={})",
                or_val, computed_or, diff
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_max_variables",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_max_variables", start.elapsed())
}

/// Test 6: Circuit with sparse (non-contiguous) variables
///
/// Creates a circuit using only variables [1, 100, 500].
/// Verifies non-contiguous variable access works correctly.
fn test_kernel_sparse_variables(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const OR: u8 = 4;

    // Sparse variables: 1, 100, 500 (non-contiguous)
    let max_var = 500usize;
    let num_nodes = 4; // 3 literals + 1 OR

    let node_type = vec![LIT, LIT, LIT, OR];
    let child_offsets = vec![0, 0, 0, 0, 3];
    let child_indices = vec![0, 1, 2];
    let lit = vec![1, 100, 500, 0]; // Lit(1), Lit(100), Lit(500), OR
    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];
    let level_nodes = vec![0, 1, 2, 3];
    let levels = vec![(0, 3), (3, 1)];

    // Initialize var weights for all variables up to max
    let mut var_log_true = vec![0.0; max_var + 1];
    let mut var_log_false = vec![0.0; max_var + 1];

    // Set specific weights for our sparse variables
    let p1 = 0.7_f64;
    let p100 = 0.3_f64;
    let p500 = 0.9_f64;
    var_log_true[1] = p1.ln();
    var_log_false[1] = (1.0 - p1).ln();
    var_log_true[100] = p100.ln();
    var_log_false[100] = (1.0 - p100).ln();
    var_log_true[500] = p500.ln();
    var_log_false[500] = (1.0 - p500).ln();

    // Expected values
    let v0 = var_log_true[1];
    let v1 = var_log_true[100];
    let v2 = var_log_true[500];

    // logsumexp for OR
    let max_val = v0.max(v1).max(v2);
    let or_val = max_val + ((v0 - max_val).exp() + (v1 - max_val).exp() + (v2 - max_val).exp()).ln();

    let expected_values = vec![v0, v1, v2, or_val];

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars: max_var,
        root: 3,
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
        expected_values: expected_values.clone(),
        expected_grad_true: vec![0.0; max_var + 1],
        expected_grad_false: vec![0.0; max_var + 1],
    };

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_sparse_variables",
                start.elapsed(),
                format!("Failed to run forward pass with sparse variables: {}", e),
            );
        }
    };

    // Verify we got correct number of values
    if values.len() != num_nodes {
        return TestResult::error(
            "test_kernel_sparse_variables",
            start.elapsed(),
            format!("Expected {} values, got {}", num_nodes, values.len()),
        );
    }

    // Verify each literal got the correct weight
    for (i, (expected, name)) in [
        (v0, "Lit(1)"),
        (v1, "Lit(100)"),
        (v2, "Lit(500)"),
    ]
    .iter()
    .enumerate()
    {
        let diff = (values[i] - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_sparse_variables",
                start.elapsed(),
                format!(
                    "{} should have value {}, got {} (diff={})",
                    name, expected, values[i], diff
                ),
            );
        }
    }

    // Verify OR root
    let computed_or = values[3];
    let diff = (computed_or - or_val).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_kernel_sparse_variables",
            start.elapsed(),
            format!(
                "OR root should have value {}, got {} (diff={})",
                or_val, computed_or, diff
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_sparse_variables",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_sparse_variables", start.elapsed())
}

/// Test 7: Circuit with 1000 literal nodes
///
/// Creates a circuit with 1000 literal node evaluations.
/// Verifies parallel literal evaluation at scale.
fn test_kernel_many_literals(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // 1000 literals + 1 OR
    let spec = gen_large_or_circuit(1000);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_kernel_many_literals",
                start.elapsed(),
                format!("Failed to run forward pass with 1000 literals: {}", e),
            );
        }
    };

    // Verify we got 1001 values
    if values.len() != 1001 {
        return TestResult::error(
            "test_kernel_many_literals",
            start.elapsed(),
            format!("Expected 1001 values, got {}", values.len()),
        );
    }

    // All 1000 literals should have ln(0.5)
    let expected_lit_val = 0.5_f64.ln();
    for i in 0..1000 {
        let diff = (values[i] - expected_lit_val).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_many_literals",
                start.elapsed(),
                format!(
                    "Literal {} should have value {}, got {} (diff={})",
                    i, expected_lit_val, values[i], diff
                ),
            );
        }
    }

    // OR root should be lit_val + ln(1000)
    let expected_or_val = expected_lit_val + (1000.0_f64).ln();
    let or_val = values[1000];
    let diff = (or_val - expected_or_val).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_kernel_many_literals",
            start.elapsed(),
            format!(
                "OR root should have value {}, got {} (diff={})",
                expected_or_val, or_val, diff
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_many_literals",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_many_literals", start.elapsed())
}

// =============================================================================
// G06.3: Numerical Stability (3 tests)
// =============================================================================

/// Test 8: Log underflow handling
///
/// Sets weights to log(1e-300) ~ -690.
/// Verifies the kernel handles extreme negative log values without producing NaN.
fn test_kernel_log_underflow(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const OR: u8 = 4;

    // Create circuit with very small probability (log underflow territory)
    let tiny_prob = 1e-300_f64;
    let log_tiny = tiny_prob.ln(); // ~ -690

    let num_nodes = 3;
    let num_vars = 2;

    let node_type = vec![LIT, LIT, OR];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    // Both variables have extremely small probabilities
    let var_log_true = vec![0.0, log_tiny, log_tiny];
    let var_log_false = vec![0.0, 0.0, 0.0]; // Not used for positive literals

    // Expected: both lits have log_tiny, OR is logsumexp ~ log_tiny + ln(2)
    let expected_or = log_tiny + 2.0_f64.ln();
    let expected_values = vec![log_tiny, log_tiny, expected_or];

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars,
        root: 2,
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
                "test_kernel_log_underflow",
                start.elapsed(),
                format!("Failed to run forward pass with underflow weights: {}", e),
            );
        }
    };

    // Verify no NaN values
    for (i, &val) in values.iter().enumerate() {
        if val.is_nan() {
            return TestResult::error(
                "test_kernel_log_underflow",
                start.elapsed(),
                format!(
                    "Node {} produced NaN with log(1e-300) weights, expected valid log-space value",
                    i
                ),
            );
        }
    }

    // Verify literals have the extreme value (should be preserved exactly)
    for i in 0..2 {
        let diff = (values[i] - log_tiny).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_kernel_log_underflow",
                start.elapsed(),
                format!(
                    "Literal {} should have value {} (log(1e-300)), got {} (diff={})",
                    i, log_tiny, values[i], diff
                ),
            );
        }
    }

    // Verify OR handles extreme values correctly
    let computed_or = values[2];
    let diff = (computed_or - expected_or).abs();
    // Allow slightly larger tolerance for extreme values
    if diff > 1e-8 {
        return TestResult::error(
            "test_kernel_log_underflow",
            start.elapsed(),
            format!(
                "OR with extreme log values should produce {}, got {} (diff={})",
                expected_or, computed_or, diff
            ),
        );
    }

    // Verify result is not incorrectly zero or positive infinity
    if computed_or >= 0.0 {
        return TestResult::error(
            "test_kernel_log_underflow",
            start.elapsed(),
            format!(
                "OR of log(1e-300) values should be large negative, got {}",
                computed_or
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_log_underflow",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_log_underflow", start.elapsed())
}

/// Test 9: Logsumexp numerical stability
///
/// OR of values with huge difference: log(1e-300) vs log(0.5).
/// Logsumexp should return the larger value (numerical stability test).
fn test_kernel_logsumexp_stability(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const OR: u8 = 4;

    // One very small, one moderate probability
    let tiny_prob = 1e-300_f64;
    let log_tiny = tiny_prob.ln(); // ~ -690
    let normal_prob = 0.5_f64;
    let log_normal = normal_prob.ln(); // ~ -0.693

    let num_nodes = 3;
    let num_vars = 2;

    let node_type = vec![LIT, LIT, OR];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let var_log_true = vec![0.0, log_tiny, log_normal];
    let var_log_false = vec![0.0, 0.0, 0.0];

    // logsumexp(log_tiny, log_normal) should be very close to log_normal
    // because exp(log_tiny - log_normal) ~ 0
    // Mathematically: logsumexp(-690, -0.693) ~ -0.693
    let max_val = log_tiny.max(log_normal);
    let expected_or = max_val + ((log_tiny - max_val).exp() + (log_normal - max_val).exp()).ln();

    let expected_values = vec![log_tiny, log_normal, expected_or];

    let spec = TinyXgcfSpec {
        num_nodes,
        num_vars,
        root: 2,
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
                "test_kernel_logsumexp_stability",
                start.elapsed(),
                format!("Failed to run forward pass with extreme value differences: {}", e),
            );
        }
    };

    // Verify no NaN values
    for (i, &val) in values.iter().enumerate() {
        if val.is_nan() {
            return TestResult::error(
                "test_kernel_logsumexp_stability",
                start.elapsed(),
                format!("Node {} produced NaN, expected valid value", i),
            );
        }
    }

    // Verify literal values
    if (values[0] - log_tiny).abs() > 1e-10 {
        return TestResult::error(
            "test_kernel_logsumexp_stability",
            start.elapsed(),
            format!(
                "Literal 0 should have value {}, got {}",
                log_tiny, values[0]
            ),
        );
    }
    if (values[1] - log_normal).abs() > 1e-10 {
        return TestResult::error(
            "test_kernel_logsumexp_stability",
            start.elapsed(),
            format!(
                "Literal 1 should have value {}, got {}",
                log_normal, values[1]
            ),
        );
    }

    // The key test: OR should be dominated by the larger value (log_normal)
    let computed_or = values[2];

    // OR result should be very close to log_normal (the dominant term)
    // The difference should be tiny: ln(1 + 1e-300) ~ 0
    let diff_from_normal = (computed_or - log_normal).abs();
    if diff_from_normal > 1e-10 {
        return TestResult::error(
            "test_kernel_logsumexp_stability",
            start.elapsed(),
            format!(
                "logsumexp({}, {}) should be very close to {}, got {} (diff={}). \
                 Numerical stability issue: small value should not affect result.",
                log_tiny, log_normal, log_normal, computed_or, diff_from_normal
            ),
        );
    }

    // Verify against expected value
    let diff_from_expected = (computed_or - expected_or).abs();
    if diff_from_expected > 1e-10 {
        return TestResult::error(
            "test_kernel_logsumexp_stability",
            start.elapsed(),
            format!(
                "OR should have value {}, got {} (diff={})",
                expected_or, computed_or, diff_from_expected
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_logsumexp_stability",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_kernel_logsumexp_stability", start.elapsed())
}

/// Test 10: Determinism across multiple runs
///
/// Runs the same circuit 10 times and verifies all results are bit-identical.
/// Tests that GPU floating-point operations are deterministic.
fn test_kernel_determinism(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Use a moderately complex circuit to stress determinism
    let spec = gen_large_or_circuit(100);

    // Run forward pass 10 times
    let mut all_results: Vec<Vec<f64>> = Vec::with_capacity(10);
    for i in 0..10 {
        let values = match run_tiny_xgcf_forward(ctx, &spec) {
            Ok(v) => v,
            Err(e) => {
                return TestResult::error(
                    "test_kernel_determinism",
                    start.elapsed(),
                    format!("Failed forward pass iteration {}: {}", i, e),
                );
            }
        };
        all_results.push(values);
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_determinism",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    // Verify all results are bit-identical to the first
    let reference = &all_results[0];
    for (run_idx, result) in all_results.iter().enumerate().skip(1) {
        if result.len() != reference.len() {
            return TestResult::error(
                "test_kernel_determinism",
                start.elapsed(),
                format!(
                    "Run {} produced {} values, reference has {}",
                    run_idx,
                    result.len(),
                    reference.len()
                ),
            );
        }

        for (node_idx, (&ref_val, &run_val)) in reference.iter().zip(result.iter()).enumerate() {
            if ref_val.to_bits() != run_val.to_bits() {
                return TestResult::error(
                    "test_kernel_determinism",
                    start.elapsed(),
                    format!(
                        "Non-determinism detected: run {} node {} has value {} (bits={:#018x}), \
                         reference has {} (bits={:#018x}). GPU execution must be deterministic.",
                        run_idx,
                        node_idx,
                        run_val,
                        run_val.to_bits(),
                        ref_val,
                        ref_val.to_bits()
                    ),
                );
            }
        }
    }

    // Verify the reference result is sensible (not all zeros or NaN)
    let root_val = reference[spec.root as usize];
    if root_val.is_nan() {
        return TestResult::error(
            "test_kernel_determinism",
            start.elapsed(),
            "Reference root value is NaN".to_string(),
        );
    }

    // Verify expected value
    let expected_root = 0.5_f64.ln() + (100.0_f64).ln();
    let diff = (root_val - expected_root).abs();
    if diff > 1e-10 {
        return TestResult::error(
            "test_kernel_determinism",
            start.elapsed(),
            format!(
                "Root value {} differs from expected {} (diff={})",
                root_val, expected_root, diff
            ),
        );
    }

    TestResult::passed("test_kernel_determinism", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_g06_kernel_single_node() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_single_node(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_single_level() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_single_level(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_deep_circuit() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_deep_circuit(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_large_circuit() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_large_circuit(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_max_variables() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_max_variables(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_sparse_variables() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_sparse_variables(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_many_literals() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_many_literals(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_log_underflow() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_log_underflow(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_logsumexp_stability() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_logsumexp_stability(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g06_kernel_determinism() {
        if let Some(ctx) = get_test_context() {
            let result = test_kernel_determinism(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }
}
