//! Category G02: Circuit Backward Kernel Tests
//!
//! Tests the backward evaluation kernels for XGCF circuits, covering adjoint propagation,
//! gradient accumulation, and numerical verification. These tests verify correct gradient
//! computation for neural-symbolic integration.

use crate::harness::xgcf::{
    gen_and_circuit, gen_decision_circuit, gen_deep_chain_circuit, gen_or_circuit,
    gen_single_lit_circuit, numerical_gradient, run_tiny_xgcf_backward, tiny_xgcf_spec,
    TinyXgcfSpec,
};
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g02_circuit_backward");
    let start = Instant::now();

    // G02.1: Adjoint Propagation (4 tests)
    results.add_result(test_backward_root_adjoint(ctx));
    results.add_result(test_backward_and_propagate(ctx));
    results.add_result(test_backward_or_propagate(ctx));
    results.add_result(test_backward_decision_propagate(ctx));

    // G02.2: Gradient Accumulation (4 tests)
    results.add_result(test_backward_lit_grad(ctx));
    results.add_result(test_backward_decision_grad(ctx));
    results.add_result(test_backward_gradient_accumulation(ctx));
    results.add_result(test_backward_negative_lit_grad(ctx));

    // G02.3: Numerical Verification (4 tests)
    results.add_result(test_backward_vs_numerical_diff(ctx));
    results.add_result(test_backward_chain_rule(ctx));
    results.add_result(test_backward_gradient_stability(ctx));
    results.add_result(test_backward_log_space_accuracy(ctx));

    results.set_duration(start.elapsed());
    results
}

// =============================================================================
// G02.1: Adjoint Propagation Tests
// =============================================================================

/// Test 1: Verify root adjoint is initialized to 1.0
///
/// The backward pass starts with adj[root] = 1.0, representing the gradient of the
/// output with respect to itself.
fn test_backward_root_adjoint(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_root_adjoint",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Verify root adjoint is 1.0
    let root_adj = run.adj[spec.root as usize];
    if (root_adj - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_root_adjoint",
            start.elapsed(),
            format!(
                "Root adjoint should be 1.0, got {} (diff={})",
                root_adj,
                (root_adj - 1.0).abs()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_root_adjoint",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_root_adjoint", start.elapsed())
}

/// Test 2: AND node adjoint propagation
///
/// For AND nodes, each child receives the parent's adjoint unchanged because
/// d(log(a*b))/d(log(a)) = 1 and d(log(a*b))/d(log(b)) = 1.
fn test_backward_and_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_and_circuit();
    // AND circuit: node 0 = Lit(+1), node 1 = Lit(+2), node 2 = AND(0,1) [root]
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_and_propagate",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Root adjoint should be 1.0
    let root_adj = run.adj[spec.root as usize];
    if (root_adj - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_and_propagate",
            start.elapsed(),
            format!("Root adjoint should be 1.0, got {}", root_adj),
        );
    }

    // For AND, children get parent's adjoint (chain rule: d(a+b)/da = 1)
    // Child 0 and child 1 should each have adjoint = root_adj = 1.0
    let child0_adj = run.adj[0];
    let child1_adj = run.adj[1];

    if (child0_adj - root_adj).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_and_propagate",
            start.elapsed(),
            format!(
                "AND child 0 adjoint should equal parent adjoint {}, got {}",
                root_adj, child0_adj
            ),
        );
    }

    if (child1_adj - root_adj).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_and_propagate",
            start.elapsed(),
            format!(
                "AND child 1 adjoint should equal parent adjoint {}, got {}",
                root_adj, child1_adj
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_and_propagate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_and_propagate", start.elapsed())
}

/// Test 3: OR node adjoint propagation
///
/// For OR nodes (logsumexp), children get adjoint weighted by softmax:
/// adj[child_i] += adj[parent] * exp(val[child_i] - val[parent])
fn test_backward_or_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_or_circuit();
    // OR circuit: node 0 = Lit(+1), node 1 = Lit(+2), node 2 = OR(0,1) [root]
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_or_propagate",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Root adjoint should be 1.0
    let root_adj = run.adj[spec.root as usize];
    let root_val = run.values[spec.root as usize];

    if (root_adj - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_or_propagate",
            start.elapsed(),
            format!("Root adjoint should be 1.0, got {}", root_adj),
        );
    }

    // For OR (logsumexp), child adjoint = parent_adj * exp(child_val - parent_val)
    let child0_val = run.values[0];
    let child1_val = run.values[1];

    let expected_adj0 = root_adj * (child0_val - root_val).exp();
    let expected_adj1 = root_adj * (child1_val - root_val).exp();

    let child0_adj = run.adj[0];
    let child1_adj = run.adj[1];

    if (child0_adj - expected_adj0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_or_propagate",
            start.elapsed(),
            format!(
                "OR child 0 adjoint should be {}, got {} (diff={})",
                expected_adj0,
                child0_adj,
                (child0_adj - expected_adj0).abs()
            ),
        );
    }

    if (child1_adj - expected_adj1).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_or_propagate",
            start.elapsed(),
            format!(
                "OR child 1 adjoint should be {}, got {} (diff={})",
                expected_adj1,
                child1_adj,
                (child1_adj - expected_adj1).abs()
            ),
        );
    }

    // Verify softmax weights sum to 1
    let softmax_sum = expected_adj0 + expected_adj1;
    if (softmax_sum - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_or_propagate",
            start.elapsed(),
            format!("OR softmax weights should sum to 1.0, got {}", softmax_sum),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_or_propagate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_or_propagate", start.elapsed())
}

/// Test 4: Decision node adjoint propagation
///
/// Decision nodes propagate adjoint to their children weighted by branch probability:
/// adj[child_false] += adj[parent] * exp(log_false + val_false - val_parent)
/// adj[child_true] += adj[parent] * exp(log_true + val_true - val_parent)
fn test_backward_decision_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();
    // Decision circuit: node 0 = CONST1, node 1 = Lit(+1), node 2 = Decision(var2, 0, 1) [root]
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_decision_propagate",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Root adjoint should be 1.0
    let root_adj = run.adj[spec.root as usize];
    let root_val = run.values[spec.root as usize];

    if (root_adj - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!("Root adjoint should be 1.0, got {}", root_adj),
        );
    }

    // Decision var is at index 2 in the decision_var array for node 2
    let decision_var = spec.decision_var[spec.root as usize] as usize;
    let log_false = spec.var_log_false[decision_var];
    let log_true = spec.var_log_true[decision_var];

    let val_false = run.values[0]; // CONST1 = 0.0
    let val_true = run.values[1]; // Lit(+1)

    // Expected child adjoints based on branch probability
    let expected_adj_false = root_adj * (log_false + val_false - root_val).exp();
    let expected_adj_true = root_adj * (log_true + val_true - root_val).exp();

    let child_false_adj = run.adj[0];
    let child_true_adj = run.adj[1];

    if (child_false_adj - expected_adj_false).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!(
                "Decision false-child adjoint should be {}, got {} (diff={})",
                expected_adj_false,
                child_false_adj,
                (child_false_adj - expected_adj_false).abs()
            ),
        );
    }

    if (child_true_adj - expected_adj_true).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!(
                "Decision true-child adjoint should be {}, got {} (diff={})",
                expected_adj_true,
                child_true_adj,
                (child_true_adj - expected_adj_true).abs()
            ),
        );
    }

    // Verify branch probabilities sum to 1
    let prob_sum = expected_adj_false + expected_adj_true;
    if (prob_sum - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!(
                "Decision branch probabilities should sum to 1.0, got {}",
                prob_sum
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_decision_propagate", start.elapsed())
}

// =============================================================================
// G02.2: Gradient Accumulation Tests
// =============================================================================

/// Test 5: Positive literal gradient contribution
///
/// A positive literal Lit(+v) contributes to grad_true[v] with the node's adjoint.
fn test_backward_lit_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_single_lit_circuit(1);
    // Single literal circuit: node 0 = Lit(+1) [root]
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_lit_grad",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Positive literal should contribute to grad_true
    let expected_grad_true = 1.0; // adjoint of root = 1.0
    let expected_grad_false = 0.0;

    if (run.grad_true[1] - expected_grad_true).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_lit_grad",
            start.elapsed(),
            format!(
                "Positive literal grad_true[1] should be {}, got {}",
                expected_grad_true, run.grad_true[1]
            ),
        );
    }

    if (run.grad_false[1] - expected_grad_false).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_lit_grad",
            start.elapsed(),
            format!(
                "Positive literal grad_false[1] should be {}, got {}",
                expected_grad_false, run.grad_false[1]
            ),
        );
    }

    // Verify against expected values from spec
    for (i, (&actual, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_backward_lit_grad",
                start.elapsed(),
                format!(
                    "grad_true[{}] mismatch: expected {}, got {}",
                    i, expected, actual
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_lit_grad",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_lit_grad", start.elapsed())
}

/// Test 6: Decision node gradient computation
///
/// Decision nodes contribute gradients to both grad_true and grad_false
/// for their decision variable based on branch probabilities.
fn test_backward_decision_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_decision_grad",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Verify decision gradients match expected values from spec
    for (i, (&actual, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_backward_decision_grad",
                start.elapsed(),
                format!(
                    "grad_true[{}] mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    for (i, (&actual, &expected)) in run
        .grad_false
        .iter()
        .zip(spec.expected_grad_false.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_backward_decision_grad",
                start.elapsed(),
                format!(
                    "grad_false[{}] mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify decision var has non-zero gradient
    let decision_var = spec.decision_var[spec.root as usize] as usize;
    if run.grad_true[decision_var].abs() < 1e-10 && run.grad_false[decision_var].abs() < 1e-10 {
        return TestResult::error(
            "test_backward_decision_grad",
            start.elapsed(),
            format!(
                "Decision variable {} should have non-zero gradient",
                decision_var
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_decision_grad",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_decision_grad", start.elapsed())
}

/// Test 7: Multiple paths accumulate gradients
///
/// When a variable appears in multiple paths, gradients should accumulate.
fn test_backward_gradient_accumulation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_gradient_accumulation",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Verify all gradients match expected values from spec
    for (i, (&actual, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_backward_gradient_accumulation",
                start.elapsed(),
                format!(
                    "grad_true[{}] mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    for (i, (&actual, &expected)) in run
        .grad_false
        .iter()
        .zip(spec.expected_grad_false.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        if diff > 1e-10 {
            return TestResult::error(
                "test_backward_gradient_accumulation",
                start.elapsed(),
                format!(
                    "grad_false[{}] mismatch: expected {}, got {} (diff={})",
                    i, expected, actual, diff
                ),
            );
        }
    }

    // Verify that gradients accumulate properly for variables appearing in multiple places
    // In tiny_xgcf_spec, var3 is a decision variable, so it should have both true and false gradients
    if run.grad_true[3].abs() < 1e-10 {
        return TestResult::error(
            "test_backward_gradient_accumulation",
            start.elapsed(),
            format!(
                "grad_true[3] should be non-zero for decision variable, got {}",
                run.grad_true[3]
            ),
        );
    }

    if run.grad_false[3].abs() < 1e-10 {
        return TestResult::error(
            "test_backward_gradient_accumulation",
            start.elapsed(),
            format!(
                "grad_false[3] should be non-zero for decision variable, got {}",
                run.grad_false[3]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_gradient_accumulation",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_gradient_accumulation", start.elapsed())
}

/// Test 8: Negative literal gradient contribution
///
/// A negative literal Lit(-v) contributes to grad_false[v] with the node's adjoint.
fn test_backward_negative_lit_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;

    // Create a negative literal circuit: root = Lit(-1)
    let spec = TinyXgcfSpec {
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

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_negative_lit_grad",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Negative literal should contribute to grad_false, not grad_true
    let expected_grad_true = 0.0;
    let expected_grad_false = 1.0; // adjoint of root = 1.0

    if (run.grad_true[1] - expected_grad_true).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_negative_lit_grad",
            start.elapsed(),
            format!(
                "Negative literal grad_true[1] should be {}, got {}",
                expected_grad_true, run.grad_true[1]
            ),
        );
    }

    if (run.grad_false[1] - expected_grad_false).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_negative_lit_grad",
            start.elapsed(),
            format!(
                "Negative literal grad_false[1] should be {}, got {}",
                expected_grad_false, run.grad_false[1]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_negative_lit_grad",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_negative_lit_grad", start.elapsed())
}

// =============================================================================
// G02.3: Numerical Verification Tests
// =============================================================================

/// Test 9: Compare GPU gradients vs finite differences
///
/// Validates gradient correctness by comparing against numerical differentiation.
fn test_backward_vs_numerical_diff(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_vs_numerical_diff",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    let eps = 1e-6;
    let tolerance = 1e-5;

    // Check numerical gradients for each variable
    for var in 1..=spec.num_vars {
        let (num_grad_true, num_grad_false) = match numerical_gradient(ctx, &spec, var, eps) {
            Ok(g) => g,
            Err(e) => {
                return TestResult::error(
                    "test_backward_vs_numerical_diff",
                    start.elapsed(),
                    format!(
                        "Failed to compute numerical gradient for var {}: {}",
                        var, e
                    ),
                );
            }
        };

        let gpu_grad_true = run.grad_true[var];
        let gpu_grad_false = run.grad_false[var];

        let diff_true = (gpu_grad_true - num_grad_true).abs();
        let diff_false = (gpu_grad_false - num_grad_false).abs();

        if diff_true > tolerance {
            return TestResult::error(
                "test_backward_vs_numerical_diff",
                start.elapsed(),
                format!(
                    "var {} grad_true mismatch: GPU={}, numerical={} (diff={})",
                    var, gpu_grad_true, num_grad_true, diff_true
                ),
            );
        }

        if diff_false > tolerance {
            return TestResult::error(
                "test_backward_vs_numerical_diff",
                start.elapsed(),
                format!(
                    "var {} grad_false mismatch: GPU={}, numerical={} (diff={})",
                    var, gpu_grad_false, num_grad_false, diff_false
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_vs_numerical_diff",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_vs_numerical_diff", start.elapsed())
}

/// Test 10: Chain rule through multiple levels
///
/// Verifies that the chain rule is correctly applied through a deep chain of nodes.
fn test_backward_chain_rule(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let depth = 10;
    let spec = gen_deep_chain_circuit(depth);
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_chain_rule",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // In a chain of ANDs, adjoint should propagate unchanged through all levels
    // Root adjoint = 1.0, and each AND passes it to its child unchanged
    for i in 0..=depth {
        let adj = run.adj[i];
        if (adj - 1.0).abs() > 1e-10 {
            return TestResult::error(
                "test_backward_chain_rule",
                start.elapsed(),
                format!(
                    "Node {} adjoint should be 1.0 (chain rule through AND), got {}",
                    i, adj
                ),
            );
        }
    }

    // The gradient should reach the literal at the bottom
    // grad_true[1] should be 1.0 (the adjoint of the literal)
    if (run.grad_true[1] - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_chain_rule",
            start.elapsed(),
            format!(
                "grad_true[1] should be 1.0 after chain rule, got {}",
                run.grad_true[1]
            ),
        );
    }

    // Verify against expected values from spec
    for (i, (&actual, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_backward_chain_rule",
                start.elapsed(),
                format!(
                    "grad_true[{}] mismatch: expected {}, got {}",
                    i, expected, actual
                ),
            );
        }
    }

    // Compare with numerical gradient
    let eps = 1e-6;
    let tolerance = 1e-5;
    let (num_grad_true, _) = match numerical_gradient(ctx, &spec, 1, eps) {
        Ok(g) => g,
        Err(e) => {
            return TestResult::error(
                "test_backward_chain_rule",
                start.elapsed(),
                format!("Failed to compute numerical gradient: {}", e),
            );
        }
    };

    let diff = (run.grad_true[1] - num_grad_true).abs();
    if diff > tolerance {
        return TestResult::error(
            "test_backward_chain_rule",
            start.elapsed(),
            format!(
                "Chain rule gradient differs from numerical: GPU={}, numerical={} (diff={})",
                run.grad_true[1], num_grad_true, diff
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_chain_rule",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_chain_rule", start.elapsed())
}

/// Test 11: Gradient stability with extreme values
///
/// Verifies that gradients remain finite (no NaN/Inf) even with extreme log values.
fn test_backward_gradient_stability(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    const AND: u8 = 3;

    // Create a circuit with extreme values
    // AND(Lit(+1), Lit(+2)) where var1 has an extremely small probability
    let spec = TinyXgcfSpec {
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
        // Extreme value: log(1e-300) for var1
        var_log_true: vec![0.0, 1e-300_f64.ln(), 0.5_f64.ln()],
        var_log_false: vec![0.0, (1.0 - 1e-300_f64).ln(), 0.5_f64.ln()],
        expected_values: vec![
            1e-300_f64.ln(),
            0.5_f64.ln(),
            1e-300_f64.ln() + 0.5_f64.ln(),
        ],
        expected_grad_true: vec![0.0, 1.0, 1.0],
        expected_grad_false: vec![0.0, 0.0, 0.0],
    };

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    // Check that no values are NaN or Inf
    for (i, &val) in run.values.iter().enumerate() {
        if val.is_nan() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("values[{}] is NaN with extreme input values", i),
            );
        }
        // Values can be -inf for log(0), but not +inf
        if val.is_infinite() && val.is_sign_positive() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("values[{}] is +Inf with extreme input values", i),
            );
        }
    }

    for (i, &adj) in run.adj.iter().enumerate() {
        if adj.is_nan() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("adj[{}] is NaN with extreme input values", i),
            );
        }
        if adj.is_infinite() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("adj[{}] is Inf with extreme input values", i),
            );
        }
    }

    for (i, &g) in run.grad_true.iter().enumerate() {
        if g.is_nan() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("grad_true[{}] is NaN with extreme input values", i),
            );
        }
        if g.is_infinite() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("grad_true[{}] is Inf with extreme input values", i),
            );
        }
    }

    for (i, &g) in run.grad_false.iter().enumerate() {
        if g.is_nan() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("grad_false[{}] is NaN with extreme input values", i),
            );
        }
        if g.is_infinite() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("grad_false[{}] is Inf with extreme input values", i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_gradient_stability",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_gradient_stability", start.elapsed())
}

/// Test 12: Log-space accuracy with tight tolerance
///
/// Verifies that gradient computations in log space achieve relative error < 1e-10.
fn test_backward_log_space_accuracy(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();
    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_backward_log_space_accuracy",
                start.elapsed(),
                format!("Failed to run backward pass: {}", e),
            );
        }
    };

    let tolerance = 1e-10;

    // Check relative error for all gradients
    for (i, (&actual, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        let abs_diff = (actual - expected).abs();
        let rel_err = if expected.abs() > 1e-15 {
            abs_diff / expected.abs()
        } else {
            abs_diff // Use absolute error for near-zero expected values
        };

        if abs_diff > tolerance && rel_err > tolerance {
            return TestResult::error(
                "test_backward_log_space_accuracy",
                start.elapsed(),
                format!(
                    "grad_true[{}] exceeds tolerance: expected {}, got {} (abs_diff={}, rel_err={})",
                    i, expected, actual, abs_diff, rel_err
                ),
            );
        }
    }

    for (i, (&actual, &expected)) in run
        .grad_false
        .iter()
        .zip(spec.expected_grad_false.iter())
        .enumerate()
    {
        let abs_diff = (actual - expected).abs();
        let rel_err = if expected.abs() > 1e-15 {
            abs_diff / expected.abs()
        } else {
            abs_diff // Use absolute error for near-zero expected values
        };

        if abs_diff > tolerance && rel_err > tolerance {
            return TestResult::error(
                "test_backward_log_space_accuracy",
                start.elapsed(),
                format!(
                    "grad_false[{}] exceeds tolerance: expected {}, got {} (abs_diff={}, rel_err={})",
                    i, expected, actual, abs_diff, rel_err
                ),
            );
        }
    }

    // Check forward values accuracy as well (they affect backward correctness)
    for (i, (&actual, &expected)) in run
        .values
        .iter()
        .zip(spec.expected_values.iter())
        .enumerate()
    {
        // Handle -inf comparison specially
        if expected.is_infinite() && expected.is_sign_negative() {
            if !actual.is_infinite() || !actual.is_sign_negative() {
                return TestResult::error(
                    "test_backward_log_space_accuracy",
                    start.elapsed(),
                    format!("values[{}] should be -inf, got {}", i, actual),
                );
            }
            continue;
        }

        let abs_diff = (actual - expected).abs();
        let rel_err = if expected.abs() > 1e-15 {
            abs_diff / expected.abs()
        } else {
            abs_diff
        };

        if abs_diff > tolerance && rel_err > tolerance {
            return TestResult::error(
                "test_backward_log_space_accuracy",
                start.elapsed(),
                format!(
                    "values[{}] exceeds tolerance: expected {}, got {} (abs_diff={}, rel_err={})",
                    i, expected, actual, abs_diff, rel_err
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_backward_log_space_accuracy",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_backward_log_space_accuracy", start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestContext;

    fn get_test_context() -> Option<TestContext> {
        TestContext::new().ok()
    }

    #[test]
    fn test_g02_backward_root_adjoint() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_root_adjoint(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_and_propagate() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_and_propagate(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_or_propagate() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_or_propagate(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_decision_propagate() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_decision_propagate(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_lit_grad() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_lit_grad(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_decision_grad() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_decision_grad(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_gradient_accumulation() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_gradient_accumulation(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_negative_lit_grad() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_negative_lit_grad(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_vs_numerical_diff() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_vs_numerical_diff(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_chain_rule() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_chain_rule(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_gradient_stability() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_gradient_stability(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }

    #[test]
    fn test_g02_backward_log_space_accuracy() {
        if let Some(ctx) = get_test_context() {
            let result = test_backward_log_space_accuracy(&ctx);
            assert!(
                result.status.is_passed(),
                "Test failed: {:?}",
                result.diagnostic
            );
        }
    }
}
