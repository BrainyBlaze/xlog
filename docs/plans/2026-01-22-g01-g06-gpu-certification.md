# G01-G06 GPU Certification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 50 production-grade GPU certification tests (G01-G06) for XGCF circuit evaluation kernels to validate neural-symbolic integration in v0.4.0-alpha.

**Architecture:** Tests verify `circuit.ptx` kernels (`xgcf_forward_level`, `xgcf_backward_level_propagate`, `xgcf_backward_level_decision_grad`, `xgcf_backward_level_lit_grad`) using the existing `xlog-cuda-tests` harness. Each category follows the C01-C25 pattern with `run_all(ctx: &TestContext) -> CategoryResult`. Tests use CPU reference implementations for verification.

**Tech Stack:** Rust, cudarc, xlog-cuda, xlog-prob (XGCF/GpuXgcf), existing test harness (TestContext, CategoryResult, TestResult).

---

## Task 1: Extend Test Harness with Circuit Generators

**Files:**
- Modify: `crates/xlog-cuda-tests/src/harness/xgcf.rs`

**Step 1: Add circuit generation utilities**

Add to `crates/xlog-cuda-tests/src/harness/xgcf.rs`:

```rust
/// Generate a single-literal circuit: root = Lit(+var)
pub fn gen_single_lit_circuit(var: u32) -> TinyXgcfSpec {
    const LIT: u8 = 2;

    let num_nodes = 1;
    let num_vars = var as usize;
    let root = 0;

    let node_type = vec![LIT];
    let child_offsets = vec![0, 0];
    let child_indices = vec![];
    let lit = vec![var as i32];
    let decision_var = vec![0];
    let decision_child_false = vec![0];
    let decision_child_true = vec![0];
    let level_nodes = vec![0];
    let levels = vec![(0, 1)];

    // Default weights: p(var=true) = 0.7
    let mut var_log_true = vec![0.0; num_vars + 1];
    let mut var_log_false = vec![0.0; num_vars + 1];
    var_log_true[var as usize] = 0.7_f64.ln();
    var_log_false[var as usize] = 0.3_f64.ln();

    let expected_values = vec![var_log_true[var as usize]];
    let mut expected_grad_true = vec![0.0; num_vars + 1];
    let mut expected_grad_false = vec![0.0; num_vars + 1];
    expected_grad_true[var as usize] = 1.0;

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate an AND circuit: root = AND(Lit(+1), Lit(+2))
pub fn gen_and_circuit() -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const AND: u8 = 3;

    // Nodes: 0=Lit(+1), 1=Lit(+2), 2=AND(0,1)
    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![LIT, LIT, AND];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0, 0, 0];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 0];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64;
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = var_log_true[1];
    let v1 = var_log_true[2];
    let v2 = v0 + v1; // AND = sum in log space
    let expected_values = vec![v0, v1, v2];

    // Gradients for AND: all children get adjoint 1.0
    let expected_grad_true = vec![0.0, 1.0, 1.0];
    let expected_grad_false = vec![0.0, 0.0, 0.0];

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate an OR circuit: root = OR(Lit(+1), Lit(+2))
pub fn gen_or_circuit() -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const OR: u8 = 4;

    // Nodes: 0=Lit(+1), 1=Lit(+2), 2=OR(0,1)
    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![LIT, LIT, OR];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0, 0, 0];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 0];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64;
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = var_log_true[1];
    let v1 = var_log_true[2];
    let v2 = logsumexp2(v0, v1); // OR = logsumexp
    let expected_values = vec![v0, v1, v2];

    // Gradients for OR: children get adjoint * softmax weight
    let p_child0 = (v0 - v2).exp();
    let p_child1 = (v1 - v2).exp();
    let expected_grad_true = vec![0.0, p_child0, p_child1];
    let expected_grad_false = vec![0.0, 0.0, 0.0];

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a Decision circuit: root = Decision(var, false_child=Const1, true_child=Lit(+1))
pub fn gen_decision_circuit() -> TinyXgcfSpec {
    const CONST1: u8 = 1;
    const LIT: u8 = 2;
    const DECISION: u8 = 5;

    // Nodes: 0=Const1, 1=Lit(+1), 2=Decision(var=2, false=0, true=1)
    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![CONST1, LIT, DECISION];
    let child_offsets = vec![0, 0, 0, 0];
    let child_indices = vec![];
    let lit = vec![0, 1, 0];
    let decision_var = vec![0, 0, 2];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 1];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64; // decision variable
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = 0.0; // Const1
    let v1 = var_log_true[1]; // Lit(+1)
    // Decision: logsumexp(log_false + v_false, log_true + v_true)
    let v2 = logsumexp2(var_log_false[2] + v0, var_log_true[2] + v1);
    let expected_values = vec![v0, v1, v2];

    let p_false = (var_log_false[2] + v0 - v2).exp();
    let p_true = (var_log_true[2] + v1 - v2).exp();

    let expected_grad_true = vec![0.0, p_true, p_true];
    let expected_grad_false = vec![0.0, 0.0, p_false];

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a large circuit with N parallel literals under an OR node
pub fn gen_large_or_circuit(num_vars: usize) -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const OR: u8 = 4;

    // Nodes: 0..num_vars = Lit(+i), num_vars = OR(all lits)
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

    // Uniform weights
    let p = 0.5_f64;
    let mut var_log_true = vec![0.0; num_vars + 1];
    let mut var_log_false = vec![0.0; num_vars + 1];
    for i in 1..=num_vars {
        var_log_true[i] = p.ln();
        var_log_false[i] = (1.0 - p).ln();
    }

    // Compute expected values
    let lit_val = p.ln();
    let mut expected_values = vec![lit_val; num_vars];
    // OR of N equal values: logsumexp = lit_val + ln(N)
    let or_val = lit_val + (num_vars as f64).ln();
    expected_values.push(or_val);

    // Each literal gets equal gradient = 1/N
    let grad_per_lit = 1.0 / num_vars as f64;
    let mut expected_grad_true = vec![0.0; num_vars + 1];
    for i in 1..=num_vars {
        expected_grad_true[i] = grad_per_lit;
    }
    let expected_grad_false = vec![0.0; num_vars + 1];

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a deep chain circuit: AND(AND(AND(...Lit(1)...)))
pub fn gen_deep_chain_circuit(depth: usize) -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const AND: u8 = 3;

    // Node 0 = Lit(+1), Nodes 1..depth = AND nodes chaining back
    let num_nodes = depth + 1;
    let num_vars = 1;
    let root = depth as u32;

    let mut node_type = vec![LIT];
    for _ in 0..depth {
        node_type.push(AND);
    }

    // Each AND has exactly one child (the previous node)
    let mut child_offsets: Vec<u32> = vec![0]; // Lit has no children
    let mut child_indices: Vec<u32> = vec![];
    for i in 0..depth {
        child_offsets.push(child_indices.len() as u32);
        child_indices.push(i as u32);
    }
    child_offsets.push(child_indices.len() as u32);

    let mut lit = vec![1i32];
    lit.extend(vec![0i32; depth]);

    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];

    // Each node is at its own level
    let level_nodes: Vec<u32> = (0..num_nodes as u32).collect();
    let levels: Vec<(u32, u32)> = (0..num_nodes).map(|i| (i as u32, 1)).collect();

    let p = 0.7_f64;
    let var_log_true = vec![0.0, p.ln()];
    let var_log_false = vec![0.0, (1.0 - p).ln()];

    // All nodes have the same value (AND of single child = child value)
    let lit_val = p.ln();
    let expected_values = vec![lit_val; num_nodes];

    // Gradient propagates unchanged through single-child ANDs
    let expected_grad_true = vec![0.0, 1.0];
    let expected_grad_false = vec![0.0, 0.0];

    TinyXgcfSpec {
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
        expected_grad_true,
        expected_grad_false,
    }
}

/// Compute numerical gradient for verification
pub fn numerical_gradient(
    ctx: &TestContext,
    spec: &TinyXgcfSpec,
    var: usize,
    eps: f64,
) -> xlog_core::Result<(f64, f64)> {
    // Gradient w.r.t. log_true[var]
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.var_log_true[var] += eps;
    spec_minus.var_log_true[var] -= eps;

    let values_plus = run_tiny_xgcf_forward(ctx, &spec_plus)?;
    let values_minus = run_tiny_xgcf_forward(ctx, &spec_minus)?;

    let grad_true = (values_plus[spec.root as usize] - values_minus[spec.root as usize]) / (2.0 * eps);

    // Gradient w.r.t. log_false[var]
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.var_log_false[var] += eps;
    spec_minus.var_log_false[var] -= eps;

    let values_plus = run_tiny_xgcf_forward(ctx, &spec_plus)?;
    let values_minus = run_tiny_xgcf_forward(ctx, &spec_minus)?;

    let grad_false = (values_plus[spec.root as usize] - values_minus[spec.root as usize]) / (2.0 * eps);

    Ok((grad_true, grad_false))
}
```

**Step 2: Run test to verify compilation**

Run: `cargo build -p xlog-cuda-tests --release`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add crates/xlog-cuda-tests/src/harness/xgcf.rs
git commit -m "feat(cert): add circuit generators for G01-G06 tests"
```

---

## Task 2: Implement G01 Circuit Forward Kernel Tests (8 tests)

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/g01_circuit_forward.rs`
- Modify: `crates/xlog-cuda-tests/src/categories/mod.rs`

**Step 1: Create G01 category file**

Create `crates/xlog-cuda-tests/src/categories/g01_circuit_forward.rs`:

```rust
//! Category G01: Circuit Forward Kernel Correctness
//!
//! Tests xgcf_forward_level PTX kernel correctness for all node types
//! and execution patterns.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::xgcf::{
    tiny_xgcf_spec, run_tiny_xgcf_forward,
    gen_single_lit_circuit, gen_and_circuit, gen_or_circuit,
    gen_decision_circuit, gen_large_or_circuit, gen_deep_chain_circuit,
};
use std::time::Instant;

/// Run all G01 tests.
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

/// G01.1a: Test CONST0/CONST1 node evaluation.
fn test_forward_const_nodes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const CONST0: u8 = 0;
    const CONST1: u8 = 1;

    // Create minimal circuit with just const nodes
    let spec = crate::harness::xgcf::TinyXgcfSpec {
        num_nodes: 2,
        num_vars: 0,
        root: 1, // CONST1
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
        Err(e) => return TestResult::error("test_forward_const_nodes", start.elapsed(), format!("Forward failed: {}", e)),
    };

    // CONST0 should be -inf
    if !values[0].is_infinite() || values[0] > 0.0 {
        return TestResult::error(
            "test_forward_const_nodes",
            start.elapsed(),
            format!("CONST0 should be -inf, got {}", values[0]),
        );
    }

    // CONST1 should be 0.0
    if (values[1] - 0.0).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_const_nodes",
            start.elapsed(),
            format!("CONST1 should be 0.0, got {}", values[1]),
        );
    }

    TestResult::passed("test_forward_const_nodes", start.elapsed())
}

/// G01.1b: Test positive and negative literal evaluation.
fn test_forward_lit_nodes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_single_lit_circuit(1);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_lit_nodes", start.elapsed(), format!("Forward failed: {}", e)),
    };

    let expected = spec.var_log_true[1];
    let actual = values[spec.root as usize];

    if (actual - expected).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_lit_nodes",
            start.elapsed(),
            format!("Lit(+1) expected {}, got {}", expected, actual),
        );
    }

    // Test negative literal
    const LIT: u8 = 2;
    let neg_spec = crate::harness::xgcf::TinyXgcfSpec {
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
        Err(e) => return TestResult::error("test_forward_lit_nodes", start.elapsed(), format!("Negative lit forward failed: {}", e)),
    };

    let neg_expected = neg_spec.var_log_false[1];
    if (neg_values[0] - neg_expected).abs() > 1e-10 {
        return TestResult::error(
            "test_forward_lit_nodes",
            start.elapsed(),
            format!("Lit(-1) expected {}, got {}", neg_expected, neg_values[0]),
        );
    }

    TestResult::passed("test_forward_lit_nodes", start.elapsed())
}

/// G01.1c: Test AND node evaluation (sum in log space).
fn test_forward_and_node(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_and_circuit();

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_and_node", start.elapsed(), format!("Forward failed: {}", e)),
    };

    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_forward_and_node",
                start.elapsed(),
                format!("Node {} expected {}, got {}", i, expected, actual),
            );
        }
    }

    TestResult::passed("test_forward_and_node", start.elapsed())
}

/// G01.1d: Test OR node evaluation (logsumexp).
fn test_forward_or_node(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_or_circuit();

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_or_node", start.elapsed(), format!("Forward failed: {}", e)),
    };

    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_forward_or_node",
                start.elapsed(),
                format!("Node {} expected {}, got {}", i, expected, actual),
            );
        }
    }

    TestResult::passed("test_forward_or_node", start.elapsed())
}

/// G01.2a: Test basic decision node evaluation.
fn test_forward_decision_basic(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_decision_basic", start.elapsed(), format!("Forward failed: {}", e)),
    };

    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_forward_decision_basic",
                start.elapsed(),
                format!("Node {} expected {}, got {}", i, expected, actual),
            );
        }
    }

    TestResult::passed("test_forward_decision_basic", start.elapsed())
}

/// G01.2b: Test chain of decision nodes.
fn test_forward_decision_chain(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Use the tiny_xgcf_spec which has a decision chain
    let spec = tiny_xgcf_spec();

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_decision_chain", start.elapsed(), format!("Forward failed: {}", e)),
    };

    for (i, (&actual, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        // Use relative tolerance for log-space values
        let tol = if expected.abs() > 1e-10 { expected.abs() * 1e-9 } else { 1e-10 };
        if (actual - expected).abs() > tol {
            return TestResult::error(
                "test_forward_decision_chain",
                start.elapsed(),
                format!("Node {} expected {}, got {} (diff {})", i, expected, actual, (actual - expected).abs()),
            );
        }
    }

    TestResult::passed("test_forward_decision_chain", start.elapsed())
}

/// G01.3a: Test multi-level circuit evaluation.
fn test_forward_multi_level(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Deep chain creates many levels
    let spec = gen_deep_chain_circuit(10);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_multi_level", start.elapsed(), format!("Forward failed: {}", e)),
    };

    // All nodes should have the same value (AND of single child = child)
    let expected = spec.expected_values[0];
    for (i, &actual) in values.iter().enumerate() {
        if (actual - expected).abs() > 1e-10 {
            return TestResult::error(
                "test_forward_multi_level",
                start.elapsed(),
                format!("Node {} expected {}, got {}", i, expected, actual),
            );
        }
    }

    TestResult::passed("test_forward_multi_level", start.elapsed())
}

/// G01.3b: Test parallel execution at same level.
fn test_forward_parallel_same_level(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Large OR creates many nodes at the same level
    let spec = gen_large_or_circuit(1000);

    let values = match run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => return TestResult::error("test_forward_parallel_same_level", start.elapsed(), format!("Forward failed: {}", e)),
    };

    // Check root value
    let root_val = values[spec.root as usize];
    let expected_root = spec.expected_values[spec.root as usize];

    if (root_val - expected_root).abs() > 1e-6 {
        return TestResult::error(
            "test_forward_parallel_same_level",
            start.elapsed(),
            format!("Root expected {}, got {} (1000-way OR)", expected_root, root_val),
        );
    }

    // Verify all literals have same value
    let lit_val = spec.expected_values[0];
    for i in 0..1000 {
        if (values[i] - lit_val).abs() > 1e-10 {
            return TestResult::error(
                "test_forward_parallel_same_level",
                start.elapsed(),
                format!("Literal {} expected {}, got {}", i, lit_val, values[i]),
            );
        }
    }

    TestResult::passed("test_forward_parallel_same_level", start.elapsed())
}
```

**Step 2: Update mod.rs**

Add to `crates/xlog-cuda-tests/src/categories/mod.rs`:

```rust
pub mod g01_circuit_forward;
```

**Step 3: Run tests to verify**

Run: `cargo test -p xlog-cuda-tests g01 --release -- --nocapture`
Expected: 8 tests pass

**Step 4: Commit**

```bash
git add crates/xlog-cuda-tests/src/categories/g01_circuit_forward.rs crates/xlog-cuda-tests/src/categories/mod.rs
git commit -m "feat(cert): add G01 circuit forward kernel tests (8 tests)"
```

---

## Task 3: Implement G02 Circuit Backward Kernel Tests (12 tests)

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/g02_circuit_backward.rs`
- Modify: `crates/xlog-cuda-tests/src/categories/mod.rs`

**Step 1: Create G02 category file**

Create `crates/xlog-cuda-tests/src/categories/g02_circuit_backward.rs`:

```rust
//! Category G02: Circuit Backward Kernels Correctness
//!
//! Tests xgcf_backward_level_* PTX kernels for gradient computation.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::xgcf::{
    tiny_xgcf_spec, run_tiny_xgcf_backward, TinyXgcfRun,
    gen_single_lit_circuit, gen_and_circuit, gen_or_circuit,
    gen_decision_circuit, gen_large_or_circuit, numerical_gradient,
};
use std::time::Instant;

/// Run all G02 tests.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("g02_circuit_backward");
    let start = Instant::now();

    // G02.1: Adjoint propagation
    results.add_result(test_backward_root_adjoint(ctx));
    results.add_result(test_backward_and_propagate(ctx));
    results.add_result(test_backward_or_propagate(ctx));
    results.add_result(test_backward_decision_propagate(ctx));

    // G02.2: Gradient accumulation
    results.add_result(test_backward_lit_grad(ctx));
    results.add_result(test_backward_decision_grad(ctx));
    results.add_result(test_backward_gradient_accumulation(ctx));
    results.add_result(test_backward_negative_lit_grad(ctx));

    // G02.3: Numerical verification
    results.add_result(test_backward_vs_numerical_diff(ctx));
    results.add_result(test_backward_chain_rule(ctx));
    results.add_result(test_backward_gradient_stability(ctx));
    results.add_result(test_backward_log_space_accuracy(ctx));

    results.set_duration(start.elapsed());
    results
}

/// G02.1a: Verify root adjoint is initialized to 1.0.
fn test_backward_root_adjoint(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_root_adjoint", start.elapsed(), format!("Backward failed: {}", e)),
    };

    let root_adj = run.adj[spec.root as usize];
    if (root_adj - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_root_adjoint",
            start.elapsed(),
            format!("Root adjoint should be 1.0, got {}", root_adj),
        );
    }

    TestResult::passed("test_backward_root_adjoint", start.elapsed())
}

/// G02.1b: Test AND node adjoint propagation.
fn test_backward_and_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_and_circuit();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_and_propagate", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // For AND, each child gets the parent's adjoint
    let root_adj = run.adj[spec.root as usize];

    // Children are nodes 0 and 1
    for child in 0..2 {
        let child_adj = run.adj[child];
        if (child_adj - root_adj).abs() > 1e-10 {
            return TestResult::error(
                "test_backward_and_propagate",
                start.elapsed(),
                format!("AND child {} adjoint expected {}, got {}", child, root_adj, child_adj),
            );
        }
    }

    TestResult::passed("test_backward_and_propagate", start.elapsed())
}

/// G02.1c: Test OR node adjoint propagation.
fn test_backward_or_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_or_circuit();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_or_propagate", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // For OR, children get adjoint * softmax weight
    let root_val = run.values[spec.root as usize];
    let root_adj = run.adj[spec.root as usize];

    for child in 0..2 {
        let child_val = run.values[child];
        let expected_adj = root_adj * (child_val - root_val).exp();
        let actual_adj = run.adj[child];

        if (actual_adj - expected_adj).abs() > 1e-9 {
            return TestResult::error(
                "test_backward_or_propagate",
                start.elapsed(),
                format!("OR child {} adjoint expected {}, got {}", child, expected_adj, actual_adj),
            );
        }
    }

    TestResult::passed("test_backward_or_propagate", start.elapsed())
}

/// G02.1d: Test Decision node adjoint propagation.
fn test_backward_decision_propagate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_decision_propagate", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Decision node: children get weighted adjoint based on branch probability
    let root_val = run.values[spec.root as usize];
    let root_adj = run.adj[spec.root as usize];

    let false_child = spec.decision_child_false[spec.root as usize] as usize;
    let true_child = spec.decision_child_true[spec.root as usize] as usize;

    let var = spec.decision_var[spec.root as usize] as usize;
    let log_false = spec.var_log_false[var];
    let log_true = spec.var_log_true[var];

    let p_false = (log_false + run.values[false_child] - root_val).exp();
    let p_true = (log_true + run.values[true_child] - root_val).exp();

    let expected_false_adj = root_adj * p_false;
    let expected_true_adj = root_adj * p_true;

    if (run.adj[false_child] - expected_false_adj).abs() > 1e-9 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!("Decision false child adjoint expected {}, got {}", expected_false_adj, run.adj[false_child]),
        );
    }

    if (run.adj[true_child] - expected_true_adj).abs() > 1e-9 {
        return TestResult::error(
            "test_backward_decision_propagate",
            start.elapsed(),
            format!("Decision true child adjoint expected {}, got {}", expected_true_adj, run.adj[true_child]),
        );
    }

    TestResult::passed("test_backward_decision_propagate", start.elapsed())
}

/// G02.2a: Test literal gradient computation.
fn test_backward_lit_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_single_lit_circuit(1);

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_lit_grad", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Lit(+1) should contribute to grad_true[1]
    let expected_grad = 1.0; // adjoint is 1.0 for root
    let actual_grad = run.grad_true[1];

    if (actual_grad - expected_grad).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_lit_grad",
            start.elapsed(),
            format!("Lit(+1) grad_true expected {}, got {}", expected_grad, actual_grad),
        );
    }

    // grad_false should be 0 for positive literal
    if run.grad_false[1].abs() > 1e-10 {
        return TestResult::error(
            "test_backward_lit_grad",
            start.elapsed(),
            format!("Lit(+1) grad_false expected 0, got {}", run.grad_false[1]),
        );
    }

    TestResult::passed("test_backward_lit_grad", start.elapsed())
}

/// G02.2b: Test decision node gradient computation.
fn test_backward_decision_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = gen_decision_circuit();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_decision_grad", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Compare with expected gradients
    for var in 1..=spec.num_vars {
        let expected_true = spec.expected_grad_true[var];
        let expected_false = spec.expected_grad_false[var];
        let actual_true = run.grad_true[var];
        let actual_false = run.grad_false[var];

        if (actual_true - expected_true).abs() > 1e-9 {
            return TestResult::error(
                "test_backward_decision_grad",
                start.elapsed(),
                format!("var {} grad_true expected {}, got {}", var, expected_true, actual_true),
            );
        }

        if (actual_false - expected_false).abs() > 1e-9 {
            return TestResult::error(
                "test_backward_decision_grad",
                start.elapsed(),
                format!("var {} grad_false expected {}, got {}", var, expected_false, actual_false),
            );
        }
    }

    TestResult::passed("test_backward_decision_grad", start.elapsed())
}

/// G02.2c: Test gradient accumulation from multiple paths.
fn test_backward_gradient_accumulation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Use the full tiny spec which has multiple paths
    let spec = tiny_xgcf_spec();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_gradient_accumulation", start.elapsed(), format!("Backward failed: {}", e)),
    };

    for var in 1..=spec.num_vars {
        let expected_true = spec.expected_grad_true[var];
        let expected_false = spec.expected_grad_false[var];
        let actual_true = run.grad_true[var];
        let actual_false = run.grad_false[var];

        if (actual_true - expected_true).abs() > 1e-9 {
            return TestResult::error(
                "test_backward_gradient_accumulation",
                start.elapsed(),
                format!("var {} grad_true expected {}, got {}", var, expected_true, actual_true),
            );
        }

        if (actual_false - expected_false).abs() > 1e-9 {
            return TestResult::error(
                "test_backward_gradient_accumulation",
                start.elapsed(),
                format!("var {} grad_false expected {}, got {}", var, expected_false, actual_false),
            );
        }
    }

    TestResult::passed("test_backward_gradient_accumulation", start.elapsed())
}

/// G02.2d: Test negative literal gradient (sign flip).
fn test_backward_negative_lit_grad(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const LIT: u8 = 2;
    let spec = crate::harness::xgcf::TinyXgcfSpec {
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
        Err(e) => return TestResult::error("test_backward_negative_lit_grad", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Negative literal should contribute to grad_false
    if run.grad_true[1].abs() > 1e-10 {
        return TestResult::error(
            "test_backward_negative_lit_grad",
            start.elapsed(),
            format!("Lit(-1) grad_true expected 0, got {}", run.grad_true[1]),
        );
    }

    if (run.grad_false[1] - 1.0).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_negative_lit_grad",
            start.elapsed(),
            format!("Lit(-1) grad_false expected 1.0, got {}", run.grad_false[1]),
        );
    }

    TestResult::passed("test_backward_negative_lit_grad", start.elapsed())
}

/// G02.3a: Compare GPU gradients to numerical differentiation.
fn test_backward_vs_numerical_diff(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_vs_numerical_diff", start.elapsed(), format!("Backward failed: {}", e)),
    };

    let eps = 1e-6;

    for var in 1..=spec.num_vars {
        let (num_grad_true, num_grad_false) = match numerical_gradient(ctx, &spec, var, eps) {
            Ok(g) => g,
            Err(e) => return TestResult::error("test_backward_vs_numerical_diff", start.elapsed(), format!("Numerical gradient failed: {}", e)),
        };

        let gpu_grad_true = run.grad_true[var];
        let gpu_grad_false = run.grad_false[var];

        let tol = 1e-5;
        if (gpu_grad_true - num_grad_true).abs() > tol {
            return TestResult::error(
                "test_backward_vs_numerical_diff",
                start.elapsed(),
                format!("var {} grad_true: GPU={}, numerical={}", var, gpu_grad_true, num_grad_true),
            );
        }

        if (gpu_grad_false - num_grad_false).abs() > tol {
            return TestResult::error(
                "test_backward_vs_numerical_diff",
                start.elapsed(),
                format!("var {} grad_false: GPU={}, numerical={}", var, gpu_grad_false, num_grad_false),
            );
        }
    }

    TestResult::passed("test_backward_vs_numerical_diff", start.elapsed())
}

/// G02.3b: Test chain rule through multiple levels.
fn test_backward_chain_rule(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Deep chain tests chain rule propagation
    let spec = crate::harness::xgcf::gen_deep_chain_circuit(5);

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_chain_rule", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Gradient should be exactly 1.0 for the single literal
    let expected = spec.expected_grad_true[1];
    let actual = run.grad_true[1];

    if (actual - expected).abs() > 1e-10 {
        return TestResult::error(
            "test_backward_chain_rule",
            start.elapsed(),
            format!("Chain rule: grad_true expected {}, got {}", expected, actual),
        );
    }

    TestResult::passed("test_backward_chain_rule", start.elapsed())
}

/// G02.3c: Test gradient stability near extreme values.
fn test_backward_gradient_stability(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let mut spec = gen_or_circuit();

    // Set one probability very close to 0
    spec.var_log_true[1] = 1e-300_f64.ln();
    spec.var_log_false[1] = (1.0 - 1e-300).ln();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_gradient_stability", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Check for NaN/Inf in gradients
    for var in 1..=spec.num_vars {
        if run.grad_true[var].is_nan() || run.grad_true[var].is_infinite() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("var {} grad_true is NaN/Inf: {}", var, run.grad_true[var]),
            );
        }
        if run.grad_false[var].is_nan() || run.grad_false[var].is_infinite() {
            return TestResult::error(
                "test_backward_gradient_stability",
                start.elapsed(),
                format!("var {} grad_false is NaN/Inf: {}", var, run.grad_false[var]),
            );
        }
    }

    TestResult::passed("test_backward_gradient_stability", start.elapsed())
}

/// G02.3d: Test log-space gradient accuracy.
fn test_backward_log_space_accuracy(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = tiny_xgcf_spec();

    let run = match run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => return TestResult::error("test_backward_log_space_accuracy", start.elapsed(), format!("Backward failed: {}", e)),
    };

    // Verify gradients match expected with high precision
    for var in 1..=spec.num_vars {
        let expected_true = spec.expected_grad_true[var];
        let expected_false = spec.expected_grad_false[var];

        // Relative error check
        let rel_tol = 1e-10;

        if expected_true.abs() > 1e-15 {
            let rel_err = (run.grad_true[var] - expected_true).abs() / expected_true.abs();
            if rel_err > rel_tol {
                return TestResult::error(
                    "test_backward_log_space_accuracy",
                    start.elapsed(),
                    format!("var {} grad_true relative error {} > {}", var, rel_err, rel_tol),
                );
            }
        }

        if expected_false.abs() > 1e-15 {
            let rel_err = (run.grad_false[var] - expected_false).abs() / expected_false.abs();
            if rel_err > rel_tol {
                return TestResult::error(
                    "test_backward_log_space_accuracy",
                    start.elapsed(),
                    format!("var {} grad_false relative error {} > {}", var, rel_err, rel_tol),
                );
            }
        }
    }

    TestResult::passed("test_backward_log_space_accuracy", start.elapsed())
}
```

**Step 2: Update mod.rs**

Add to `crates/xlog-cuda-tests/src/categories/mod.rs`:

```rust
pub mod g02_circuit_backward;
```

**Step 3: Run tests**

Run: `cargo test -p xlog-cuda-tests g02 --release -- --nocapture`
Expected: 12 tests pass

**Step 4: Commit**

```bash
git add crates/xlog-cuda-tests/src/categories/g02_circuit_backward.rs crates/xlog-cuda-tests/src/categories/mod.rs
git commit -m "feat(cert): add G02 circuit backward kernel tests (12 tests)"
```

---

## Task 4: Implement G03-G06 Categories

Due to length constraints, I'll provide the structure. Each follows the same pattern.

**G03: Weight Injection Patterns (6 tests)** - `g03_weight_injection.rs`
- test_weight_upload_integrity
- test_weight_buffer_reuse
- test_weight_variable_mapping
- test_weight_extreme_values
- test_weight_uniform_distribution
- test_weight_sparse_nonzero

**G04: Transfer Efficiency (8 tests)** - `g04_transfer_efficiency.rs`
- test_transfer_weight_size
- test_transfer_gradient_size
- test_transfer_circuit_cached
- test_transfer_compute_ratio_mnist
- test_transfer_compute_ratio_large
- test_transfer_dominance_check
- test_training_loop_no_cpu_copy
- test_batch_eval_no_sync

**G05: Circuit Cache GPU Integration (6 tests)** - `g05_circuit_cache.rs`
- test_cache_hit_no_d4
- test_cache_hit_gpu_reuse
- test_cache_hit_speedup
- test_cache_key_predicate
- test_cache_key_same_structure
- test_cache_correctness

**G06: PTX Kernel Robustness (10 tests)** - `g06_ptx_robustness.rs`
- test_kernel_single_node
- test_kernel_single_level
- test_kernel_deep_circuit
- test_kernel_large_circuit
- test_kernel_max_variables
- test_kernel_sparse_variables
- test_kernel_many_literals
- test_kernel_log_underflow
- test_kernel_logsumexp_stability
- test_kernel_determinism

---

## Task 5: Update Certification Suite

**Files:**
- Modify: `crates/xlog-cuda-tests/tests/certification_suite.rs`

**Step 1: Add G01-G06 to certification suite**

Update certification_suite.rs to include:

```rust
// After C25...

println!("Running G01: Circuit Forward...");
results.add_category(categories::g01_circuit_forward::run_all(&ctx));

println!("Running G02: Circuit Backward...");
results.add_category(categories::g02_circuit_backward::run_all(&ctx));

println!("Running G03: Weight Injection...");
results.add_category(categories::g03_weight_injection::run_all(&ctx));

println!("Running G04: Transfer Efficiency...");
results.add_category(categories::g04_transfer_efficiency::run_all(&ctx));

println!("Running G05: Circuit Cache...");
results.add_category(categories::g05_circuit_cache::run_all(&ctx));

println!("Running G06: PTX Robustness...");
results.add_category(categories::g06_ptx_robustness::run_all(&ctx));
```

**Step 2: Run full suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`
Expected: 200 tests pass (150 C01-C25 + 50 G01-G06)

**Step 3: Commit**

```bash
git add crates/xlog-cuda-tests/tests/certification_suite.rs
git commit -m "feat(cert): integrate G01-G06 into certification suite (200 total tests)"
```

---

## Summary

| Task | Category | Tests | Key Focus |
|------|----------|-------|-----------|
| 1 | Harness | - | Circuit generators, numerical gradient |
| 2 | G01 | 8 | Forward kernel (const, lit, and, or, decision) |
| 3 | G02 | 12 | Backward kernels (adjoint, gradients, numerical verification) |
| 4a | G03 | 6 | Weight buffer management |
| 4b | G04 | 8 | Transfer efficiency, 0% CPU bottleneck |
| 4c | G05 | 6 | Circuit cache GPU integration |
| 4d | G06 | 10 | PTX robustness (large, deep, edge cases) |
| 5 | Suite | - | Integration into certification_suite.rs |

**Total: 50 new tests across 6 categories**

---

Plan complete and saved to `docs/plans/2026-01-22-g01-g06-gpu-certification.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
