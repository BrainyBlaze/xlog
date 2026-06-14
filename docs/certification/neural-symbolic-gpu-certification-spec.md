# XLOG Neural-Symbolic GPU Certification Test Specification

> **Certification target:** Neural-symbolic GPU kernel certification
> **Date:** January 21, 2026
> **Status:** Implemented in `xlog-cuda-tests`; historical specification, not a current release gate
> **Scope:** Neural-symbolic GPU kernel certification subset of the full certification suite

---

## Overview

The neural-symbolic integration work introduced GPU-accelerated circuit evaluation and gradient computation. This certification extends the existing CUDA infrastructure certification suite with **six GPU-focused categories** specifically for neural-symbolic kernel correctness and efficiency.

**Primary Goals:**
1. Verify all circuit evaluation kernels are correctly implemented
2. Ensure PTX kernels are robust across edge cases
3. Guarantee 100% GPU-native operation for training loops
4. Verify 0% CPU data transfer bottlenecks in hot paths

**New Categories:**
| Area | Name | Tests | Focus |
|------|------|-------|-------|
| Forward evaluation | Circuit Forward Kernel | 8 | `xgcf_forward_level` PTX correctness |
| Reverse-mode gradients | Circuit Backward Kernels | 12 | `xgcf_backward_level_*` gradient computation |
| Neural weight inputs | Weight Injection Patterns | 6 | GPU weight buffer management |
| Host-transfer discipline | Transfer Efficiency | 8 | 0% CPU bottleneck verification |
| Cached circuits | Circuit Cache GPU Integration | 6 | GpuXgcf reuse and external compiler avoidance |
| Kernel robustness | PTX Kernel Robustness | 10 | Large circuits, edge cases, stability |

**Total New Tests:** 50
**Total Suite (with existing CUDA infrastructure categories):** 200 tests (snapshot at time of writing)

**Current main note:** The certification suite has grown beyond this neural-symbolic subset (e.g., SAT/CDCL + device-count
categories). For the full current suite, see `docs/architecture/cuda-certification.md`.

---

## Target Kernel Architecture

### PTX Module: `kernels/circuit.ptx` (SM_70)

**Forward Pass Kernel:**
```
.visible .entry xgcf_forward_level(
    node_type, child_offsets, child_indices, lit,
    decision_var, decision_child_false, decision_child_true,
    level_nodes, level_range, var_log_true, var_log_false, values
)
```
- Evaluates XGCF circuit nodes level-by-level
- Computes log-space weighted model count (WMC)
- Node types: Const0(0), Const1(1), Lit(2), And(3), Or(4), Decision(5)

**Backward Pass Kernels:**
```
.visible .entry xgcf_backward_level_propagate(...)    // Adjoint propagation
.visible .entry xgcf_backward_level_decision_grad(...) // Decision node gradients
.visible .entry xgcf_backward_level_lit_grad(...)      // Literal gradients
```
- Implements reverse-mode automatic differentiation
- Gradients w.r.t. leaf log-weights for backprop through neural networks

### GPU Data Structures: `GpuXgcf` (crates/xlog-prob/src/gpu.rs)

**Circuit Structure (immutable, uploaded once):**
- `node_type: TrackedCudaSlice<u8>` - Node type enum
- `child_offsets/indices: TrackedCudaSlice<u32>` - Child pointers
- `lit: TrackedCudaSlice<i32>` - Literal variables
- `decision_*: TrackedCudaSlice<u32>` - Decision node structure
- `level_nodes/offsets` - Level-ordered execution schedule

**Working Buffers (reused per evaluation):**
- `var_log_true/false: TrackedCudaSlice<f64>` - Weight inputs
- `values: TrackedCudaSlice<f64>` - Forward pass results
- `adj: TrackedCudaSlice<f64>` - Adjoint buffer
- `grad_true/false: TrackedCudaSlice<f64>` - Gradient outputs

---

## Circuit Forward Kernel Correctness (8 tests)

Validates `xgcf_forward_level` PTX kernel produces correct log-space WMC values.

### Node Type Evaluation (4 tests)

| Test | Description | Circuit | Expected |
|------|-------------|---------|----------|
| `test_forward_const_nodes` | Const0/Const1 evaluation | `[Const0, Const1]` | values[0]=-inf, values[1]=0.0 |
| `test_forward_lit_nodes` | Positive/negative literals | `Lit(1), Lit(-1)` | log(p), log(1-p) respectively |
| `test_forward_and_node` | AND of two children | `And(Lit(1), Lit(2))` | log(p1) + log(p2) |
| `test_forward_or_node` | OR of two children | `Or(Lit(1), Lit(2))` | logsumexp(log(p1), log(p2)) |

### Decision Node Evaluation (2 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_forward_decision_basic` | Decision node with true/false branches | Correct WMC selection |
| `test_forward_decision_chain` | Chain of decision nodes | Correct propagation |

### Level-by-Level Execution (2 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_forward_multi_level` | 5-level circuit, dependencies | Values computed in correct order |
| `test_forward_parallel_same_level` | 1000 nodes at same level | Correct parallel execution |

---

## Circuit Backward Kernels Correctness (12 tests)

Validates `xgcf_backward_level_*` kernels compute correct gradients.

### Adjoint Propagation (4 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_backward_root_adjoint` | Root adjoint = 1.0 | adj[root] == 1.0 after init |
| `test_backward_and_propagate` | AND node adjoint flow | adj[child] += adj[parent] |
| `test_backward_or_propagate` | OR node adjoint flow | adj[child] += adj[parent] * softmax_weight |
| `test_backward_decision_propagate` | Decision node adjoint | Correct branch adjoint |

### Gradient Accumulation (4 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_backward_lit_grad` | Literal gradient computation | grad = adj * d(value)/d(weight) |
| `test_backward_decision_grad` | Decision gradient computation | Correct grad_true/grad_false |
| `test_backward_gradient_accumulation` | Multiple paths to same variable | Gradients sum correctly |
| `test_backward_negative_lit_grad` | Negative literal gradient | Sign flip in gradient |

### Numerical Verification (4 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_backward_vs_numerical_diff` | Compare to finite differences | |GPU grad - numerical| < 1e-6 |
| `test_backward_chain_rule` | Multi-level chain rule | Correct gradient chain |
| `test_backward_gradient_stability` | Near-zero/one probabilities | No NaN/Inf in gradients |
| `test_backward_log_space_accuracy` | Log-space gradient accuracy | Relative error < 1e-10 |

---

## Weight Injection Memory Patterns (6 tests)

Validates GPU weight buffer management for neural network integration.

### Weight Buffer Operations (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_weight_upload_integrity` | htod_sync_copy_into weights | Exact value preservation |
| `test_weight_buffer_reuse` | Same buffer, different weights | Buffer reused, no realloc |
| `test_weight_variable_mapping` | Variable index → buffer position | Correct DIMACS-to-buffer mapping |

### Weight Value Handling (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_weight_extreme_values` | log(1e-308), log(0.9999...) | No overflow/underflow |
| `test_weight_uniform_distribution` | All weights equal | Correct uniform probability |
| `test_weight_sparse_nonzero` | Most weights log(0) | Handles -inf correctly |

---

## Transfer Efficiency - 0% CPU Bottleneck Verification (8 tests)

Validates that GPU operations dominate, with minimal CPU-GPU transfer overhead.

### Transfer Size Analysis (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_transfer_weight_size` | Weight upload size | 2 * num_vars * 8 bytes only |
| `test_transfer_gradient_size` | Gradient download size | 2 * num_vars * 8 bytes only |
| `test_transfer_circuit_cached` | Circuit NOT re-uploaded on cache hit | 0 bytes circuit transfer |

### Transfer vs Compute Ratio (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_transfer_compute_ratio_mnist` | MNIST (20 vars) | Transfer < 5% of total time |
| `test_transfer_compute_ratio_large` | 1000 variables | Transfer < 10% of total time |
| `test_transfer_dominance_check` | Profile forward+backward | GPU compute > 90% of time |

### Hot Path Verification (2 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_training_loop_no_cpu_copy` | 100 iteration training loop | No intermediate CPU copies |
| `test_batch_eval_no_sync` | 32 queries batched | Single sync at end only |

---

## Circuit Cache GPU Integration (6 tests)

Validates GpuXgcf reuse and external compiler avoidance for cached queries.

### Cache Hit GPU Behavior (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_cache_hit_no_external_compile` | Same template query twice | External compiler invoked only once |
| `test_cache_hit_gpu_reuse` | GpuXgcf instance reused | Same GPU memory addresses |
| `test_cache_hit_speedup` | Measure cached vs uncached | Cached > 10x faster |

### Cache Key Correctness (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_cache_key_predicate` | Different predicate → miss | New circuit compiled |
| `test_cache_key_same_structure` | Same structure, different indices | Cache HIT |
| `test_cache_correctness` | Cached result == fresh result | Bit-identical outputs |

---

## PTX Kernel Robustness (10 tests)

Validates circuit.ptx handles edge cases, large circuits, and numerical extremes.

### Circuit Size Edge Cases (4 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_kernel_single_node` | 1-node circuit (just root) | Correct evaluation |
| `test_kernel_single_level` | All nodes at same level | Parallel execution works |
| `test_kernel_deep_circuit` | 100 levels, 1 node each | Sequential levels work |
| `test_kernel_large_circuit` | 100K nodes, 10K levels | Memory and execution correct |

### Variable Count Limits (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_kernel_max_variables` | 65536 variables | Buffer allocation succeeds |
| `test_kernel_sparse_variables` | Variables [1, 1000, 50000] | Non-contiguous works |
| `test_kernel_many_literals` | 10K literal nodes | All literals evaluated |

### Numerical Stability (3 tests)

| Test | Description | Validation |
|------|-------------|------------|
| `test_kernel_log_underflow` | log(1e-300) weights | No -inf propagation issues |
| `test_kernel_logsumexp_stability` | Extreme value differences | Stable logsumexp |
| `test_kernel_determinism` | Same inputs → same outputs | Bit-identical across runs |

---

## Implementation Notes

### Test Harness Extension

Add category modules under `crates/xlog-cuda-tests/src/categories/` for:

- forward kernel tests
- backward kernel tests
- weight buffer tests
- CPU bottleneck verification
- cache GPU integration
- edge cases and stability

Register those modules from `mod.rs` using the category-module names used by the CUDA certification harness.

### Test Context Extension

Extend `TestContext` with circuit evaluation methods:

```rust
impl TestContext {
    /// Create a test XGCF circuit with specified structure
    fn create_test_circuit(&self, spec: CircuitSpec) -> Result<GpuXgcf>;

    /// Evaluate circuit forward pass
    fn eval_forward(&self, circuit: &mut GpuXgcf, weights: &[(f64, f64)]) -> Result<f64>;

    /// Evaluate circuit with gradients
    fn eval_with_grads(&self, circuit: &mut GpuXgcf, weights: &[(f64, f64)])
        -> Result<(f64, Vec<f64>, Vec<f64>)>;

    /// Measure GPU transfer time
    fn measure_transfer_time<F: FnOnce()>(&self, op: F) -> Duration;

    /// Measure GPU compute time
    fn measure_compute_time<F: FnOnce()>(&self, op: F) -> Duration;
}
```

### Reference Implementations for Verification

```rust
/// CPU reference forward pass for verification
fn reference_forward(circuit: &Xgcf, weights: &[(f64, f64)]) -> f64 {
    // Recursive evaluation matching GPU kernel semantics
}

/// Numerical differentiation for gradient verification
fn numerical_gradient(
    circuit: &Xgcf,
    weights: &[(f64, f64)],
    var: usize,
    eps: f64
) -> (f64, f64) {
    let w_plus = weights.with_var(var, weights[var].0 + eps, weights[var].1);
    let w_minus = weights.with_var(var, weights[var].0 - eps, weights[var].1);
    let grad_true = (reference_forward(circuit, &w_plus) -
                    reference_forward(circuit, &w_minus)) / (2.0 * eps);
    // Similarly for grad_false
    (grad_true, grad_false)
}
```

---

## Test Execution

### Run Neural-Symbolic GPU Certification

```bash
# Full GPU kernel certification
cargo test -p xlog-cuda-tests --test gpu_certification --release -- --nocapture

# Quick smoke test for forward and backward kernels
cargo test -p xlog-cuda-tests --test gpu_smoke --release

# Transfer-efficiency category
cargo test -p xlog-cuda-tests --test gpu_certification transfer_efficiency --release
```

### Run Combined Suite

```bash
# Full combined certification (200 tests)
cargo test -p xlog-cuda-tests --test full_certification --release -- --nocapture
```

### Profiling for Transfer Efficiency

```bash
# With CUDA profiler
nsys profile cargo test -p xlog-cuda-tests --test gpu_certification transfer_efficiency --release
```

---

## Pass Criteria

### Category Pass Rates

| Category | Required Pass Rate | Critical Tests |
|----------|-------------------|----------------|
| Forward kernels | 100% | All |
| Backward kernels | 100% | All |
| Weight injection | 100% | All |
| Transfer efficiency | 100% | `test_transfer_dominance_check` |
| Circuit cache | 100% | `test_cache_hit_no_external_compile` |
| Kernel robustness | 100% | `test_kernel_determinism` |

### Neural-Symbolic Certification Gate

- [ ] All neural-symbolic GPU tests pass (50 tests)
- [ ] All existing CUDA infrastructure tests pass (150 tests)
- [ ] Transfer efficiency: GPU compute > 90% of forward+backward time
- [ ] Cache speedup: Cached queries > 10x faster than uncached
- [ ] No NaN/Inf in any gradient computation
- [ ] Bit-identical results across runs (determinism)
- [ ] Large circuit (100K nodes) completes without OOM

### Performance Benchmarks (Non-Blocking)

| Metric | Target | Notes |
|--------|--------|-------|
| Forward pass (10K nodes) | < 5ms | GPU kernel only |
| Backward pass (10K nodes) | < 10ms | 3 kernels |
| Weight upload (100 vars) | < 0.1ms | 1.6KB transfer |
| Gradient download (100 vars) | < 0.1ms | 1.6KB transfer |
| Cache hit speedup | > 30x | vs external compilation |

---

## Appendix: Circuit Test Generators

### Basic Circuit Structures

```rust
/// Create simple literal-only circuit
pub fn gen_literal_circuit(num_vars: u32) -> Xgcf {
    // Root OR of all positive literals
    Xgcf::or_of_literals((1..=num_vars).collect())
}

/// Create AND-OR tree
pub fn gen_and_or_tree(depth: u32, branching: u32) -> Xgcf {
    // Alternating AND/OR levels
}

/// Create decision tree circuit from external compiler output
pub fn gen_decision_tree(num_vars: u32, depth: u32) -> Xgcf {
    // Binary decision tree structure
}
```

### Weight Generators

```rust
/// Uniform weights
pub fn gen_uniform_weights(num_vars: usize, p: f64) -> Vec<(f64, f64)> {
    vec![(p.ln(), (1.0 - p).ln()); num_vars]
}

/// Random weights
pub fn gen_random_weights(num_vars: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_vars).map(|_| {
        let p: f64 = rng.gen();
        (p.ln(), (1.0 - p).ln())
    }).collect()
}

/// Extreme weights (near 0 and 1)
pub fn gen_extreme_weights(num_vars: usize) -> Vec<(f64, f64)> {
    (0..num_vars).map(|i| {
        if i % 2 == 0 {
            (1e-300_f64.ln(), (1.0 - 1e-300).ln())
        } else {
            ((1.0 - 1e-300).ln(), 1e-300_f64.ln())
        }
    }).collect()
}
```

---

## Change Log

| Date | Revision | Changes |
|------|----------|---------|
| 2026-01-21 | Initial | API-focused specification |
| 2026-01-21 | GPU kernel focus | Revised to neural-symbolic GPU kernel certification |
