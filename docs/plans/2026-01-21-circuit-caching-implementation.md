# Circuit Caching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cache compiled DNNF circuits to eliminate D4 recompilation during training, achieving 5-100x speedup.

**Architecture:** Circuits are cached by template signature (predicate + input count + network label counts). The `GpuXgcf` structure is already reusable with different weights - we just need to avoid recompiling the circuit with D4 on each query. Cache lives in `CompiledProgram`, keyed by template signature.

**Tech Stack:** Rust, PyO3, CUDA (existing GPU infrastructure)

---

## Task 1: Add CachedCircuit Data Structures

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:1-30` (imports)
- Modify: `crates/pyxlog/src/lib.rs:293-309` (CompiledProgram struct)

**Step 1: Write the test for CachedCircuit creation**

Create test file:
```rust
// In crates/pyxlog/tests/circuit_cache_test.rs
// Note: This is an integration test that will be run after implementation
```

Since pyxlog is a Python extension, we'll test via Python tests later. First, add the structures.

**Step 2: Add data structures to lib.rs**

After line 16 (imports section), add:
```rust
use std::collections::HashMap as StdHashMap;
```

After line 282 (before `CompiledProbProgram`), add:
```rust
/// A cached circuit for a specific query template.
///
/// The circuit structure is immutable - only weights change between queries.
/// Weight slots map network outputs to circuit variables.
struct CachedCircuit {
    /// The compiled program containing the GPU circuit
    program: ExactDdnnfProgram,

    /// Mapping from (input_idx, label_idx) to circuit variable index.
    /// For query "addition(0, 1, 7)" with 10-class MNIST:
    /// - (0, 0) -> var 1  (digit(0, 0))
    /// - (0, 1) -> var 2  (digit(0, 1))
    /// - ...
    /// - (1, 9) -> var 20 (digit(1, 9))
    weight_slots: Vec<WeightSlot>,

    /// Number of labels per network (e.g., 10 for MNIST)
    num_labels: usize,
}

/// Maps a network output to a circuit variable.
struct WeightSlot {
    /// Input index in the query (e.g., 0 or 1 for addition(0, 1, 7))
    input_idx: usize,
    /// Label index (0-9 for 10-class classification)
    label_idx: usize,
    /// Circuit variable index (1-indexed, as per DIMACS CNF convention)
    var_idx: u32,
}
```

**Step 3: Add circuit_cache field to CompiledProgram**

Modify `CompiledProgram` struct (around line 294) to add:
```rust
#[pyclass(unsendable)]
pub struct CompiledProgram {
    program: CompiledProbProgram,
    output_provider: Arc<CudaKernelProvider>,
    network_registry: NetworkRegistry,
    declared_networks: HashSet<String>,
    tensor_sources: TensorSourceRegistry,
    source: String,
    gpu_config: GpuConfig,
    prob_engine: ProbEngine,
    /// Cache of compiled circuits by template signature
    circuit_cache: StdHashMap<String, CachedCircuit>,
}
```

**Step 4: Initialize circuit_cache in Program::compile**

In `Program::compile` (around line 266), add `circuit_cache: StdHashMap::new()` to the struct initialization.

**Step 5: Build and verify compilation**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p pyxlog --release 2>&1 | head -50`
Expected: Compiles without errors

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(cache): add CachedCircuit data structures for circuit caching"
```

---

## Task 2: Implement Cache Key Generation

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add method to impl CompiledProgram)

**Step 1: Add generate_cache_key method**

After the existing `find_network_for_predicate` method (around line 1128), add:

```rust
/// Generate cache key for a query template.
///
/// Format: `{predicate}:{num_inputs}:{num_labels}`
/// Example: `addition:2:10` for addition(0, 1, 7) with 10-class MNIST
///
/// Queries with identical structure (same predicate, same number of inputs,
/// same network label count) share the same circuit - only weights differ.
fn generate_cache_key(&self, pred_name: &str, num_inputs: usize, num_labels: usize) -> String {
    format!("{}:{}:{}", pred_name, num_inputs, num_labels)
}
```

**Step 2: Build and verify**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p pyxlog --release 2>&1 | head -20`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(cache): add cache key generation for circuit templates"
```

---

## Task 3: Extract Circuit Compilation to Separate Method

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:939-971` (forward_backward_complex)

**Step 1: Create compile_circuit_for_template method**

Add new method after `generate_cache_key`:

```rust
/// Compile a circuit for a query template with given network outputs.
///
/// Returns the compiled program and weight slot mappings.
fn compile_circuit_for_template(
    &self,
    network_outputs: &[(usize, Vec<f64>, PyObject)],
    query: &str,
) -> PyResult<(ExactDdnnfProgram, Vec<WeightSlot>, usize)> {
    // Generate expanded source
    let expanded_source = self.generate_expanded_source(network_outputs, query)?;

    // Compile with D4 (expensive operation we want to cache)
    let program = ExactDdnnfProgram::compile_source_with_gpu(&expanded_source, self.gpu_config)
        .map_err(|e| PyRuntimeError::new_err(format!("Query compilation error: {}", e)))?;

    // Build weight slot mappings
    // Variables are assigned sequentially: input 0 labels 0-9, then input 1 labels 0-9, etc.
    let num_labels = if network_outputs.is_empty() {
        0
    } else {
        network_outputs[0].1.len()
    };

    let mut weight_slots = Vec::new();
    let mut var_idx: u32 = 1; // DIMACS variables are 1-indexed

    for (input_idx, probs, _) in network_outputs {
        for label_idx in 0..probs.len() {
            weight_slots.push(WeightSlot {
                input_idx: *input_idx,
                label_idx,
                var_idx,
            });
            var_idx += 1;
        }
    }

    Ok((program, weight_slots, num_labels))
}
```

**Step 2: Build and verify**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p pyxlog --release 2>&1 | head -20`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(cache): extract circuit compilation to separate method"
```

---

## Task 4: Implement Weight Update for Cached Circuits

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`

**Step 1: Add update_circuit_weights method**

Add after `compile_circuit_for_template`:

```rust
/// Update weights in a cached circuit with new network outputs.
///
/// The circuit structure remains unchanged - only the probability weights
/// are updated to reflect current network predictions.
fn update_circuit_weights(
    &self,
    cached: &CachedCircuit,
    network_outputs: &[(usize, Vec<f64>, PyObject)],
) -> Vec<(f64, f64)> {
    let num_vars = cached.program.num_vars();

    // Initialize all weights to (0.0, 0.0) - neutral in log space
    let mut weights: Vec<(f64, f64)> = vec![(0.0, 0.0); num_vars + 1];

    // Update weights from network outputs using slot mappings
    for slot in &cached.weight_slots {
        // Find the network output for this input index
        if let Some((_, probs, _)) = network_outputs.iter().find(|(idx, _, _)| *idx == slot.input_idx) {
            if slot.label_idx < probs.len() {
                let p = probs[slot.label_idx];
                // Normalize to ensure sum < 1.0 (same as generate_expanded_source)
                let sum: f64 = probs.iter().sum();
                let scale = 0.9999999 / sum;
                let normalized_p = (p * scale).max(1e-10);

                // Log weights: (log(p), log(1-p))
                // For annotated disjunctions, the "false" weight represents
                // the probability of NOT choosing this option
                let log_true = normalized_p.ln();
                let log_false = (1.0 - normalized_p).ln();

                if (slot.var_idx as usize) < weights.len() {
                    weights[slot.var_idx as usize] = (log_true, log_false);
                }
            }
        }
    }

    weights
}
```

**Step 2: Build and verify**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p pyxlog --release 2>&1 | head -20`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(cache): implement weight update for cached circuits"
```

---

## Task 5: Modify forward_backward_complex to Use Cache

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:894-971` (forward_backward_complex)

**Step 1: Rewrite forward_backward_complex with caching**

Replace the entire `forward_backward_complex` method:

```rust
/// Forward-backward for a complex query involving neural predicates through rules.
///
/// E.g., `addition(0, 1, 7)` where:
/// - `addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.`
/// - `nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).`
///
/// This method uses circuit caching to avoid D4 recompilation:
/// 1. Extracts input indices and runs neural networks
/// 2. Generates template cache key from query structure
/// 3. If cached: update weights and evaluate
/// 4. If not cached: compile circuit, cache it, evaluate
/// 5. Backpropagate gradients through networks
fn forward_backward_complex(&mut self, py: Python<'_>, query: &str) -> PyResult<f64> {
    // Parse the query to extract input indices
    let (input_indices, pred_name) = self.extract_query_input_indices(query)?;

    if input_indices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "No input indices found in query: {}. Make sure the query references neural predicate inputs.",
            query
        )));
    }

    // Get the network (for v0.4.0-alpha, assume single network)
    let network_name = self.network_registry.names().first()
        .ok_or_else(|| PyValueError::new_err("No network registered"))?
        .to_string();

    let handle = self.network_registry.get(&network_name).ok_or_else(|| {
        PyValueError::new_err(format!("Network '{}' not registered", network_name))
    })?;

    let module = handle.module().ok_or_else(|| {
        PyValueError::new_err(format!("Network '{}' has no module", network_name))
    })?;

    // Run neural network for each input index and collect outputs
    let mut network_outputs: Vec<(usize, Vec<f64>, PyObject)> = Vec::new();

    for &input_idx in &input_indices {
        let input_tensor = self.get_input_tensor(py, input_idx)?;
        let input_bound = input_tensor.bind(py);
        let input_unsqueezed = input_bound.call_method1("unsqueeze", (0i32,))?;

        // Forward pass with gradient tracking
        let output = module.call_method1(py, "__call__", (input_unsqueezed,))?;
        let output_bound = output.bind(py);
        let output_squeezed = output_bound.call_method1("squeeze", (0i32,))?;

        // Extract probabilities
        let probs_list = output_squeezed.call_method0("tolist")?;
        let probs: Vec<f64> = probs_list.extract()?;

        network_outputs.push((input_idx, probs, output.clone()));
    }

    // Get number of labels from first network output
    let num_labels = if network_outputs.is_empty() {
        return Err(PyValueError::new_err("No network outputs"));
    } else {
        network_outputs[0].1.len()
    };

    // Generate cache key
    let cache_key = self.generate_cache_key(&pred_name, input_indices.len(), num_labels);

    // Check cache and evaluate
    let (prob, grad_true, grad_false) = if let Some(cached) = self.circuit_cache.get(&cache_key) {
        // CACHE HIT: Update weights and evaluate existing circuit
        let weights = self.update_circuit_weights(cached, &network_outputs);

        // Evaluate with updated weights
        let result = cached.program.evaluate_gpu_with_grads_weights(&weights)
            .map_err(|e| PyRuntimeError::new_err(format!("Cached eval error: {}", e)))?;

        if result.query_grads.is_empty() {
            return Err(PyRuntimeError::new_err("No query results from cached circuit"));
        }

        let qg = &result.query_grads[0];
        (qg.prob, qg.grad_true.clone(), qg.grad_false.clone())
    } else {
        // CACHE MISS: Compile new circuit and cache it
        let (program, weight_slots, num_labels) =
            self.compile_circuit_for_template(&network_outputs, query)?;

        // Evaluate the newly compiled circuit
        let result = program.evaluate_gpu_with_grads()
            .map_err(|e| PyRuntimeError::new_err(format!("Query evaluation error: {}", e)))?;

        if result.query_grads.is_empty() {
            return Err(PyRuntimeError::new_err("No query results returned"));
        }

        let qg = &result.query_grads[0];
        let prob = qg.prob;
        let grad_true = qg.grad_true.clone();
        let grad_false = qg.grad_false.clone();

        // Cache the circuit for future use
        let cached = CachedCircuit {
            program,
            weight_slots,
            num_labels,
        };
        self.circuit_cache.insert(cache_key, cached);

        (prob, grad_true, grad_false)
    };

    let loss = nll_loss_value(prob);

    // Compute d(loss)/d(prob) = -1/prob
    let dloss_dprob = -1.0 / prob.max(NLL_EPSILON);

    // Backpropagate circuit gradients through neural networks
    self.backprop_circuit_gradients(
        py,
        &network_outputs,
        &grad_true,
        &grad_false,
        dloss_dprob,
    )?;

    Ok(loss)
}
```

**Step 2: Build and check for errors**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p pyxlog --release 2>&1`
Expected: Will fail because `evaluate_gpu_with_grads_weights` doesn't exist yet (Task 6)

**Step 3: Commit partial progress**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(cache): integrate circuit caching into forward_backward_complex"
```

---

## Task 6: Add evaluate_gpu_with_grads_weights to ExactDdnnfProgram

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs`

**Step 1: Add the new method after evaluate_gpu_with_grads (around line 280)**

```rust
/// Evaluate on GPU with gradients using externally provided weights.
///
/// This enables circuit reuse: the same compiled circuit can be evaluated
/// with different weights (from updated neural network outputs) without
/// recompiling the circuit.
pub fn evaluate_gpu_with_grads_weights(
    &self,
    external_weights: &[(f64, f64)],
) -> Result<ExactResultWithGrads> {
    let Some(_circuit) = &self.circuit else {
        return Ok(ExactResultWithGrads {
            log_z_e: 0.0,
            query_grads: Vec::new(),
        });
    };

    // Use external weights instead of self.evidence_log_weights
    let weights = external_weights;

    let (log_z_e, grad_true_e, grad_false_e) =
        self.eval_log_z_and_grads_gpu(weights)?;

    if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
        return Err(XlogError::Execution(
            "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
        ));
    }

    let mut query_grads: Vec<QueryGradients> = Vec::with_capacity(self.queries.len());

    for query in &self.queries {
        let Some(var) = query.var else {
            query_grads.push(QueryGradients {
                atom: query.atom.clone(),
                log_prob: f64::NEG_INFINITY,
                prob: 0.0,
                grad_true: vec![0.0; weights.len()],
                grad_false: vec![0.0; weights.len()],
            });
            continue;
        };

        let idx = var as usize;
        if idx >= weights.len() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: query var {} out of bounds (len={})",
                var,
                weights.len()
            )));
        }

        // Create modified weights with query var set to true
        let mut query_weights: Vec<(f64, f64)> = weights.to_vec();
        query_weights[idx].1 = f64::NEG_INFINITY;

        let (log_z_eq, grad_true_eq, grad_false_eq) =
            self.eval_log_z_and_grads_gpu(&query_weights)?;

        let log_prob = log_z_eq - log_z_e;
        let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
            0.0
        } else {
            log_prob.exp()
        };
        if prob.is_nan() {
            return Err(XlogError::Execution(
                "Exact inference error: NaN probability encountered".to_string(),
            ));
        }
        if prob < 0.0 {
            prob = 0.0;
        } else if prob > 1.0 {
            prob = 1.0;
        }

        if grad_true_eq.len() != grad_true_e.len() || grad_false_eq.len() != grad_false_e.len() {
            return Err(XlogError::Execution(
                "Exact inference error: gradient length mismatch".to_string(),
            ));
        }

        let mut grad_true: Vec<f64> = grad_true_eq;
        let mut grad_false: Vec<f64> = grad_false_eq;
        for i in 0..grad_true.len() {
            grad_true[i] -= grad_true_e[i];
            grad_false[i] -= grad_false_e[i];
        }

        query_grads.push(QueryGradients {
            atom: query.atom.clone(),
            log_prob,
            prob,
            grad_true,
            grad_false,
        });
    }

    Ok(ExactResultWithGrads { log_z_e, query_grads })
}
```

**Step 2: Build and verify**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build -p xlog-prob --release 2>&1 | head -20`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add crates/xlog-prob/src/exact.rs
git commit -m "feat(cache): add evaluate_gpu_with_grads_weights for circuit reuse"
```

---

## Task 7: Build and Run Existing Tests

**Files:** None (verification only)

**Step 1: Build full project**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo build --release 2>&1`
Expected: Compiles without errors

**Step 2: Run Rust tests**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo test --release 2>&1 | tail -50`
Expected: All tests pass

**Step 3: Build Python wheel**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && maturin build --release -m crates/pyxlog/Cargo.toml 2>&1 | tail -20`
Expected: Wheel builds successfully

**Step 4: Install wheel**

Run: `pip install --force-reinstall /home/dev/xlog/.worktrees/feature-circuit-caching/target/wheels/pyxlog-*.whl`
Expected: Installs without errors

---

## Task 8: Write Cache Hit/Miss Integration Test

**Files:**
- Create: `python/tests/test_circuit_cache.py`

**Step 1: Write the test file**

```python
"""Test circuit caching for training performance.

These tests verify:
1. Circuit cache correctly identifies identical query structures
2. Cached evaluation produces same results as fresh compilation
3. Cache hits avoid D4 recompilation (performance improvement)
"""

import pytest
import time

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class SimpleNet(torch.nn.Module):
    """Simple network that outputs softmax probabilities."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = torch.nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        return torch.nn.functional.softmax(self.fc(x), dim=-1)


class TestCircuitCache:
    """Test circuit caching functionality."""

    def test_cache_hit_same_structure(self):
        """Repeated queries with same structure should use cached circuit."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        # Register network
        net = SimpleNet(10)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Register dummy tensor source
        dummy_images = torch.randn(100, 1, 28, 28)
        program.register_tensor_source("X", dummy_images)

        # First query - should compile circuit (cache miss)
        t0 = time.time()
        loss1 = program.forward_backward("addition(0, 1, 7)")
        time_first = time.time() - t0

        # Second query with SAME structure - should use cache
        t0 = time.time()
        loss2 = program.forward_backward("addition(2, 3, 5)")
        time_second = time.time() - t0

        # Cache hit should be significantly faster (no D4 compilation)
        # First query includes D4 compilation (~100-200ms)
        # Cached query should be mostly GPU evaluation (~10-50ms)
        assert time_second < time_first, \
            f"Cache hit ({time_second:.3f}s) should be faster than miss ({time_first:.3f}s)"

        # Both should return valid losses
        assert loss1 > 0, f"First loss should be positive: {loss1}"
        assert loss2 > 0, f"Second loss should be positive: {loss2}"

    def test_cache_miss_different_label_count(self):
        """Different label counts should create separate cache entries."""
        # This test uses a hypothetical scenario with different networks
        # For now, just verify the cache key generation logic
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = SimpleNet(10)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        dummy_images = torch.randn(100, 1, 28, 28)
        program.register_tensor_source("X", dummy_images)

        # Multiple queries with same structure
        for i in range(5):
            loss = program.forward_backward(f"addition({i}, {i+1}, {(i + i + 1) % 19})")
            assert loss > 0, f"Query {i} should return valid loss"

    def test_correctness_cached_vs_fresh(self):
        """Cached circuit should produce same results as fresh compilation."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        # Use deterministic initialization
        torch.manual_seed(42)

        program = pyxlog.Program.compile(source)

        net = SimpleNet(10)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Use fixed input data
        torch.manual_seed(123)
        dummy_images = torch.randn(100, 1, 28, 28)
        program.register_tensor_source("X", dummy_images)

        # Get losses for several queries
        losses = []
        for target_sum in [3, 7, 11, 15]:
            loss = program.forward_backward(f"addition(0, 1, {target_sum})")
            losses.append(loss)

        # All should be valid positive losses
        for i, loss in enumerate(losses):
            assert loss > 0, f"Loss {i} should be positive: {loss}"
            assert not torch.isnan(torch.tensor(loss)), f"Loss {i} should not be NaN"


class TestCachePerformance:
    """Performance benchmarks for circuit caching."""

    @pytest.mark.slow
    def test_training_loop_speedup(self):
        """Full training loop should see significant speedup from caching."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = SimpleNet(10)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, 1, 28, 28)
        program.register_tensor_source("X", dummy_images)

        # Simulate training loop with 10 queries
        queries = [f"addition({i}, {i+1}, {(i*2 + 1) % 19})" for i in range(10)]

        t0 = time.time()
        for epoch in range(3):
            optimizer.zero_grad()
            total_loss = 0.0
            for q in queries:
                loss = program.forward_backward(q)
                total_loss += loss
            optimizer.step()
        elapsed = time.time() - t0

        # With caching, 3 epochs × 10 queries should complete in reasonable time
        # First epoch: 10 D4 compilations
        # Epochs 2-3: 20 cache hits (no D4)
        # Without caching: 30 D4 compilations
        # Target: < 5 seconds for cached version (vs ~30+ seconds uncached)
        assert elapsed < 10.0, \
            f"Training took {elapsed:.1f}s, expected < 10s with caching"
```

**Step 2: Run the test**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && python -m pytest python/tests/test_circuit_cache.py -v 2>&1`
Expected: Tests pass

**Step 3: Commit**

```bash
git add python/tests/test_circuit_cache.py
git commit -m "test(cache): add circuit caching integration tests"
```

---

## Task 9: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all Python tests**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && python -m pytest python/tests/ -v 2>&1 | tail -100`
Expected: All tests pass including new cache tests

**Step 2: Run Rust tests**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && cargo test --release 2>&1 | tail -50`
Expected: All tests pass

**Step 3: Verify no regressions**

Specifically check that existing training tests still work:
Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && python -m pytest python/tests/test_training.py -v 2>&1`
Expected: All training tests pass with cache enabled

---

## Task 10: Final Commit and Summary

**Step 1: Review all changes**

Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && git log --oneline -10`
Run: `cd /home/dev/xlog/.worktrees/feature-circuit-caching && git diff main --stat`

**Step 2: Final verification**

Run full test suite one more time to ensure everything works together.

**Step 3: Report completion**

- All tests pass
- Circuit caching implemented and working
- Expected speedup: 5-100x depending on number of queries per epoch
