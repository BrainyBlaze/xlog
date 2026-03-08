# P2a: Term Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add training-only embedding execution to XLOG: `register_embedding()`, `forward_embedding()`, cross-registration validation. No inference path (dot/cosine, grounded query) in v0.5.

**Architecture:** New `EmbeddingHandle` struct in xlog-neural, parallel embedding storage in `NetworkRegistry`, by-network form index in pyxlog for compile-time mixed-form rejection, `register_embedding` and `forward_embedding` as PyO3 methods on `CompiledProgram`.

**Tech Stack:** Rust (xlog-neural, pyxlog), PyO3, PyTorch (nn.Embedding, torch.Tensor)

**Design doc:** `docs/plans/2026-03-08-p2a-term-embeddings-design.md`

**Scope boundary:** `forward_embedding()` is the ONLY embedding execution API. Do NOT add dot/cosine evaluation, embedding cache, or query() integration. Those are deferred to v0.5.1+.

---

### Task 1: Add `EmbeddingHandle` struct to xlog-neural

**Files:**
- Modify: `crates/xlog-neural/src/handle.rs` (insert after line 183, before `impl Default for NetworkHandle`)
- Modify: `crates/xlog-neural/src/lib.rs` (add re-export at line 38)

**Step 1: Write the `EmbeddingHandle` struct**

Add after `NetworkHandle` impl blocks, before `impl Default for NetworkHandle`:

```rust
/// Handle to a registered embedding module.
///
/// Wraps either a trainable `nn.Embedding` or a frozen `torch.Tensor`.
/// Created via `CompiledProgram.register_embedding()` in Python.
#[derive(Debug)]
pub struct EmbeddingHandle {
    /// Unique name matching the nn() declaration
    pub name: String,

    /// The PyTorch nn.Embedding or tensor
    #[cfg(feature = "python")]
    pub module: Option<PyObject>,

    /// Whether gradients flow through this embedding
    pub trainable: bool,

    /// Embedding vector dimension (second axis of weight matrix)
    pub dim: usize,

    /// Number of embedding entries (first axis of weight matrix)
    pub vocab_size: usize,
}

impl EmbeddingHandle {
    /// Create a new embedding handle.
    pub fn new(name: String, trainable: bool, dim: usize, vocab_size: usize) -> Self {
        Self {
            name,
            #[cfg(feature = "python")]
            module: None,
            trainable,
            dim,
            vocab_size,
        }
    }

    /// Check if the PyTorch module/tensor has been set.
    #[cfg(feature = "python")]
    pub fn has_module(&self) -> bool {
        self.module.is_some()
    }

    #[cfg(not(feature = "python"))]
    pub fn has_module(&self) -> bool {
        false
    }

    /// Set the PyTorch module/tensor.
    #[cfg(feature = "python")]
    pub fn set_module(&mut self, module: PyObject) {
        self.module = Some(module);
    }

    /// Get a reference to the PyTorch module/tensor.
    #[cfg(feature = "python")]
    pub fn module(&self) -> Option<&PyObject> {
        self.module.as_ref()
    }
}
```

**Step 2: Add re-export in lib.rs**

In `crates/xlog-neural/src/lib.rs`, change line 38 from:
```rust
pub use handle::NetworkHandle;
```
to:
```rust
pub use handle::{EmbeddingHandle, NetworkHandle};
```

**Step 3: Write unit test**

Add to the `#[cfg(test)] mod tests` block in `handle.rs`:

```rust
#[test]
fn test_embedding_handle_new() {
    let handle = EmbeddingHandle::new("test_embed".to_string(), true, 64, 1000);
    assert_eq!(handle.name, "test_embed");
    assert!(handle.trainable);
    assert_eq!(handle.dim, 64);
    assert_eq!(handle.vocab_size, 1000);
    assert!(!handle.has_module());
}

#[test]
fn test_embedding_handle_frozen() {
    let handle = EmbeddingHandle::new("frozen".to_string(), false, 128, 500);
    assert!(!handle.trainable);
    assert_eq!(handle.dim, 128);
    assert_eq!(handle.vocab_size, 500);
}
```

**Step 4: Run tests**

Run: `cargo test -p xlog-neural --release`
Expected: All existing tests pass + 2 new tests pass.

**Step 5: Commit**

```
git add crates/xlog-neural/src/handle.rs crates/xlog-neural/src/lib.rs
git commit -m "feat(neural): add EmbeddingHandle struct for embedding registration"
```

---

### Task 2: Add embedding storage to `NetworkRegistry`

**Files:**
- Modify: `crates/xlog-neural/src/registry.rs` (add field + methods to `NetworkRegistry`)

**Step 1: Write failing tests**

Add to the `#[cfg(test)] mod tests` block in `registry.rs`:

```rust
use crate::handle::EmbeddingHandle;

#[test]
fn test_registry_embedding_register_get() {
    let mut registry = NetworkRegistry::new();
    let handle = EmbeddingHandle::new("embed1".to_string(), true, 64, 100);
    registry.register_embedding(handle);

    assert!(registry.contains_embedding("embed1"));
    assert!(!registry.contains_embedding("nonexistent"));

    let h = registry.get_embedding("embed1").unwrap();
    assert_eq!(h.dim, 64);
    assert_eq!(h.vocab_size, 100);
}

#[test]
fn test_registry_embedding_get_mut() {
    let mut registry = NetworkRegistry::new();
    let handle = EmbeddingHandle::new("embed1".to_string(), true, 64, 100);
    registry.register_embedding(handle);

    let h = registry.get_embedding_mut("embed1").unwrap();
    h.trainable = false;
    assert!(!registry.get_embedding("embed1").unwrap().trainable);
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p xlog-neural --release`
Expected: FAIL — `register_embedding`, `contains_embedding`, `get_embedding`, `get_embedding_mut` do not exist.

**Step 3: Add embedding storage to `NetworkRegistry`**

In `registry.rs`, add to the `NetworkRegistry` struct (line 118):

```rust
pub struct NetworkRegistry {
    /// Map from network name to handle
    networks: HashMap<String, NetworkHandle>,
    /// Map from embedding name to handle
    embeddings: HashMap<String, EmbeddingHandle>,
}
```

Update `new()` (line 124):
```rust
pub fn new() -> Self {
    Self {
        networks: HashMap::new(),
        embeddings: HashMap::new(),
    }
}
```

Add import at top of file:
```rust
use crate::handle::EmbeddingHandle;
```

Add methods after existing `iter_mut` (before `impl Default`):

```rust
/// Register an embedding with the given handle.
pub fn register_embedding(&mut self, handle: EmbeddingHandle) {
    self.embeddings.insert(handle.name.clone(), handle);
}

/// Get a reference to an embedding handle by name.
pub fn get_embedding(&self, name: &str) -> Option<&EmbeddingHandle> {
    self.embeddings.get(name)
}

/// Get a mutable reference to an embedding handle by name.
pub fn get_embedding_mut(&mut self, name: &str) -> Option<&mut EmbeddingHandle> {
    self.embeddings.get_mut(name)
}

/// Check if an embedding is registered.
pub fn contains_embedding(&self, name: &str) -> bool {
    self.embeddings.contains_key(name)
}
```

**Step 4: Run tests**

Run: `cargo test -p xlog-neural --release`
Expected: All tests pass including new embedding tests.

**Step 5: Commit**

```
git add crates/xlog-neural/src/registry.rs
git commit -m "feat(neural): add embedding storage to NetworkRegistry"
```

---

### Task 3: Build by-network form index at compile time

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (CompiledProgram struct + compile path)

**Context:** Currently `CompiledProgram` has `declared_networks: HashSet<String>` (lib.rs:596) built at compile time (lib.rs:478-482). We add a parallel `HashMap<String, bool>` mapping network name → is_embedding, with mixed-form rejection.

**Step 1: Add `declared_network_forms` field to `CompiledProgram`**

In `CompiledProgram` struct (lib.rs:588), add after `declared_networks`:

```rust
    /// Map from network name to form: true = embedding, false = classification
    declared_network_forms: HashMap<String, bool>,
```

**Step 2: Build the form index during compilation**

In the `compile` function, after the `declared_networks` construction (lib.rs:478-482), add:

```rust
        // Build by-network form index: network name -> is_embedding
        let mut declared_network_forms: HashMap<String, bool> = HashMap::new();
        for np in &ast.neural_predicates {
            let is_embedding = np.labels.is_none();
            match declared_network_forms.get(&np.network) {
                Some(&existing_form) if existing_form != is_embedding => {
                    return Err(PyValueError::new_err(format!(
                        "network '{}' is declared as both classification and embedding; \
                         each network name must have a single form",
                        np.network
                    )));
                }
                _ => {
                    declared_network_forms.insert(np.network.clone(), is_embedding);
                }
            }
        }
```

**Step 3: Store in the struct literal**

In the `CompiledProgram` struct literal (lib.rs:504), add:

```rust
            declared_network_forms,
```

**Step 4: Run Rust tests**

Run: `cargo test -p pyxlog --release`
Expected: All existing tests pass (no mixed-form declarations in existing tests).

**Step 5: Commit**

```
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): add by-network form index with mixed-form rejection"
```

---

### Task 4: Add `register_embedding` and cross-registration guard

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `register_embedding` method, modify `register_network`)
- Modify: `crates/xlog-neural/src/lib.rs` (ensure `EmbeddingHandle` import available)

**Context:** `register_network` is at lib.rs:808. It validates against `declared_networks` (a `HashSet<String>`). We add `register_embedding` nearby and add a cross-registration guard to `register_network`.

**Step 1: Update imports in pyxlog lib.rs**

At line 20, change:
```rust
use xlog_neural::{NetworkConfig, NetworkRegistry, TensorMetadata, TensorSourceRegistry};
```
to:
```rust
use xlog_neural::{EmbeddingHandle, NetworkConfig, NetworkRegistry, TensorMetadata, TensorSourceRegistry};
```

**Step 2: Add cross-registration guard to `register_network`**

In `register_network` (lib.rs:808), after the existing `declared_networks` check (line 821-827), add:

```rust
        // Cross-registration: reject if this network is declared as an embedding
        if let Some(&true) = self.declared_network_forms.get(&name) {
            return Err(PyValueError::new_err(format!(
                "declaration '{}' is an embedding; use register_embedding()",
                name
            )));
        }
```

**Step 3: Add `register_embedding` method**

Add in the `#[pymethods] impl CompiledProgram` block, after `register_network`:

```rust
    /// Register an embedding module for a declared embedding predicate.
    ///
    /// # Arguments
    ///
    /// * `name` - Must match an nn() declaration with no label list
    /// * `module_or_tensor` - PyTorch nn.Embedding or 2D torch.Tensor
    /// * `trainable` - Whether gradients flow; must be false for raw tensors
    #[pyo3(signature = (name, module_or_tensor, trainable=true))]
    fn register_embedding(
        &mut self,
        py: Python<'_>,
        name: String,
        module_or_tensor: PyObject,
        trainable: bool,
    ) -> PyResult<()> {
        // Validate network name exists
        if !self.declared_networks.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "Network '{}' not declared in program. Declared networks: {:?}",
                name,
                self.declared_networks.iter().collect::<Vec<_>>()
            )));
        }

        // Cross-registration: reject if this network is declared as classification
        match self.declared_network_forms.get(&name) {
            Some(&false) => {
                return Err(PyValueError::new_err(format!(
                    "declaration '{}' is a classification network; use register_network()",
                    name
                )));
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "Network '{}' not found in form index",
                    name
                )));
            }
            _ => {} // is_embedding = true, correct
        }

        let torch = py.import_bound("torch")?;
        let nn = py.import_bound("torch.nn")?;
        let obj = module_or_tensor.bind(py);

        // Detect payload type and extract shape
        let embedding_cls = nn.getattr("Embedding")?;
        let is_nn_embedding = obj.is_instance(&embedding_cls)?;

        let (vocab_size, dim) = if is_nn_embedding {
            // nn.Embedding: read .weight shape
            let weight = obj.getattr("weight")?;
            let shape = weight.getattr("shape")?;
            let vs: usize = shape.get_item(0)?.extract()?;
            let d: usize = shape.get_item(1)?.extract()?;

            // Validate dtype is float
            let dtype = weight.getattr("dtype")?;
            let float32 = torch.getattr("float32")?;
            let float64 = torch.getattr("float64")?;
            let float16 = torch.getattr("float16")?;
            let bfloat16 = torch.getattr("bfloat16")?;
            if !dtype.eq(&float32)? && !dtype.eq(&float64)?
                && !dtype.eq(&float16)? && !dtype.eq(&bfloat16)? {
                return Err(PyValueError::new_err(
                    "nn.Embedding weight must have float dtype"
                ));
            }

            (vs, d)
        } else {
            // Raw torch.Tensor: must be frozen
            let tensor_cls = torch.getattr("Tensor")?;
            if !obj.is_instance(&tensor_cls)? {
                return Err(PyValueError::new_err(
                    "module_or_tensor must be nn.Embedding or torch.Tensor"
                ));
            }

            if trainable {
                return Err(PyValueError::new_err(
                    "trainable=True requires nn.Embedding; raw torch.Tensor is always frozen"
                ));
            }

            // Validate rank == 2
            let ndim: usize = obj.getattr("ndim")?.extract()?;
            if ndim != 2 {
                return Err(PyValueError::new_err(format!(
                    "embedding tensor must be 2D [vocab_size, dim], got {}D",
                    ndim
                )));
            }

            let shape = obj.getattr("shape")?;
            let vs: usize = shape.get_item(0)?.extract()?;
            let d: usize = shape.get_item(1)?.extract()?;

            // Validate dtype is float
            let dtype = obj.getattr("dtype")?;
            let float32 = torch.getattr("float32")?;
            let float64 = torch.getattr("float64")?;
            let float16 = torch.getattr("float16")?;
            let bfloat16 = torch.getattr("bfloat16")?;
            if !dtype.eq(&float32)? && !dtype.eq(&float64)?
                && !dtype.eq(&float16)? && !dtype.eq(&bfloat16)? {
                return Err(PyValueError::new_err(
                    "embedding tensor must have float dtype"
                ));
            }

            (vs, d)
        };

        let mut handle = EmbeddingHandle::new(name.clone(), trainable, dim, vocab_size);
        handle.set_module(module_or_tensor);
        self.network_registry.register_embedding(handle);

        Ok(())
    }
```

**Step 4: Run Rust compilation**

Run: `cargo build -p pyxlog --release`
Expected: Compiles without errors.

**Step 5: Commit**

```
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): add register_embedding with cross-registration validation"
```

---

### Task 5: Add `forward_embedding` method

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `forward_embedding` method)

**Context:** This is a standalone method, NOT inside the `forward_backward_*` pipeline. It looks up the embedding handle, calls the module with the given integer IDs, and returns the raw torch.Tensor.

**Step 1: Add `forward_embedding` method**

Add in the `#[pymethods] impl CompiledProgram` block, after `register_embedding`:

```rust
    /// Look up embedding vectors by integer IDs.
    ///
    /// Returns a batched torch.Tensor with shape [len(ids), dim].
    /// For nn.Embedding: tensor has autograd graph (grad-enabled).
    /// For frozen torch.Tensor: tensor has requires_grad=False.
    ///
    /// This is the only gradient-carrying embedding API in v0.5.
    fn forward_embedding(
        &self,
        py: Python<'_>,
        name: String,
        ids: Vec<i64>,
    ) -> PyResult<PyObject> {
        let handle = self.network_registry.get_embedding(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Embedding '{}' not registered. Did you call register_embedding()?",
                name
            ))
        })?;

        let module = handle.module().ok_or_else(|| {
            PyValueError::new_err(format!("Embedding '{}' has no module", name))
        })?;

        let torch = py.import_bound("torch")?;

        if handle.trainable {
            // nn.Embedding: call module(ids_tensor)
            let ids_tensor = torch.call_method1(
                "tensor",
                (ids, torch.getattr("long")?),
            )?;
            let result = module.call_method1(py, "__call__", (ids_tensor,))?;
            Ok(result)
        } else {
            // Frozen tensor: index directly
            let ids_tensor = torch.call_method1(
                "tensor",
                (ids, torch.getattr("long")?),
            )?;
            let result = module.call_method1(py, "__getitem__", (ids_tensor,))?;
            Ok(result)
        }
    }
```

**Step 2: Run Rust compilation**

Run: `cargo build -p pyxlog --release`
Expected: Compiles without errors.

**Step 3: Commit**

```
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): add forward_embedding for batched tensor lookup"
```

---

### Task 6: Python tests

**Files:**
- Create: `python/tests/test_embeddings.py`

**Context:** 5 tests per the design doc. Requires `maturin develop` before running. All tests use embedding declarations (`nn()` with no label list).

**Step 1: Build the Python package**

Run: `cd /home/dev/projects/xlog && maturin develop --release`
Expected: Build succeeds, pyxlog installed in .venv.

**Step 2: Write all 5 tests**

Create `python/tests/test_embeddings.py`:

```python
"""Tests for P2a term embedding registration and forward_embedding API."""

import pytest
import torch
import pyxlog


EMBEDDING_SOURCE = """
    nn(entity_embed, [X], E) :: embed(X, E).
"""

CLASSIFICATION_SOURCE = """
    nn(classifier, [X], Y, [0, 1, 2]) :: classify(X, Y).
"""

BOTH_SOURCE = """
    nn(entity_embed, [X], E) :: embed(X, E).
    nn(classifier, [X], Y, [0, 1, 2]) :: classify(X, Y).
"""


class TestRegisterEmbeddingNnEmbedding:
    """Test 1: register_embedding with nn.Embedding produces correct vectors."""

    def test_forward_embedding_shape_and_values(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        vocab_size, dim = 10, 8
        embedding = torch.nn.Embedding(vocab_size, dim)

        program.register_embedding("entity_embed", embedding, trainable=True)

        # Look up 3 entities
        result = program.forward_embedding("entity_embed", [0, 3, 7])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, dim)

        # Verify values match direct nn.Embedding call
        expected = embedding(torch.tensor([0, 3, 7]))
        assert torch.allclose(result, expected)


class TestRegisterEmbeddingFrozenTensor:
    """Test 2: register_embedding with frozen torch.Tensor."""

    def test_frozen_tensor_correct_values(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        program.register_embedding("entity_embed", weights, trainable=False)

        result = program.forward_embedding("entity_embed", [2, 5])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 8)
        assert torch.allclose(result, weights[[2, 5]])

    def test_trainable_true_with_tensor_rejected(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        with pytest.raises(ValueError, match="trainable=True requires nn.Embedding"):
            program.register_embedding("entity_embed", weights, trainable=True)


class TestCrossRegistrationErrors:
    """Test 3: cross-registration errors in both directions."""

    def test_embedding_decl_reject_register_network(self):
        """Embedding declaration + register_network -> error."""
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        net = torch.nn.Embedding(10, 8)
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError, match="is an embedding.*register_embedding"):
            program.register_network("entity_embed", net, optimizer)

    def test_classification_decl_reject_register_embedding(self):
        """Classification declaration + register_embedding -> error."""
        program = pyxlog.Program.compile(CLASSIFICATION_SOURCE)

        embedding = torch.nn.Embedding(10, 8)

        with pytest.raises(ValueError, match="is a classification.*register_network"):
            program.register_embedding("classifier", embedding, trainable=True)


class TestGradientFlow:
    """Test 4: gradient flow through forward_embedding."""

    def test_backward_updates_embedding_weights(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        embedding = torch.nn.Embedding(10, 8)
        optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)

        program.register_embedding("entity_embed", embedding, trainable=True)

        # Save original weights
        original_weights = embedding.weight.data.clone()

        # Forward
        result = program.forward_embedding("entity_embed", [0, 1])

        # External loss
        target = torch.randn(2, 8)
        loss = torch.nn.functional.mse_loss(result, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights must have changed
        assert not torch.equal(embedding.weight.data, original_weights)


class TestFrozenOutputNonTrainable:
    """Test 5: frozen embedding output has no gradient path."""

    def test_frozen_tensor_requires_grad_false(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        program.register_embedding("entity_embed", weights, trainable=False)

        result = program.forward_embedding("entity_embed", [0, 1])
        assert not result.requires_grad
```

**Step 3: Run Python tests**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_embeddings.py -v`
Expected: All 6 test cases pass (5 test functions, `TestRegisterEmbeddingFrozenTensor` has 2).

**Step 4: Commit**

```
git add python/tests/test_embeddings.py
git commit -m "test(python): add P2a embedding registration and forward_embedding tests"
```

---

### Task 7: Update existing test and regression check

**Files:**
- Modify: `python/tests/test_network_registry.py` (update `test_embedding_network`)

**Context:** The existing `test_embedding_network` (test_network_registry.py:164) registers an embedding declaration via `register_network`. With the cross-registration guard, this test will now fail. Update it to use `register_embedding`.

**Step 1: Update the test**

In `python/tests/test_network_registry.py`, replace `test_embedding_network` (line 164-175):

```python
    def test_embedding_network(self):
        """Test registering an embedding network (no labels)."""
        program = pyxlog.Program.compile("""
            nn(encoder, [X], Embedding) :: encode(X, Embedding).
        """)

        embedding = torch.nn.Embedding(100, 128)
        program.register_embedding("encoder", embedding, trainable=True)

        # Verify it is registered (not in network_names, which is classification only)
        result = program.forward_embedding("encoder", [0])
        assert result.shape == (1, 128)
```

**Step 2: Run the updated test file**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_network_registry.py -v`
Expected: All tests pass including updated `test_embedding_network`.

**Step 3: Run full regression suite**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: All Rust tests pass.

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/ -v`
Expected: All Python tests pass (including new test_embeddings.py).

**Step 4: Commit**

```
git add python/tests/test_network_registry.py
git commit -m "test(python): update test_embedding_network for cross-registration guard"
```

---

### Task 8: Documentation and changelog

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/plans/2026-03-08-p2a-term-embeddings-design.md` (status → Implemented)

**Step 1: Add changelog entries**

Add under `[Unreleased]` in `CHANGELOG.md`:

```markdown
### Added
- **P2a: Term Embeddings (training-only)** — `register_embedding()` for
  `nn.Embedding` (trainable) and `torch.Tensor` (frozen) payloads.
  `forward_embedding(name, ids)` returns batched tensors with autograd
  support. Cross-registration validation: embedding declarations reject
  `register_network()` and vice versa. Compile-time mixed-form rejection
  for network names.
```

**Step 2: Update ROADMAP**

In `docs/ROADMAP.md`, update the v0.5.0 entry (line 728) to reflect P2a completion, and update the current milestone line.

**Step 3: Update design doc status**

In `docs/plans/2026-03-08-p2a-term-embeddings-design.md`, change:
```
**Status:** Approved (revised)
```
to:
```
**Status:** Implemented (2026-03-08, branch feat/p2a-term-embeddings)
```

**Step 4: Commit**

```
git add CHANGELOG.md docs/ROADMAP.md docs/plans/2026-03-08-p2a-term-embeddings-design.md
git commit -m "docs: record P2a term embeddings in changelog, roadmap, and design doc"
```
