# P2b: Extended Training Controls — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-network learning rate access, gradient clipping, early stopping, and per-network scheduler stepping to the neural training API.

**Architecture:** All five features are additive — new methods on `CompiledProgram` and new optional parameters on `train_model`/`train_model_tensor`. The Rust code calls into PyTorch via `PyObject::call_method` / `getattr` / `setattr`. No kernel or GPU changes. The training loop in `train_epoch_internal` gains a `max_grad_norm` parameter; the epoch loop in `train_model` gains `val_queries`/`patience` for early stopping.

**Tech Stack:** Rust (PyO3), Python (PyTorch `torch.nn.utils.clip_grad_norm_`, `optimizer.param_groups`)

**Design doc:** `docs/plans/2026-03-05-v050-execution-design.md` §P2b (lines 211–236)

---

## Context for the implementer

### Codebase orientation

- **Main file:** `crates/pyxlog/src/lib.rs` (~66K lines). All Python-visible training methods live here.
- **Network storage:** `CompiledProgram.network_registry: NetworkRegistry` (defined in `crates/xlog-neural/src/registry.rs`). Each entry is a `NetworkHandle` (defined in `crates/xlog-neural/src/handle.rs`) holding `module: Option<PyObject>`, `optimizer: Option<PyObject>`, `scheduler: Option<PyObject>`.
- **Existing methods on `CompiledProgram`** (in `#[pymethods]` block starting at line 619):
  - `zero_grad()` (line 1133) — iterates all networks, calls `optimizer.zero_grad()`
  - `optimizer_step()` (line 1148) — iterates all networks, calls `optimizer.step()`
  - `scheduler_step()` (line 1163) — iterates all networks, calls `scheduler.step()`
  - `evaluate_loss(queries)` (line 1211) — mean NLL over queries (no gradient)
- **Training loop internals:**
  - `train_epoch_internal()` (line 1348) — per-batch: `zero_grad → forward_backward (each query) → optimizer_step`
  - `train_model()` (line 3495, free function `#[pyfunction]`) — epoch loop calling `train_epoch_internal`, shuffles queries, records history
  - `train_model_tensor()` (line 3544) — identical but uses `train_epoch_tensor_internal`
- **Tests:** `python/tests/test_training.py` — 13 tests across 6 classes. Pattern: `pyxlog.Program.compile(…)`, register network with `SimpleNet`, train, assert.

### How to run tests

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /home/dev/projects/xlog

# Build pyxlog (must rebuild after Rust changes)
cd crates/pyxlog && maturin develop --release 2>&1 | tail -5 && cd ../..

# Run training tests
.venv/bin/python -m pytest python/tests/test_training.py -v

# Run a single test
.venv/bin/python -m pytest python/tests/test_training.py::TestGetSetLr::test_get_lr -v
```

### PyO3 patterns used in this file

Calling Python methods on a `PyObject`:
```rust
// No args:
optimizer.call_method0(py, "zero_grad")?;
// With args:
let torch_nn_utils = py.import("torch.nn.utils")?;
torch_nn_utils.call_method1("clip_grad_norm_", (params, max_norm))?;
// Get attribute:
let param_groups = optimizer.getattr(py, "param_groups")?;
// Index into list:
let group0 = param_groups.call_method1(py, "__getitem__", (0i32,))?;
// Get dict item:
let lr = group0.call_method1(py, "__getitem__", ("lr",))?;
let lr_f64: f64 = lr.extract(py)?;
// Set dict item:
group0.call_method(py, "__setitem__", ("lr", new_lr), None)?;
```

---

## Task 1: `get_lr(network_name)` — read learning rate

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:1163` (add method after `scheduler_step`)
- Test: `python/tests/test_training.py`

### Step 1: Write the failing test

Add a new test class at the end of `python/tests/test_training.py`:

```python
class TestGetSetLr:
    """Tests for get_lr() and set_lr() methods."""

    def test_get_lr(self):
        """get_lr returns the optimizer's current learning rate."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.042)
        program.register_network("test_net", net, optimizer)

        lr = program.get_lr("test_net")
        assert lr == pytest.approx(0.042)

    def test_get_lr_unknown_network_raises(self):
        """get_lr raises ValueError for an unregistered network name."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        with pytest.raises(ValueError):
            program.get_lr("nonexistent")
```

### Step 2: Run test to verify it fails

Run: `.venv/bin/python -m pytest python/tests/test_training.py::TestGetSetLr::test_get_lr -v`
Expected: FAIL — `AttributeError: 'Program' object has no attribute 'get_lr'`

### Step 3: Write minimal implementation

In `crates/pyxlog/src/lib.rs`, after `scheduler_step` (line 1172), add:

```rust
    /// Get the current learning rate for a registered network.
    ///
    /// Reads `optimizer.param_groups[0]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    fn get_lr(&self, py: Python<'_>, network_name: &str) -> PyResult<f64> {
        let handle = self
            .network_registry
            .get(network_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No network registered with name '{network_name}'"
                ))
            })?;
        let optimizer = handle
            .optimizer()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Network '{network_name}' has no optimizer"
                ))
            })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let group0 = param_groups.call_method1(py, "__getitem__", (0i32,))?;
        let lr = group0.call_method1(py, "__getitem__", ("lr",))?;
        lr.extract(py)
    }
```

### Step 4: Rebuild and run tests

```bash
cd crates/pyxlog && maturin develop --release 2>&1 | tail -3 && cd ../..
.venv/bin/python -m pytest python/tests/test_training.py::TestGetSetLr -v
```

Expected: 2/2 PASS

### Step 5: Commit

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_training.py
git commit -m "feat(training): add get_lr(network_name) for reading optimizer LR"
```

---

## Task 2: `set_lr(network_name, lr)` — write learning rate

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add method after `get_lr`)
- Test: `python/tests/test_training.py`

### Step 1: Write the failing test

Add to the `TestGetSetLr` class:

```python
    def test_set_lr(self):
        """set_lr updates the optimizer's learning rate for all param groups."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        program.set_lr("test_net", 0.123)

        # Verify via get_lr
        assert program.get_lr("test_net") == pytest.approx(0.123)
        # Verify the Python optimizer object is updated too
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.123)

    def test_set_lr_unknown_network_raises(self):
        """set_lr raises ValueError for an unregistered network name."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        with pytest.raises(ValueError):
            program.set_lr("nonexistent", 0.1)
```

### Step 2: Run test to verify it fails

Run: `.venv/bin/python -m pytest python/tests/test_training.py::TestGetSetLr::test_set_lr -v`
Expected: FAIL — `AttributeError: 'Program' object has no attribute 'set_lr'`

### Step 3: Write minimal implementation

In `crates/pyxlog/src/lib.rs`, after `get_lr`, add:

```rust
    /// Set the learning rate for a registered network.
    ///
    /// Writes to all `optimizer.param_groups[i]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    /// * `lr` - New learning rate value
    fn set_lr(&self, py: Python<'_>, network_name: &str, lr: f64) -> PyResult<()> {
        let handle = self
            .network_registry
            .get(network_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No network registered with name '{network_name}'"
                ))
            })?;
        let optimizer = handle
            .optimizer()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Network '{network_name}' has no optimizer"
                ))
            })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let num_groups: usize = param_groups.call_method0(py, "__len__")?.extract(py)?;
        for i in 0..num_groups {
            let group = param_groups.call_method1(py, "__getitem__", (i as i32,))?;
            group.call_method(py, "__setitem__", ("lr", lr), None)?;
        }
        Ok(())
    }
```

### Step 4: Rebuild and run tests

```bash
cd crates/pyxlog && maturin develop --release 2>&1 | tail -3 && cd ../..
.venv/bin/python -m pytest python/tests/test_training.py::TestGetSetLr -v
```

Expected: 4/4 PASS (all TestGetSetLr tests)

### Step 5: Commit

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_training.py
git commit -m "feat(training): add set_lr(network_name, lr) for writing optimizer LR"
```

---

## Task 3: Per-network `scheduler_step(network_name=None)`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:1163` (change existing `scheduler_step` signature)
- Test: `python/tests/test_training.py`

### Step 1: Write the failing test

Add a new test class:

```python
class TestPerNetworkScheduler:
    """Tests for per-network scheduler_step()."""

    def test_scheduler_step_single_network(self):
        """scheduler_step(name) steps only that network's scheduler."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [a, b, c]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=10, output_dim=3)
        opt_a = torch.optim.SGD(net_a.parameters(), lr=1.0)
        sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
        program.register_network("net_a", net_a, opt_a, sched_a)

        net_b = SimpleNet(input_dim=10, output_dim=2)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=1.0)
        sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
        program.register_network("net_b", net_b, opt_b, sched_b)

        # Step only net_a's scheduler
        program.scheduler_step("net_a")

        assert opt_a.param_groups[0]['lr'] == pytest.approx(0.5)
        assert opt_b.param_groups[0]['lr'] == pytest.approx(1.0)  # Unchanged

    def test_scheduler_step_none_steps_all(self):
        """scheduler_step(None) or scheduler_step() steps all schedulers (backward compat)."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [a, b, c]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=10, output_dim=3)
        opt_a = torch.optim.SGD(net_a.parameters(), lr=1.0)
        sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
        program.register_network("net_a", net_a, opt_a, sched_a)

        net_b = SimpleNet(input_dim=10, output_dim=2)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=1.0)
        sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
        program.register_network("net_b", net_b, opt_b, sched_b)

        # Step all (backward-compatible call)
        program.scheduler_step()

        assert opt_a.param_groups[0]['lr'] == pytest.approx(0.5)
        assert opt_b.param_groups[0]['lr'] == pytest.approx(0.5)
```

### Step 2: Run test to verify it fails

Run: `.venv/bin/python -m pytest python/tests/test_training.py::TestPerNetworkScheduler::test_scheduler_step_single_network -v`
Expected: FAIL — `TypeError: scheduler_step() takes 1 positional argument but 2 were given`

### Step 3: Modify `scheduler_step` to accept optional name

Replace the existing `scheduler_step` at line 1163 with:

```rust
    /// Step the learning rate scheduler.
    ///
    /// If `network_name` is provided, steps only that network's scheduler.
    /// If `None` (default), steps all registered schedulers.
    #[pyo3(signature = (network_name=None))]
    fn scheduler_step(
        &self,
        py: Python<'_>,
        network_name: Option<&str>,
    ) -> PyResult<()> {
        match network_name {
            Some(name) => {
                let handle = self
                    .network_registry
                    .get(name)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "No network registered with name '{name}'"
                        ))
                    })?;
                if let Some(ref scheduler) = handle.scheduler() {
                    scheduler.call_method0(py, "step")?;
                }
            }
            None => {
                for name in self.network_registry.names() {
                    if let Some(handle) = self.network_registry.get(name) {
                        if let Some(ref scheduler) = handle.scheduler() {
                            scheduler.call_method0(py, "step")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
```

### Step 4: Rebuild and run tests

```bash
cd crates/pyxlog && maturin develop --release 2>&1 | tail -3 && cd ../..
# Run new tests AND the existing scheduler test to verify backward compat
.venv/bin/python -m pytest python/tests/test_training.py::TestPerNetworkScheduler python/tests/test_training.py::TestTrainingWithScheduler -v
```

Expected: 3/3 PASS (2 new + 1 existing backward-compat)

### Step 5: Commit

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_training.py
git commit -m "feat(training): per-network scheduler_step(network_name=None)"
```

---

## Task 4: Gradient clipping in `train_model`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:1348` (`train_epoch_internal` — add `max_grad_norm` param)
- Modify: `crates/pyxlog/src/lib.rs:3495` (`train_model` — pass through `max_grad_norm`)
- Modify: `crates/pyxlog/src/lib.rs:3544` (`train_model_tensor` — pass through `max_grad_norm`)
- Test: `python/tests/test_training.py` (scalar path)
- Test: `python/tests/test_train_model_tensor.py` (tensor path)

### Step 1: Write the failing test

Add a new test class:

```python
class TestGradientClipping:
    """Tests for gradient clipping in train_model."""

    def test_grad_clipping_limits_param_delta(self):
        """Tight max_grad_norm produces smaller parameter changes than no clipping."""
        torch.manual_seed(42)

        def make_program_and_net():
            prog = pyxlog.Program.compile("""
                nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
            """)
            n = SimpleNet()
            opt = torch.optim.SGD(n.parameters(), lr=1.0)
            prog.register_network("test_net", n, opt)
            inputs = torch.randn(20, 10)
            prog.add_tensor_source("data", inputs)
            return prog, n

        queries = [f"pred({i}, a)" for i in range(10)]

        # Run WITHOUT clipping
        prog_no_clip, net_no_clip = make_program_and_net()
        w_before_no_clip = net_no_clip.fc.weight.clone()
        pyxlog.train_model(prog_no_clip, queries, epochs=1, batch_size=10, shuffle=False)
        delta_no_clip = (net_no_clip.fc.weight - w_before_no_clip).norm().item()

        # Run WITH tight clipping
        prog_clip, net_clip = make_program_and_net()
        w_before_clip = net_clip.fc.weight.clone()
        pyxlog.train_model(prog_clip, queries, epochs=1, batch_size=10,
                           shuffle=False, max_grad_norm=0.001)
        delta_clip = (net_clip.fc.weight - w_before_clip).norm().item()

        # Clipped update must be strictly smaller
        assert delta_clip < delta_no_clip, \
            f"Clipped delta {delta_clip:.6f} not smaller than unclipped {delta_no_clip:.6f}"

    def test_grad_clipping_none_is_default(self):
        """train_model without max_grad_norm works as before (no clipping)."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        # No max_grad_norm — backward compatible
        history = pyxlog.train_model(program, queries, epochs=2, batch_size=5)
        assert len(history.epoch_losses) == 2
```

### Step 2: Run test to verify it fails

Run: `.venv/bin/python -m pytest python/tests/test_training.py::TestGradientClipping::test_grad_clipping_limits_param_delta -v`
Expected: FAIL — `TypeError: train_model() got an unexpected keyword argument 'max_grad_norm'`

### Step 3: Implement gradient clipping

**3a. Add a `clip_grad_norms` helper method to `CompiledProgram`** (after `optimizer_step`, around line 1157):

```rust
    /// Clip gradient norms for all registered networks.
    ///
    /// Uses `torch.nn.utils.clip_grad_norm_`.
    fn clip_grad_norms(&self, py: Python<'_>, max_norm: f64) -> PyResult<()> {
        let clip_fn = py
            .import("torch.nn.utils")?
            .getattr("clip_grad_norm_")?;
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(ref module) = handle.module() {
                    let params = module.call_method0(py, "parameters")?;
                    clip_fn.call1((params, max_norm))?;
                }
            }
        }
        Ok(())
    }
```

**3b. Add `max_grad_norm` to `train_epoch_internal`** (line 1348). Change signature:

```rust
    pub(crate) fn train_epoch_internal(
        &mut self,
        py: Python<'_>,
        queries: &[String],
        batch_size: usize,
        log_iter: usize,
        max_grad_norm: Option<f64>,
        history: &mut TrainingHistory,
    ) -> PyResult<EpochStats> {
```

Insert gradient clipping between `forward_backward` and `optimizer_step` (between current lines 1376 and 1379):

```rust
            // Clip gradients if requested
            if let Some(max_norm) = max_grad_norm {
                self.clip_grad_norms(py, max_norm)?;
            }

            // Update parameters
            self.optimizer_step(py)?;
```

**3c. Update `train_model` signature** (line 3494):

```rust
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true, max_grad_norm=None))]
pub fn train_model(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
    max_grad_norm: Option<f64>,
) -> PyResult<TrainingHistory> {
```

Pass `max_grad_norm` to `train_epoch_internal`:

```rust
        let stats =
            program.train_epoch_internal(py, &epoch_queries, batch_size, log_iter, max_grad_norm, &mut history)?;
```

**3d. Update `train_model_tensor` identically** (line 3543) — same signature change and pass-through to `train_epoch_tensor_internal`.

**3e. Update `train_epoch` (the public `#[pymethods]` wrapper, line 1192)** — add `max_grad_norm=None` to its signature, pass through to `train_epoch_internal`:

```rust
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_internal(py, &queries, batch_size, usize::MAX, max_grad_norm, &mut history)
    }
```

**3f. Update `train_epoch_tensor_internal`** (line 1406) — same `max_grad_norm` parameter addition, same clip logic between `forward_backward` and `optimizer_step`.

**3g. Update `train_epoch_tensor` (the public `#[pymethods]` wrapper, line 1258)** — add `max_grad_norm=None` to its signature, pass through to `train_epoch_tensor_internal`:

```rust
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch_tensor(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_tensor_internal(py, &queries, batch_size, usize::MAX, max_grad_norm, &mut history)
    }
```

### Step 4: Rebuild and run scalar-path tests

```bash
cd crates/pyxlog && maturin develop --release 2>&1 | tail -3 && cd ../..
# Run gradient clipping tests AND all existing tests for backward compat
.venv/bin/python -m pytest python/tests/test_training.py -v
```

Expected: All tests PASS (existing + 2 new)

### Step 5: Write tensor-path test for `train_model_tensor(..., max_grad_norm=...)`

Add to `python/tests/test_train_model_tensor.py`, in a new class at the end:

```python
class TestTensorTrainingGradClipping:
    """Test gradient clipping works through train_model_tensor entrypoint."""

    def test_tensor_grad_clipping_limits_param_delta(self):
        """Tight max_grad_norm via tensor path produces smaller weight changes."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        def make_program_and_net():
            torch.manual_seed(42)
            n = SimpleNet()
            prog = pyxlog.Program.compile(source)
            opt = torch.optim.SGD(n.parameters(), lr=1.0)
            prog.register_network("test_net", n, opt)
            torch.manual_seed(99)
            inputs = torch.randn(20, 10)
            prog.add_tensor_source("data", inputs)
            return prog, n

        queries = [f"pred({i}, a)" for i in range(10)]

        # Run WITHOUT clipping
        prog_no_clip, net_no_clip = make_program_and_net()
        w_before = net_no_clip.fc.weight.clone()
        pyxlog.train_model_tensor(prog_no_clip, queries, epochs=1,
                                  batch_size=10, shuffle=False)
        delta_no_clip = (net_no_clip.fc.weight - w_before).norm().item()

        # Run WITH tight clipping
        prog_clip, net_clip = make_program_and_net()
        w_before = net_clip.fc.weight.clone()
        pyxlog.train_model_tensor(prog_clip, queries, epochs=1,
                                  batch_size=10, shuffle=False,
                                  max_grad_norm=0.001)
        delta_clip = (net_clip.fc.weight - w_before).norm().item()

        assert delta_clip < delta_no_clip, \
            f"Clipped delta {delta_clip:.6f} not smaller than unclipped {delta_no_clip:.6f}"
```

### Step 6: Run tensor-path test to verify it passes

```bash
.venv/bin/python -m pytest python/tests/test_train_model_tensor.py::TestTensorTrainingGradClipping -v
```

Expected: 1/1 PASS

### Step 7: Commit

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_training.py python/tests/test_train_model_tensor.py
git commit -m "feat(training): gradient clipping via max_grad_norm in train_model"
```

---

## Task 5: Early stopping in `train_model`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:3495` (`train_model` — add `val_queries`, `patience`)
- Modify: `crates/pyxlog/src/lib.rs:3544` (`train_model_tensor` — same)
- Modify: `crates/pyxlog/src/lib.rs:3446` (`TrainingHistory` — add `stopped_early` field)
- Test: `python/tests/test_training.py` (scalar path)
- Test: `python/tests/test_train_model_tensor.py` (tensor path)

### Step 1: Write the failing test

Add a new test class:

```python
class TestEarlyStopping:
    """Tests for early stopping in train_model."""

    def test_early_stopping_triggers(self):
        """train_model stops early when val loss stops improving.

        Uses lr=0.0 so the network never updates — val loss is flat from
        epoch 1, guaranteeing early stop after exactly `patience` epochs
        of no improvement (plus the initial improving epoch = patience+1
        total, though epoch 0 sets best_val_loss, so we get patience+1
        epochs if first epoch sets baseline then patience epochs with no
        improvement).
        """
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        # lr=0: optimizer.step() is a no-op → val loss never improves
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        train_queries = [f"pred({i}, a)" for i in range(10)]
        val_queries = [f"pred({i}, b)" for i in range(10, 15)]

        patience = 3
        history = pyxlog.train_model(
            program, train_queries, epochs=100,
            batch_size=5, val_queries=val_queries, patience=patience
        )

        # Epoch 0 sets baseline (improvement), epochs 1..patience have no
        # improvement → stop after patience+1 total epochs.
        assert len(history.epoch_losses) == patience + 1
        assert history.stopped_early is True

    def test_early_stopping_disabled_by_default(self):
        """Without val_queries/patience, all epochs run (backward compat)."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        history = pyxlog.train_model(program, queries, epochs=3, batch_size=5)

        assert len(history.epoch_losses) == 3
        assert history.stopped_early is False

    def test_early_stopping_requires_both_params(self):
        """val_queries without patience (or vice versa) raises ValueError."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        with pytest.raises(ValueError):
            pyxlog.train_model(
                program, queries, epochs=5, batch_size=5,
                val_queries=queries  # patience not provided
            )
```

### Step 2: Run test to verify it fails

Run: `.venv/bin/python -m pytest python/tests/test_training.py::TestEarlyStopping::test_early_stopping_requires_both_params -v`
Expected: FAIL — `TypeError: train_model() got an unexpected keyword argument 'val_queries'`

### Step 3: Implement early stopping

**3a. Add `stopped_early` field to `TrainingHistory`** (line 3446):

```rust
pub struct TrainingHistory {
    #[pyo3(get)]
    pub epoch_losses: Vec<f64>,
    #[pyo3(get)]
    pub epoch_times: Vec<f64>,
    #[pyo3(get)]
    pub batch_losses: Vec<f64>,
    /// True if training was stopped early due to validation loss plateau.
    #[pyo3(get)]
    pub stopped_early: bool,
}
```

Update `TrainingHistory::new()` to initialize `stopped_early: false`.

**3b. Update `train_model` signature and body** (line 3494):

```rust
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true, max_grad_norm=None, val_queries=None, patience=None))]
pub fn train_model(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
    max_grad_norm: Option<f64>,
    val_queries: Option<Vec<String>>,
    patience: Option<usize>,
) -> PyResult<TrainingHistory> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use std::time::Instant;

    // Validate: val_queries and patience must both be present or both absent
    match (&val_queries, &patience) {
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "val_queries and patience must both be provided for early stopping"
            ));
        }
        _ => {}
    }

    let mut history = TrainingHistory::new();
    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let epoch_start = Instant::now();
        let stats = program.train_epoch_internal(
            py, &epoch_queries, batch_size, log_iter, max_grad_norm, &mut history,
        )?;
        history.add_epoch(stats.avg_loss, epoch_start.elapsed().as_secs_f64());

        println!(
            "Epoch {}/{}: avg_loss={:.6}",
            epoch + 1, epochs, stats.avg_loss
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // Early stopping check
        if let (Some(ref val_q), Some(pat)) = (&val_queries, patience) {
            let val_loss = program.evaluate_loss(val_q.clone())?;
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }
            if epochs_without_improvement >= pat {
                history.stopped_early = true;
                break;
            }
        }
    }

    Ok(history)
}
```

**3c. Update `train_model_tensor` identically** — same new parameters, same early stopping logic using `evaluate_loss`.

### Step 4: Rebuild and run scalar-path tests

```bash
cd crates/pyxlog && maturin develop --release 2>&1 | tail -3 && cd ../..
.venv/bin/python -m pytest python/tests/test_training.py -v
```

Expected: All tests PASS (existing + 3 new)

### Step 5: Write tensor-path test for `train_model_tensor(..., val_queries=..., patience=...)`

Add to `python/tests/test_train_model_tensor.py`, in a new class at the end:

```python
class TestTensorTrainingEarlyStopping:
    """Test early stopping works through train_model_tensor entrypoint."""

    def test_tensor_early_stopping_triggers(self):
        """train_model_tensor stops early when val loss plateaus.

        Uses lr=0.0 so optimizer is a no-op — val loss never improves,
        triggering early stop after patience+1 total epochs.
        """
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        torch.manual_seed(42)
        net = SimpleNet()
        prog = pyxlog.Program.compile(source)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        prog.register_network("test_net", net, opt)

        torch.manual_seed(99)
        inputs = torch.randn(20, 10)
        prog.add_tensor_source("data", inputs)

        train_queries = [f"pred({i}, a)" for i in range(10)]
        val_queries = [f"pred({i}, b)" for i in range(10, 15)]

        patience = 3
        history = pyxlog.train_model_tensor(
            prog, train_queries, epochs=100,
            batch_size=5, shuffle=False,
            val_queries=val_queries, patience=patience,
        )

        assert len(history.epoch_losses) == patience + 1
        assert history.stopped_early is True

    def test_tensor_early_stopping_requires_both_params(self):
        """val_queries without patience (or vice versa) raises ValueError via tensor path."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        torch.manual_seed(42)
        net = SimpleNet()
        prog = pyxlog.Program.compile(source)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        prog.register_network("test_net", net, opt)

        torch.manual_seed(99)
        inputs = torch.randn(20, 10)
        prog.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(8)]

        with pytest.raises(ValueError):
            pyxlog.train_model_tensor(
                prog, queries, epochs=5, batch_size=4, shuffle=False,
                val_queries=queries,  # patience not provided
            )

    def test_tensor_early_stopping_stopped_early_false(self):
        """Without val_queries, all epochs run and stopped_early is False."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        torch.manual_seed(42)
        net = SimpleNet()
        prog = pyxlog.Program.compile(source)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        prog.register_network("test_net", net, opt)

        torch.manual_seed(99)
        inputs = torch.randn(20, 10)
        prog.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(8)]

        history = pyxlog.train_model_tensor(
            prog, queries, epochs=3, batch_size=4, shuffle=False,
        )

        assert len(history.epoch_losses) == 3
        assert history.stopped_early is False
```

### Step 6: Run tensor-path test to verify it passes

```bash
.venv/bin/python -m pytest python/tests/test_train_model_tensor.py::TestTensorTrainingEarlyStopping -v
```

Expected: 3/3 PASS

### Step 7: Commit

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_training.py python/tests/test_train_model_tensor.py
git commit -m "feat(training): early stopping via val_queries + patience in train_model"
```

---

## Task 6: Full regression + commit docs

**Files:**
- Test: All training tests
- Modify: `CHANGELOG.md`

### Step 1: Run full training + related test suites

This plan modifies `scheduler_step`, `train_model_tensor`, and tensor-epoch plumbing.
Existing coverage for those lives in `test_backward.py` and `test_train_model_tensor.py`,
so all three suites must pass.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /home/dev/projects/xlog
.venv/bin/python -m pytest python/tests/test_training.py python/tests/test_backward.py python/tests/test_train_model_tensor.py -v
```

Expected: All tests PASS (13 original training + 11 new + backward + tensor suites)

### Step 2: Run Rust workspace tests

```bash
cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20
```

Expected: All pass, exit code 0

### Step 3: Update CHANGELOG.md

Add under `## [Unreleased]` → `### Added`:

```markdown
- `program.get_lr(network_name)` — read current learning rate for a registered network
- `program.set_lr(network_name, lr)` — set learning rate for all param groups of a registered network
- `train_model(..., max_grad_norm=N)` — gradient clipping via `torch.nn.utils.clip_grad_norm_`
- `train_model(..., val_queries=[...], patience=N)` — early stopping when validation loss plateaus
- `program.scheduler_step(network_name)` — step a single network's scheduler (None = all, backward compatible)
- `TrainingHistory.stopped_early` — boolean flag indicating early stopping was triggered
```

### Step 4: Commit docs

```bash
git add CHANGELOG.md
git commit -m "docs: record P2b extended training controls in changelog"
```

---

## Summary

| Task | Feature | New tests | Test file |
|------|---------|-----------|-----------|
| 1 | `get_lr(network_name)` | 2 | `test_training.py` |
| 2 | `set_lr(network_name, lr)` | 2 | `test_training.py` |
| 3 | Per-network `scheduler_step(network_name=None)` | 2 | `test_training.py` |
| 4 | Gradient clipping (`max_grad_norm`) | 2 + 1 | `test_training.py` + `test_train_model_tensor.py` |
| 5 | Early stopping (`val_queries` + `patience`) | 3 + 3 | `test_training.py` + `test_train_model_tensor.py` |
| 6 | Regression + docs | 0 | all three suites |

Total: 6 tasks, 15 new tests (11 in `test_training.py`, 4 in `test_train_model_tensor.py`), 6 commits.
