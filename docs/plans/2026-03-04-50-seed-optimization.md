# 50-Seed Runtime Budget Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring `test_ga_reliability_50` from ~1700s to ≤600s by compiling once per `train_only()` and resetting mutable runtime state between attempts.

**Architecture:** Add `reset_runtime()` to `CompiledIlpProgram` in Rust (clears executor store/ILP registry/counters, reloads facts, re-executes plan). Refactor Python `train_only()` to compile once and call `reset_runtime()` between attempts. Add instrumentation and env var overrides for the GA test.

**Tech Stack:** Rust (PyO3), Python, CUDA (via xlog-cuda)

**Design doc:** `docs/plans/2026-03-04-50-seed-optimization-design.md`

---

### Task 1: Add `IlpRegistry::clear()` in Rust

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs:71-75`

**Step 1: Add the `clear` method**

In `crates/xlog-runtime/src/ilp_registry.rs`, add after the `new()` method (after line 75):

```rust
    /// Clear all registered masks, releasing GPU buffers.
    pub fn clear(&mut self) {
        self.masks.clear();
    }
```

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-runtime`
Expected: compiles with no errors.

**Step 3: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs
git commit -m "feat(runtime): add IlpRegistry::clear() for mask cleanup"
```

---

### Task 2: Add `Executor::reset_for_ilp()` in Rust

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs:270-276`

**Step 1: Add the method**

In `crates/xlog-runtime/src/executor.rs`, add after `reset_for_mc()` (after line 276):

```rust
    /// Reset executor state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks + tagged results), relation storage,
    /// join index cache, stats, and profiler. Preserves relation name
    /// registrations (rel_names, name_to_rel) since those are immutable
    /// compile artifacts.
    pub fn reset_for_ilp(&mut self) {
        self.ilp_registry.clear();
        self.ilp_last_result = None;
        self.store.clear();
        self.join_index_cache.clear();
        self.stats = StatsManager::new();
        self.profiler = Profiler::default();
    }
```

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-runtime`
Expected: compiles with no errors. `StatsManager::new()` is at `crates/xlog-runtime/src/statistics.rs:63`, `Profiler::default()` is at `crates/xlog-runtime/src/profiler.rs:532` — both already in scope.

**Step 3: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "feat(runtime): add Executor::reset_for_ilp() for attempt reuse"
```

---

### Task 3: Add `CompiledIlpProgram::reset_runtime()` in Rust (PyO3)

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:3887-3866` (the `#[pymethods] impl CompiledIlpProgram` block)

**Step 1: Add the method**

In `crates/pyxlog/src/lib.rs`, add inside the `#[pymethods] impl CompiledIlpProgram` block (after the existing `set_rule_mask` method, or at the end of the block). Place it near `evaluate()` since it's structurally similar:

```rust
    /// Reset mutable runtime state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks/tagged results), executor store,
    /// join index cache, stats, and profiler. Then re-registers schemas
    /// with empty buffers, reloads base facts from AST, and re-executes
    /// the plan. Preserves all immutable compile artifacts (AST, plan,
    /// schemas, rel_index, provider, TMJ metadata, max_active_rules).
    ///
    /// After reset, the program is in the same state as a fresh compile()
    /// with the same source — ready for set_rule_mask / evaluate cycles.
    pub fn reset_runtime(&mut self, py: Python<'_>) -> PyResult<()> {
        let _result: xlog_core::Result<()> = py.allow_threads(|| {
            // 1. Clear all mutable state (ILP registry, store, caches, stats)
            self.executor.reset_for_ilp();

            // 2. Re-register schemas with empty buffers
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }

            // 3. Reload base facts from preserved AST
            load_facts_into_store(
                &self.ast, &self.provider, &mut self.executor, &self.schemas,
            )?;

            // 4. Re-execute plan (populates derived relations)
            self.executor.execute_plan(&self.plan)?;

            Ok(())
        });
        _result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 5. Reset D2H transfer counter
        self.provider.reset_d2h_transfer_count();

        Ok(())
    }
```

**Step 2: Build the full pyxlog crate**

Run: `cargo build -p pyxlog --release`
Expected: compiles with no errors.

**Step 3: Rebuild the Python extension**

Run: `cd crates/pyxlog && ../../.venv/bin/maturin develop --release`
Expected: installs successfully.

**Step 4: Quick smoke test from Python**

Run:
```bash
.venv/bin/python -c "
import pyxlog
prog = pyxlog.IlpProgramFactory.compile('edge(1,2). edge(2,3). learnable(W) :: reach(X,Y) :- bL(X,Z), bR(Z,Y).', device=0, memory_mb=512)
print('Before reset:', prog.ilp_schema_size())
prog.reset_runtime()
print('After reset:', prog.ilp_schema_size())
print('OK')
"
```
Expected: prints schema size twice (same value) and "OK".

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): add CompiledIlpProgram.reset_runtime() for attempt reuse"
```

---

### Task 4: Write parity and state-leak tests

**Files:**
- Create: `python/tests/test_ilp_reset.py`

**Step 1: Write the test file**

Create `python/tests/test_ilp_reset.py`:

```python
"""Tests for CompiledIlpProgram.reset_runtime() parity and state isolation."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
edge(1, 2). edge(2, 3). edge(3, 4).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""

POS = [("reach", [1, 3]), ("reach", [2, 4])]


def _compile():
    return pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)


def _get_facts(prog, rel_name):
    """Return sorted list of tuples for deterministic comparison."""
    facts = prog.relation_facts(rel_name)
    return sorted(tuple(f) for f in facts)


def test_reset_parity_with_fresh_compile():
    """compile+reset+evaluate must match a fresh compile."""
    # Fresh compile A
    prog_a = _compile()
    facts_a = _get_facts(prog_a, "edge")

    # Compile B, then reset
    prog_b = _compile()
    prog_b.reset_runtime()
    facts_b = _get_facts(prog_b, "edge")

    assert facts_a == facts_b, f"Parity failed: {facts_a} != {facts_b}"

    # Also check derived relations match
    reach_a = _get_facts(prog_a, "reach")
    reach_b = _get_facts(prog_b, "reach")
    assert reach_a == reach_b, f"Derived parity failed: {reach_a} != {reach_b}"


def test_reset_clears_ilp_registry():
    """After reset, ILP masks and tagged results must be gone."""
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()
    c = len(cands)

    # Set a sparse mask and evaluate
    ids = list(range(c))
    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse("W", ids, soft, 32)
    prog.evaluate()
    tagged_before = prog.get_tagged_results()
    assert len(tagged_before) > 0, "Should have tagged results after evaluate"

    # Reset
    prog.reset_runtime()

    # After reset: no tagged results (ILP registry cleared)
    tagged_after = prog.get_tagged_results()
    assert len(tagged_after) == 0, (
        f"Tagged results should be empty after reset, got {tagged_after}"
    )


def test_reset_no_state_leak_across_masks():
    """Setting mask after reset must not be influenced by pre-reset mask."""
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()
    c = len(cands)

    # Run with uniform soft probs, get result
    ids = list(range(c))
    soft_uniform = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse("W", ids, soft_uniform, 32)
    prog.evaluate()
    results_before = prog.get_tagged_results()

    # Reset
    prog.reset_runtime()

    # Run with concentrated soft probs (all weight on candidate 0)
    cands_after = prog.valid_candidates("W", False)
    c_after = len(cands_after)
    assert c_after == c, "Candidate count should match after reset"

    soft_concentrated = torch.zeros(c, device="cuda", dtype=torch.float64)
    soft_concentrated[0] = 1.0
    prog.set_rule_mask_sparse("W", list(range(c)), soft_concentrated, 1)
    prog.evaluate()
    results_after = prog.get_tagged_results()

    # Results should differ (concentrated mask selects only 1 candidate)
    assert results_after != results_before, (
        "Results should differ between uniform and concentrated masks"
    )


def test_reset_d2h_counter_cleared():
    """D2H transfer counter must be zero after reset."""
    prog = _compile()
    prog.reset_d2h_transfer_count()
    prog.reset_runtime()
    assert prog.d2h_transfer_count() == 0, "D2H counter should be 0 after reset"
```

**Step 2: Run the tests**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_reset.py -v --timeout=60`
Expected: 4/4 PASS.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_reset.py
git commit -m "test: add reset_runtime() parity, state isolation, and D2H counter tests"
```

---

### Task 5: Refactor `train_only()` to compile once

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:31-107` (`train_only()`)
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:227-517` (`_run_single_attempt()`)

**Step 1: Refactor `_run_single_attempt()` signature**

Change `_run_single_attempt()` (trainer.py:227) to accept a pre-compiled program:

Old signature:
```python
def _run_single_attempt(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _AttemptResult:
```

New signature:
```python
def _run_single_attempt(
    prog: object,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _AttemptResult:
```

Remove the internal `compile()` call (lines 244-247):
```python
        prog = pyxlog.IlpProgramFactory.compile(
            source, device=config.device, memory_mb=config.memory_mb,
            max_active_rules=config.max_active_rules,
        )
```

The function now uses the `prog` parameter directly. Everything else in `_run_single_attempt` remains unchanged — `valid_candidates()`, `ilp_schema_size()`, mask setting, evaluation, convergence checks all work on the passed-in program.

**Step 2: Refactor `train_only()` attempt loop**

Replace the body of `train_only()` (lines 44-107) with:

```python
    _validate_inputs(source, mask_name, positives, negatives, config)

    if config.deterministic:
        _set_deterministic_cuda(config.seed)

    # Compile once for all attempts
    import time as _time
    _compile_t0 = _time.perf_counter()
    prog = pyxlog.IlpProgramFactory.compile(
        source, device=config.device, memory_mb=config.memory_mb,
        max_active_rules=config.max_active_rules,
    )
    compile_ms = (_time.perf_counter() - _compile_t0) * 1000.0

    attempts: list[_AttemptResult] = []
    global_steps = 0
    numeric_failures = 0
    reset_ms_list: list[float] = []

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        if config.deterministic:
            _seed_for_attempt(config.seed, attempt_idx)

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        # Reset runtime state for attempts after the first
        if attempt_idx > 0:
            _reset_t0 = _time.perf_counter()
            prog.reset_runtime()
            reset_ms_list.append((_time.perf_counter() - _reset_t0) * 1000.0)

        try:
            result = _run_single_attempt(
                prog,
                mask_name,
                positives,
                negatives,
                config,
                attempt_idx,
                step_budget,
            )
            # Attach compile/reset instrumentation to winning attempt
            result.telemetry_timings["compile_ms_once"] = compile_ms
            if reset_ms_list:
                result.telemetry_timings["reset_ms_total"] = sum(reset_ms_list)
                result.telemetry_timings["reset_ms_p95"] = _percentile_95(reset_ms_list)
            attempts.append(result)
            global_steps += result.steps_used
        except _NumericFailure:
            numeric_failures += 1
            global_steps += 1  # count as at least 1 step
            # Reset before next attempt after numeric failure too
            if attempt_idx < config.max_attempts - 1:
                _reset_t0 = _time.perf_counter()
                prog.reset_runtime()
                reset_ms_list.append((_time.perf_counter() - _reset_t0) * 1000.0)
            if numeric_failures >= config.max_numeric_failures:
                raise IlpTrainingError(
                    "numeric_instability: too many NaN/Inf failures",
                    {
                        "attempt": attempt_idx,
                        "step": None,
                        "C": 0,
                        "k": config.max_active_rules,
                        "device_name": _device_name(config),
                        "allocated_bytes": _allocated_bytes(config.device),
                        "terminal_reason": "numeric_instability",
                    },
                )
        except _AttemptRuntimeFailure as e:
            context = e.context
            raise IlpTrainingError(f"{e.terminal_reason}: {e.cause}", context)

    return _select_winner(
        attempts,
        source,
        mask_name,
        positives,
        negatives,
        config,
        global_steps,
        _compute_holdout=_compute_holdout,
    )
```

Note: `import time as _time` uses underscore-prefix to avoid polluting the module namespace. If `time` is already imported at the top of the file, use that instead and skip this import.

**Step 3: Run the full Python test suite (non-slow)**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/ --timeout=120 -k "not slow and not test_second_epoch and not test_non_monotone" -v`
Expected: All pass (255 pass, ~11 skip). This validates the refactor didn't break anything.

**Step 4: Run the ILP reliability gate**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600`
Expected: 20/20 PASS. This is the beta reliability gate — must remain green.

**Step 5: Run the reset parity tests**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_reset.py -v --timeout=60`
Expected: 4/4 PASS.

**Step 6: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py
git commit -m "feat(ilp): compile once per train_only(), reset_runtime() between attempts"
```

---

### Task 6: Add train-level deterministic reproducibility test

**Files:**
- Modify: `python/tests/test_ilp_reset.py`

**Step 1: Add the test**

Append to `python/tests/test_ilp_reset.py`:

```python
from pyxlog.ilp import TrainConfig, train_only


def test_train_deterministic_reproducibility():
    """Two train_only() calls with same seed must produce identical results."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        deterministic=True,
        debug_dense_mask=False,
    )

    result_1 = train_only(
        source=SOURCE,
        mask_name="W",
        positives=POS,
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    result_2 = train_only(
        source=SOURCE,
        mask_name="W",
        positives=POS,
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    assert result_1.converged == result_2.converged, (
        f"Determinism: converged mismatch {result_1.converged} vs {result_2.converged}"
    )
    assert result_1.discovered_rule == result_2.discovered_rule, (
        f"Determinism: rule mismatch {result_1.discovered_rule!r} vs {result_2.discovered_rule!r}"
    )
```

**Step 2: Run it**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_reset.py::test_train_deterministic_reproducibility -v --timeout=120`
Expected: PASS.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_reset.py
git commit -m "test: add train-level deterministic reproducibility test for reset path"
```

---

### Task 7: Add GA test env var overrides

**Files:**
- Modify: `python/tests/test_ilp_ga_reliability.py:45-53`

**Step 1: Add env var reads**

In `test_ga_reliability_50()`, after line 31 (`seed_count = max(1, seed_count)`), add:

```python
    step_budget = int(os.getenv("GA_RELIABILITY_STEP_BUDGET", "150"))
    max_attempts = int(os.getenv("GA_RELIABILITY_MAX_ATTEMPTS", "7"))
```

Then modify the `TrainConfig` construction (lines 45-53) to use these variables:

```python
            config = TrainConfig(
                step_budget_per_attempt=step_budget,
                max_attempts=max_attempts,
                tau_start=2.0,
                tau_floor=0.05,
                device=0,
                memory_mb=512,
                debug_dense_mask=False,
                seed=seed,
            )
```

**Step 2: Verify existing behavior unchanged**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib GA_RELIABILITY_SEEDS=2 .venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v --timeout=120 -k "test_ga_reliability_50"`
Expected: Runs 2 seeds (8 runs). Will fail CI assertion (too few samples), but should not error. Verify output shows `runs: 8` and `step_budget_per_attempt=150, max_attempts=7` (defaults).

**Step 3: Commit**

```bash
git add python/tests/test_ilp_ga_reliability.py
git commit -m "feat(test): add GA_RELIABILITY_STEP_BUDGET and GA_RELIABILITY_MAX_ATTEMPTS env var overrides"
```

---

### Task 8: Measure before/after with GA reliability test

**Files:**
- No files modified — measurement only.

**Step 1: Run the 50-seed GA test with timing**

Run:
```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib \
  time .venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v --timeout=1800 -s
```

(Timeout raised to 1800s to give the test room to finish.)

**Step 2: Record the results**

Note:
- Wall clock time (from `time` output)
- `runs`, `success`, `clopper_pearson_95_lower` from test stdout
- Whether the CI assertion passes (≥ 0.929)

**Step 3: If PASS, commit evidence**

Save the output to `docs/evidence/2026-03-04-50-seed-optimization/ga-reliability-post.log`:

```bash
mkdir -p docs/evidence/2026-03-04-50-seed-optimization
# (redirect test output to the log file)
git add -f docs/evidence/2026-03-04-50-seed-optimization/ga-reliability-post.log
git commit -m "evidence: 50-seed GA reliability post-optimization timing"
```

If FAIL (CI too low or timeout), investigate and report — do not force pass.

---

### Task 9: Run full validation matrix

Run the 7-suite validation matrix from the beta release to verify no regressions:

**Step 1: Rust workspace**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: 1208 pass, 0 fail.

**Step 2: CUDA certification**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: 206/206.

**Step 3: Python non-slow batch**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/ --timeout=120 -k "not slow and not test_second_epoch and not test_non_monotone" -v`
Expected: All pass.

**Step 4: ILP reliability**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600`
Expected: 20/20.

**Step 5: ILP sparse**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120`
Expected: 8/8.

**Step 6: ILP performance**

Run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib .venv/bin/python -m pytest python/tests/test_ilp_performance.py -v --timeout=600`
Expected: 3/3.

**Step 7: Commit validation evidence if all pass**

```bash
git commit --allow-empty -m "evidence: full 7-suite validation pass after 50-seed optimization"
```
