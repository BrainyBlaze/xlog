# Wave 4: Deduplication & Cleanup

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated code, consolidate entry points, fix silent exception handling, and make hardcoded constants configurable.

**Architecture:** Extract shared functions, deprecate redundant entry points, add logging for caught exceptions.

**Tech Stack:** Rust + Python. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (M1–M12)

**Prerequisite:** Wave 3 (API coherence) should be complete, but individual tasks are independent.

---

### Task 1: WFS entry point consolidation (M1)

**Files:**
- Modify: `crates/xlog-prob/src/wfs.rs`

**Context:** Five public entry points exist. Only `evaluate_wfs_rules()` is used in production (called from `provenance.rs:784`).

- [ ] **Step 1: Identify the 5 entry points**

```
evaluate_wfs_rules         — the real one (production caller)
evaluate_wfs_scc           — labeled "Legacy interface"
evaluate_wfs_scc_with_config
evaluate_wfs_with_rules
evaluate_wfs_with_rules_config
```

- [ ] **Step 2: Deprecate the 4 unused entry points**

Add `#[deprecated(since = "0.7.0", note = "Use evaluate_wfs_rules() instead")]` to each.

- [ ] **Step 3: Run tests**

Run: `cargo test -p xlog-prob --release 2>&1 | tail -10`

Expected: All pass (deprecation warnings are warnings, not errors).

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-prob/src/wfs.rs
git commit -m "refactor(prob): deprecate 4 unused WFS entry points, keep evaluate_wfs_rules()"
```

### Task 2: Relocate schema compatibility check (M2)

**Files:**
- Modify: `crates/xlog-core/src/schema.rs` (or wherever Schema is defined)
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Context:** `schemas_type_compatible` is a private method on `CudaKernelProvider` at line 7422. Move it to `xlog-core` as a free function on `Schema`.

- [ ] **Step 1: Write test in xlog-core**

```rust
#[test]
fn schemas_type_compatible_same_types() {
    let a = Schema::new(vec![("x".into(), ScalarType::U32)]);
    let b = Schema::new(vec![("y".into(), ScalarType::U32)]);
    assert!(a.type_compatible(&b));
}

#[test]
fn schemas_type_compatible_different_types() {
    let a = Schema::new(vec![("x".into(), ScalarType::U32)]);
    let b = Schema::new(vec![("x".into(), ScalarType::F64)]);
    assert!(!a.type_compatible(&b));
}
```

- [ ] **Step 2: Add `type_compatible` method to Schema**

- [ ] **Step 3: Update provider to use the new method**

Replace `self.schemas_type_compatible(a, b)` with `a.type_compatible(b)`.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "refactor(core): move schema type compatibility check to Schema method"
```

### Task 3: Python _commit_rule() dedup (M4)

**Files:**
- Create: `crates/pyxlog/python/pyxlog/ilp/_utils.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/holdout.py:207-216`
- Modify: `crates/pyxlog/python/pyxlog/ilp/promoter.py:376-386`

- [ ] **Step 1: Extract shared function**

Create `_utils.py` with the shared `_commit_rule()` function.

- [ ] **Step 2: Import in both holdout.py and promoter.py**

Replace local definitions with `from ._utils import _commit_rule`.

- [ ] **Step 3: Run Python tests**

```bash
.venv/bin/python -m pytest python/tests/ -x -q
```

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/
git commit -m "refactor(python): extract shared _commit_rule() to _utils.py"
```

### Task 4: Split _run_single_attempt (M5)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:247-586`

- [ ] **Step 1: Identify natural split points in the 340-line function**

Break into:
- `_prepare_attempt()` — setup, optimizer creation
- `_run_training_loop()` — the step/epoch loop
- `_evaluate_candidate()` — accuracy scoring
- `_finalize_attempt()` — result packaging

- [ ] **Step 2: Extract each function, keeping the original as a coordinator**

```python
def _run_single_attempt(self, ...):
    state = self._prepare_attempt(...)
    result = self._run_training_loop(state, ...)
    score = self._evaluate_candidate(result, ...)
    return self._finalize_attempt(score, result, ...)
```

- [ ] **Step 3: Run Python tests**

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(python): split _run_single_attempt into 4 focused functions"
```

### Task 5: Fix silent exception swallowing (M6)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/promoter.py:370`
- Modify: `crates/pyxlog/python/pyxlog/ilp/holdout.py:84`
- Modify: `crates/pyxlog/python/pyxlog/ilp/holdout.py:155`

- [ ] **Step 1: Replace bare except with specific catches**

```python
# Before
try:
    ...
except Exception:
    continue

# After
import logging
logger = logging.getLogger(__name__)

try:
    ...
except (ValueError, RuntimeError) as e:
    logger.debug("Skipping candidate: %s", e)
    continue
except Exception:
    logger.warning("Unexpected error during candidate evaluation", exc_info=True)
    continue
```

Identify the actual expected exception types by reading the try block to understand what operations could fail.

- [ ] **Step 2: Run Python tests**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(python): replace silent exception swallowing with specific catches + logging"
```

### Task 6: Python configurable constants (M11)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/holdout.py`

- [ ] **Step 1: Add fields to IlpConfig (or TrainerConfig)**

Find the existing config dataclass and add:
```python
learning_rate: float = 0.1
mining_interval: int = 20
loo_threshold: int = 20
```

- [ ] **Step 2: Replace hardcoded values with config references**

- `trainer.py:295` — Adam LR=0.1 → `self.config.learning_rate`
- `trainer.py:349` — mining interval=20 → `self.config.mining_interval`
- `holdout.py:31` — LOO threshold=20 → `self.config.loo_threshold`

- [ ] **Step 3: Run Python tests**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(python): make hardcoded training constants configurable via IlpConfig"
```

### Task 7: Row count helper dedup (M8)

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`

**Context:** `buffer_row_count()` in executor.rs (line 3050) duplicates `device_row_count()` in provider.rs (line 7200). The executor version adds caching.

- [ ] **Step 1: Assess if these are truly duplicates**

If executor's version adds caching that provider's doesn't, they serve different purposes. In that case, document the distinction rather than deduplicating.

- [ ] **Step 2: If dedup is appropriate, delegate executor's version to provider**

```rust
fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
    // Use provider's device_row_count which handles the D2H read
    Ok(self.provider.device_row_count(buffer)? as u32)
}
```

- [ ] **Step 3: Run tests, commit**

### Task 8: Kernel launch macro for gpu_cache.rs (M12)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`

- [ ] **Step 1: Define the macro**

```rust
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        $func.clone().launch($cfg, ($($args,)*))
            .map_err(|e| XlogError::Kernel(format!("Kernel launch failed: {}", e)))
    }};
}
```

- [ ] **Step 2: Replace 55+ instances**

Search for `.clone().launch(` in `gpu_cache.rs` and replace with `launch_kernel!`.

- [ ] **Step 3: Run MC tests**

Run: `cargo test -p xlog-prob --release --test mc`

Expected: 21/21 PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(prob): replace 55 repetitive kernel launches with launch_kernel! macro"
```

### Task 9: Remaining dedup items (M3, M9, M10)

These are lower-priority and can be addressed opportunistically:

- [ ] **M3 — mc.rs clone reduction:** Profile first, then target the largest clones.
- [ ] **M9 — gpu_d4.rs frontier types:** Merge duplicate definitions.
- [ ] **M10 — lower.rs join planning:** Extract shared logic.

### Task 10: Final validation

- [ ] **Step 1:** Full workspace Rust tests.
- [ ] **Step 2:** CUDA certification (206/206).
- [ ] **Step 3:** Python tests (all pass).
