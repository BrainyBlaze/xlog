# 50-Seed Runtime Budget Optimization Design

**Goal:** Bring `test_ga_reliability_50` (50 seeds × 4 stages = 200 runs) from ~1700s to ≤600s by eliminating redundant per-attempt compilation within `train_only()`.

**Approach:** Rust-side `reset_runtime()` on `CompiledIlpProgram` — compile once per `train_only()` call, reset mutable runtime state between attempts.

**Speedup target is empirical.** Instrumentation before/after will verify actual gains. Eliminating 6 of 7 compile() calls per run (~600ms each) predicts ~3x, but this must be measured.

---

## 1. Problem

Each `train_only()` call runs up to 7 attempts (`max_attempts=7`). Each attempt calls `IlpProgramFactory.compile()` (lib.rs:3792-3866), which re-executes the full pipeline: parse → AST → plan → GPU provider → executor → schema registration → fact loading → plan execution → TMJ metadata extraction. Source text is identical across all attempts within a single `train_only()` call. At 300-1500ms per compile and up to 7 compiles per run, compilation dominates runtime.

---

## 2. `reset_runtime()` Contract

New method on `CompiledIlpProgram` (Rust, exposed via PyO3).

### Must reset (mutable runtime state)

| State | Location | Reset action |
|-------|----------|-------------|
| ILP registry (masks, tagged results) | `executor.ilp_registry` | Clear before plan re-execution |
| ILP last result | `executor.ilp_last_result` | Set to `None` |
| Executor store | `executor.store` | Clear all relations |
| Join index cache | `executor.join_index_cache` | Clear |
| Stats/profiler | `executor.stats`, `executor.profiler` | Reset counters |
| Schema registration | `executor.store` | Re-register empty buffers from `self.schemas` |
| Base facts | `executor.store` | Re-run `load_facts_into_store()` from preserved AST |
| Plan execution | `executor` | Re-run `execute_plan(&self.plan)` |
| D2H transfer counter | `self.provider` | Reset to 0 |

Order matters: clear ILP registry/last_result → clear store → re-register schemas → reload facts → execute plan → reset D2H counter.

### Must preserve (immutable compile artifacts)

| Artifact | Field |
|----------|-------|
| Source text | `base_source`, `_learnable_source` |
| Parsed AST | `ast` |
| Compiled execution plan | `plan` |
| Relation schemas | `schemas` |
| Relation ID mapping | `rel_index` |
| GPU kernel provider | `provider` (shared Arc) |
| TMJ join metadata | `left_keys`, `right_keys`, `head_projection`, `compiled_schema_size`, `head_rel_name` |
| Candidate ordering bound | `max_active_rules` |

### Relationship to `evaluate()`

The existing `evaluate()` method (lib.rs:3998-4012) does a subset of this work: it calls `reset_for_mc()` (clears store + join index cache), re-creates empty buffers, reloads facts, and re-executes the plan. But it does **not** clear the ILP registry or last result. `reset_runtime()` must explicitly clear those before re-executing the plan, otherwise stale masks from a previous attempt bleed into the reset run.

---

## 3. Python Trainer Refactor

### `train_only()` (trainer.py:31-107)

Compile once before the attempt loop:

```python
prog = pyxlog.IlpProgramFactory.compile(
    source, device=config.device, memory_mb=config.memory_mb,
    max_active_rules=config.max_active_rules,
)
```

For each attempt:
- Attempt 0: use `prog` as-is (already compiled and executed).
- Attempts 1+: call `prog.reset_runtime()` before entering the step loop.

### `_run_single_attempt()` (trainer.py:227-517)

Change signature to accept a pre-compiled `CompiledIlpProgram` instead of `source: str`. Remove the internal `compile()` call (line 244-247).

### Candidate handling: NOT cached across attempts

`valid_candidates()` and `candidate_triples_for_mask()` depend on executor store state (they read `has_tuples` from the store, lib.rs:4549). They are **not** purely plan-dependent. Therefore:

- Candidates must be re-enumerated per attempt after `reset_runtime()` (which re-executes the plan and repopulates the store).
- Only the compile artifacts (AST/plan/schemas/provider) are reused.
- Per-step candidate handling in the sparse backend remains unchanged.

---

## 4. Instrumentation

Run-level timing fields added to `_AttemptResult.telemetry_timings`:

| Field | Scope | Description |
|-------|-------|-------------|
| `compile_ms` | Once per `train_only()` | Wall-clock time for `IlpProgramFactory.compile()` |
| `reset_ms` | Per attempt (0 for attempt 0) | Wall-clock time for `prog.reset_runtime()` |

These aggregate in `TrainResult.artifact.telemetry.step_timings` as:
- `compile_ms_once`: The single compile time
- `reset_ms_total`: Sum of all reset times across attempts
- `reset_ms_p95`: 95th percentile of per-attempt reset times

---

## 5. GA Test Env Var Overrides

In `test_ilp_ga_reliability.py`, add two optional env vars alongside existing `GA_RELIABILITY_SEEDS`:

| Env var | Default | Purpose |
|---------|---------|---------|
| `GA_RELIABILITY_SEEDS` | 50 | Seed count (existing) |
| `GA_RELIABILITY_STEP_BUDGET` | 150 | Override `step_budget_per_attempt` |
| `GA_RELIABILITY_MAX_ATTEMPTS` | 7 | Override `max_attempts` |

These are profiling knobs. Canonical gate semantics (150/7/50 with CI ≥ 0.929) remain the default.

---

## 6. Tests

### 6a. Program-level parity

Verify that `compile() + reset_runtime() + evaluate` produces the same store state as a fresh `compile()`.

- Compile program A with source S, seed 42.
- Extract `relation_facts("target_rel")` → facts_A.
- Call `A.reset_runtime()`.
- Extract `relation_facts("target_rel")` → facts_A_reset.
- Compile program B (fresh) with same source S, seed 42.
- Extract `relation_facts("target_rel")` → facts_B.
- Assert facts_A_reset == facts_B.

### 6b. No state leak across attempts

- Compile once.
- Set mask with seed=0 values, evaluate → observe ILP tagged results.
- Call `reset_runtime()`.
- Verify ILP registry is empty (no masks, no last result).
- Set mask with seed=1 values, evaluate → results must reflect seed=1 state only.

### 6c. Train-level deterministic reproducibility

After refactor (compile-once + reset path is the only path):

- Run `train_only(source, ..., config=TrainConfig(deterministic=True, seed=42))` → result_1.
- Run `train_only(source, ..., config=TrainConfig(deterministic=True, seed=42))` → result_2.
- Assert `result_1.discovered_rule == result_2.discovered_rule`.
- Assert `result_1.converged == result_2.converged`.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Missed mutable state in reset | Low | Parity test (6a) catches; modeled on existing `evaluate()` plus ILP clear |
| GPU memory leak on repeated reset | Low | Provider is shared Arc, buffers re-created from schemas |
| Speedup insufficient (<2x) | Low | Instrumentation verifies; even partial compile elimination helps |
| Candidate ordering changes | Low | Candidates re-enumerated per attempt from fresh store state |
| Stale ILP masks after reset | Low | Explicit registry clear before plan re-execution |

---

## 8. Files

| File | Action |
|------|--------|
| `crates/pyxlog/src/lib.rs` | Add `reset_runtime()` to `CompiledIlpProgram` |
| `crates/xlog-runtime/src/executor.rs` | Add `reset_for_ilp()` (or extend `reset_for_mc()`) |
| `crates/pyxlog/python/pyxlog/ilp/trainer.py` | Refactor `train_only()` + `_run_single_attempt()` |
| `python/tests/test_ilp_ga_reliability.py` | Add env var overrides |
| `python/tests/test_ilp_reset.py` | New: parity + state leak + determinism tests |

---

## 9. Out of Scope

- Cross-call compile caching (LRU) — deferred unless instrumentation shows compile still >10-15% after this change.
- Step budget or max_attempts default changes — gate semantics unchanged.
- Any changes to the sparse/dense mask backend logic.
