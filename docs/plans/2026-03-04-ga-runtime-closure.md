# GA Runtime Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `python/tests/test_ilp_ga_reliability.py::test_ga_reliability_50` wall-clock runtime from ~1447s to <=600s while preserving statistical reliability (`Clopper-Pearson lower95 >= 0.929`).

**Architecture:** Use a measurement-first approach. First add per-step timing breakdown in the trainer to identify dominant cost centers. Then remove avoidable per-step duplication in Python, add preloaded fact-buffer APIs in Rust to eliminate repeated fact upload/marshaling, and run a controlled GA budget sweep to select the fastest config that still passes the statistical gate.

**Tech Stack:** Python (`pyxlog.ilp.trainer`), Rust/PyO3 (`crates/pyxlog/src/lib.rs`), runtime (`crates/xlog-runtime`), CUDA provider (`xlog-cuda`), pytest.

**Baseline Evidence:** `docs/evidence/2026-03-04-v0.4.0-beta/` and commits `09c2942c`, `6f697fe0`.

---

### Task 1: Add Step-Phase Timing Breakdown (Measurement First)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_trainer.py`

**Step 1: Add phase timers in the step loop**

Instrument these phases with `time.perf_counter()` (CPU) and store microseconds:
- `apply_mask_us`
- `evaluate_us` (already timed with CUDA events; keep as source of truth)
- `loss_credit_us`
- `loss_reduce_us`
- `membership_us`
- `convergence_us`
- `backward_step_us`

Store p95/mean totals in `result.telemetry_timings`.

**Step 2: Add regression test for phase keys**

In `python/tests/test_ilp_trainer.py`, assert timing keys are present and non-negative after a short run.

**Step 3: Run tests**

Run:
```bash
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_trainer.py -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py
git commit -m "perf(ilp): add per-step phase timing breakdown to trainer telemetry"
```

---

### Task 2: Move Static Fact Grouping Out of Step Loop

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_trainer.py`

**Step 1: Precompute grouped facts once per attempt**

Before entering `for step in range(step_budget)`, precompute:
- `pos_by_rel`, `neg_by_rel`
- per-relation index maps for witness reconstruction

Pass these into loss/witness/convergence helpers instead of rebuilding every step.

**Step 2: Keep functional behavior identical**

Do not change loss formulas or convergence gates.

**Step 3: Run tests**

Run:
```bash
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_trainer.py python/tests/test_ilp_reliability.py -v --timeout=1800
```
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py
git commit -m "perf(ilp): hoist static fact grouping out of trainer hot loop"
```

---

### Task 3: Eliminate Duplicate Membership Calls in-Step

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`
- Modify: `python/tests/test_ilp_trainer.py`

**Step 1: Cache per-step membership results**

Compute membership masks once per relation per step and reuse for:
- witness coverage
- convergence training-mask gate(s)

When `stable_count < 5`, skip convergence membership checks entirely.

**Step 2: Keep argmax-only verification semantics intact**

Argmax-only re-evaluation can still run separate checks; only remove duplicate calls in the same step context.

**Step 3: Add regression test**

Add a test that monkeypatches membership API wrapper and asserts call-count reduction for one step.

**Step 4: Run tests**

Run:
```bash
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_trainer.py python/tests/test_ilp_d2h_gate.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_trainer.py
git commit -m "perf(ilp): reuse per-step membership masks across witness and convergence paths"
```

---

### Task 4: Single Credit Query per Relation per Step

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_trainer.py`

**Step 1: Merge pos/neg credit fetches per relation**

If both positive and negative facts exist for a relation:
- call `batch_tagged_credit(rel, merged_facts)` once
- split returned credits by index ranges

This avoids duplicate GPU query uploads and per-call overhead.

**Step 2: Validate loss equivalence**

Add a unit test that compares old and new loss on a deterministic tiny case (same seed/config), asserting near-equality tolerance.

**Step 3: Run tests**

Run:
```bash
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_trainer.py -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py
git commit -m "perf(ilp): merge positive/negative credit queries per relation"
```

---

### Task 5: Add Preloaded Fact Buffer APIs in PyO3

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `python/tests/test_ilp_sparse.py`
- Create: `python/tests/test_ilp_preloaded_buffers.py`

**Step 1: Add API methods**

Add methods on `CompiledIlpProgram`:
- `upload_fact_buffer(relation: str, facts: list[list[int]]) -> int`
- `batch_fact_membership_preloaded(handle: int) -> list[bool]`
- `batch_tagged_credit_preloaded(relation: str, handle: int) -> list[list[tuple[int,int,int]]]`
- `drop_fact_buffer(handle: int) -> None`

**Step 2: Use schema-aware encoding**

Build upload buffer with schema-driven encoding (`push_term_bytes` path), not raw `u32` casts.

**Step 3: Add handle safety**

Reject invalid/stale handles with clear `PyValueError`.

**Step 4: Add tests**

`python/tests/test_ilp_preloaded_buffers.py`:
- upload + membership parity with non-preloaded API
- upload + credit parity with non-preloaded API
- drop handle invalidates future use

**Step 5: Run tests**

Run:
```bash
cargo build -p pyxlog --release
cd crates/pyxlog && ../../.venv/bin/maturin develop --release
cd ../..
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_preloaded_buffers.py python/tests/test_ilp_sparse.py -v
```
Expected: PASS.

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_preloaded_buffers.py python/tests/test_ilp_sparse.py
git commit -m "feat(ilp): add preloaded fact-buffer APIs for hot-loop reuse"
```

---

### Task 6: Use Preloaded Buffers in Trainer Hot Loop

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`

**Step 1: Preload facts once per attempt**

In `_run_single_attempt`, upload positive/negative fact buffers per relation once before step loop.

**Step 2: Replace non-preloaded calls**

Use preloaded variants in:
- loss computation
- witness coverage
- convergence checks

**Step 3: Ensure cleanup**

Use `try/finally` to call `drop_fact_buffer()` for all handles.

**Step 4: Add test coverage**

Add test asserting preloaded path returns identical results to non-preloaded path.

**Step 5: Run tests**

Run:
```bash
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest \
  python/tests/test_ilp_trainer.py python/tests/test_ilp_d2h_gate.py -v
```
Expected: PASS.

**Step 6: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py python/tests/test_ilp_d2h_gate.py
git commit -m "perf(ilp): switch trainer hot loop to preloaded fact buffers"
```

---

### Task 7: Add Runtime Sweep Harness for GA Budget Search

**Files:**
- Modify: `python/tests/test_ilp_ga_reliability.py`
- Create: `docs/evidence/2026-03-05-ga-runtime-closure/README.md`

**Step 1: Add profiling output fields**

In GA test summary output, print:
- `total_wall_s`
- per-stage success rates
- configured `step_budget/max_attempts`

**Step 2: Define sweep matrix**

Evaluate:
- `(150, 7)` baseline
- `(120, 6)`
- `(100, 5)`
- `(80, 5)`
- `(60, 3)`

using env vars:
- `GA_RELIABILITY_STEP_BUDGET`
- `GA_RELIABILITY_MAX_ATTEMPTS`

**Step 3: Document sweep commands**

Add exact commands and output capture in evidence README.

**Step 4: Commit**

```bash
git add python/tests/test_ilp_ga_reliability.py docs/evidence/2026-03-05-ga-runtime-closure/README.md
git commit -m "test(ilp): add GA reliability runtime sweep instrumentation and evidence scaffold"
```

---

### Task 8: Select Fastest Config That Preserves Statistical Gate

**Files:**
- Modify: `python/tests/test_ilp_ga_reliability.py` (if default updates are warranted)
- Modify: `CHANGELOG.md` (if gate profile changes)
- Modify: `docs/ROADMAP.md` (if target achieved)

**Step 1: Run sweep and choose winning config**

Selection rule:
- must satisfy `lower95 >= 0.929`
- choose smallest wall-clock

**Step 2: Decide default behavior**

If winner is not `(150,7)`, either:
- keep defaults and only use env in CI, or
- update defaults with explicit changelog note.

**Step 3: Validate chosen config**

Run the chosen config once with `GA_RELIABILITY_SEEDS=50` and save logs.

**Step 4: Commit**

```bash
git add python/tests/test_ilp_ga_reliability.py CHANGELOG.md docs/ROADMAP.md
git commit -m "perf(ilp): adopt validated GA reliability runtime profile"
```

---

### Task 9: Final Verification and Closeout Evidence

**Files:**
- Create: `docs/reports/2026-03-05-ga-runtime-closure-report.md`
- Update: `docs/evidence/2026-03-05-ga-runtime-closure/` (logs)

**Step 1: Run final validation set**

Run:
```bash
cargo test --workspace --all-targets --exclude pyxlog --release
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=1800
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v --timeout=3600
PYTHONPATH=crates/pyxlog/python .venv/bin/python -m pytest python/tests/test_ilp_performance.py -v --timeout=1200
```

**Step 2: Produce report**

Include:
- before vs after wall-clock
- per-phase timing deltas
- selected GA budget profile
- reliability CI bound
- whether `<=600s` target was achieved

**Step 3: Commit**

```bash
git add docs/reports/2026-03-05-ga-runtime-closure-report.md docs/evidence/2026-03-05-ga-runtime-closure/
git commit -m "evidence(ilp): GA runtime closure validation and final report"
```

---

## Acceptance Criteria

1. `test_ga_reliability_50` passes with `lower95 >= 0.929`.
2. Runtime reduced from ~1447s toward target `<=600s` (or a quantified residual gap is reported with measured bottlenecks).
3. No regression in:
   - `python/tests/test_ilp_reliability.py`
   - `python/tests/test_ilp_sparse.py`
   - `python/tests/test_ilp_performance.py`
4. Evidence and report files are committed and reproducible.

---

## Notes

- This plan is intentionally measurement-first. Do not make sweeping runtime changes before Task 1 data is available.
- Keep gate semantics statistically defensible. Runtime improvements must not silently weaken reliability thresholds.
- If `<=600s` is still not reached after Task 8, open a follow-up plan focused on deeper Rust-side credit aggregation (single-kernel per relation).

