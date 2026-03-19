# Strict Winner Result Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the actual learned winner directly on the strict ILP result surface so DTS can read the winning candidate/rule without scanning `candidate_map` or using compatibility export.

**Architecture:** Keep the strict relation-native path strict by reusing the existing `argmax_candidate_id_device` source of truth already tracked in the trainer. Materialize only the winner id and decoded rule metadata needed for the public strict result surface; do not materialize relation facts/examples and do not route through compatibility export.

**Tech Stack:** Python dataclasses in `pyxlog.ilp.types`, strict trainer/result assembly in `pyxlog.ilp.trainer`, pytest integration tests on CUDA-backed `pyxlog`.

### Task 1: Add RED tests for strict winner surfacing

**Files:**
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_sparse_guard.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`

**Step 1: Write the failing tests**

- Add a strict relation-native integration test that asserts `train_on_compiled_relations(...)` returns a `StrictTrainResult` with `winner_candidate_id` and `discovered_rule`.
- Add a deterministic strict relation-native test that runs twice under fixed seed and asserts identical winner metadata.
- Add a focused unit test that compares `_build_relation_native_strict_train_result(...)` output to the actual `argmax_candidate_id_device` from `_StrictAttemptState`.
- Update the D2H guard test that currently asserts strict results do not expose `discovered_rule`; replace it with an assertion that winner metadata is present while `export_compat_result()` is still unavailable on the relation-native strict path.

**Step 2: Run tests to verify they fail**

Run:
```bash
/home/dev/projects/xlog/.venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_d2h_gate.py -q
```

Expected: failure because `StrictTrainResult` does not yet expose winner metadata.

### Task 2: Implement strict winner fields on the public surface

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`

**Step 1: Extend strict dataclasses minimally**

- Add `winner_candidate_id: int | None = None` to `StrictTrainResult`.
- Add `discovered_rule: str | None = None` to `StrictTrainResult`.

**Step 2: Add strict helper(s)**

- Add a helper in `trainer.py` that derives winner metadata from:
  - `best_state.argmax_candidate_id_device`
  - `best_state.candidate_map`
  - `best_state.rel_names` / `best_state.candidates`
- Keep this path independent from compatibility export helpers.

**Step 3: Populate the strict result builders**

- Update `_build_relation_native_strict_train_result(...)` to set `winner_candidate_id` and `discovered_rule`.
- Update `_build_strict_train_result(...)` the same way so the generic strict path stays consistent.

### Task 3: Verify GREEN on targeted surfaces

**Files:**
- No new files

**Step 1: Run targeted strict tests**

Run:
```bash
/home/dev/projects/xlog/.venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_d2h_gate.py -q
```

Expected: PASS.

**Step 2: Rebuild native bindings**

Run:
```bash
maturin develop --release -m crates/pyxlog/Cargo.toml
```

Expected: build succeeds.

**Step 3: Run the recommended strict verification slice**

Run:
```bash
/home/dev/projects/xlog/.venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_credit_gpu.py -q
```

Expected: PASS with the strict relation-native winner surface exercised.

### Task 4: Commit

**Files:**
- Stage only the strict winner result implementation and tests

**Step 1: Commit**

Run:
```bash
git add crates/pyxlog/python/pyxlog/ilp/types.py crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_d2h_gate.py docs/plans/2026-03-19-strict-winner-result-surface.md
git commit -m "feat(ilp): expose strict winner metadata"
```
