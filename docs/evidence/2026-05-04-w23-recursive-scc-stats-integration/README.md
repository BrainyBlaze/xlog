# W2.3 Evidence — Recursive-SCC Stats Integration

**Closes board item: W2.3.**
**Date:** 2026-05-04
**Branch:** `feat/w23-recursive-scc-stats-integration`
**Base:** `main` at `d10bb72a` (W2.3 plan commit; on top of
`da644e3d` W2.1 closure-board commit).
**Plan:** `docs/plans/2026-05-04-w23-recursive-scc-stats-integration-plan.md`
(approved iteration 4).

## Summary

Wires unconditional per-iteration cardinality updates into
`Executor::execute_recursive_scc` per direction (b) of the W2.3
plan. Updates fire whether or not the iteration's body ran via
WCOJ, so the cost model on iteration N+1 always sees iteration
N's stats for the recursive predicate's full + delta RelIds.

## Step 1 Audit Findings

| Invariant | Status | Anchor |
|-----------|--------|--------|
| A1 — delta RelIds stats-registered | ✓ | `recursive.rs:279` → `mod.rs:335` |
| A2 — full RelIds stats-registered via Executor | ✓ | `mod.rs:332-336`. Compile-time `mgr` (compile.rs:175) is a separate manager not used here. |
| A3 — `name_to_rel_id` resolves heads | ✓ | New private accessor `mod.rs:339-348`, wraps `self.name_to_rel.get` |
| A4 — `cached_row_count` preservation | ✓ | `Executor::buffer_row_count` (`mod.rs:855-872`) returns cached or `dtoh_scalar_untracked` (metadata-plane). W2.3 reuses; no new D2H on data plane. |

No plan amendment needed.

## Acceptance Gate

10 tests in `crates/xlog-runtime/tests/test_w23_recursive_stats.rs`
(behind the `recursive-stats-trace` feature):

### Part A — Iteration-level cardinality evolution (3 tests)

```
test recursive_triangle_e1_full_card_grows_across_iterations ... ok
test recursive_triangle_e1_delta_evolves_across_iterations ... ok
test recursive_4cycle_e1_full_card_grows_across_iterations ... ok
```

* Triangle full: trace's `full_rows` for `pred == "e1"` is
  monotonically non-decreasing across iterations; strict `>` on at
  least one transition.
* Triangle delta: at least one pre-convergence `delta_rows` for
  `pred == "e1"` is non-zero AND the converged-iteration Phase 2
  record has `delta_rows == 0`.
* 4-cycle: same shape as triangle full, on the slice-4 4-cycle
  fixture.

### Part B — `binary_est_for_variant` reflects delta_e1 card (2 tests)

```
test triangle_binary_est_reflects_delta_e1_card_per_iteration ... ok
test cycle4_binary_est_reflects_delta_e1_card_per_iteration ... ok
```

For both fixtures, `pred == "e1"` rewrites `Scan(e1) → Scan(delta_e1)`;
the cost model's first-binary-hop estimate is
`estimate_join_cardinality(delta_e1, e2, &[1], &[0])` (joining
on `delta_e1.col1 = X = e2.col0`). The trace populates
`binary_est_for_variant` inline at each Phase 2 site (must be
inline; `delta_rel` is unregistered at fixpoint exit, so
post-`execute_plan` recomputation is impossible).

Each test asserts:
1. ≥ 2 Phase 2 trace entries with `binary_est_for_variant.is_some()`
   for `pred == "e1"`. Proves the cost-model lookup `(delta_e1, e2,
   &[1], &[0])` succeeded with both rels registered + cards
   populated.
2. ≥ 2 distinct `delta_rows` values across Phase 2 entries —
   proves the cost model's INPUT to the formula evolved (W2.3's
   `update_cardinality(delta_rel, delta_new_rows)` fired with
   different values per iteration).

The combination of (1) + (2) proves the cost model reads
iteration-current `delta_rel.cardinality`, not seed-only state.
Non-constancy of the formula's OUTPUT is **not** asserted: the
slice-4 fixtures' tiny cardinalities make the formula's
floor-of-1 dominate; W2.3's responsibility is at the input
layer (which Part A's `delta_rows`-evolves test already pins).

### Part C — Row-set + dispatch-counter parity (4 tests)

```
test recursive_triangle_row_set_unchanged_under_default_config ... ok
test recursive_triangle_dispatch_counter_unchanged_under_default_config ... ok
test recursive_4cycle_row_set_unchanged_under_default_config ... ok
test recursive_4cycle_dispatch_counter_unchanged_under_default_config ... ok
```

W2.3 must not perturb execution semantics. Each test runs the
slice-4 fixture with force-WCOJ-on + W2.3 wired AND with
force-WCOJ-off (binary-join reference); asserts the row sets
match bit-for-bit and the dispatch counter is ≥ 2 (matches
slice-4's "seed + ≥ 1 variant" assertion).

### Part D — Multi-recursive bodies untouched (1 test)

```
test multi_recursive_triangle_per_iteration_update_does_not_promote ... ok
```

Slice-4 multirec_triangle fixture pattern (`tri(X, Y, Z) :- r1(X, Y),
r2(Y, Z), r3(X, Z).`, with `r1` + `r2` both being recursive
IDBs in the SCC). Promoter's `recursive_scan_count <= 1` gate
refuses promotion. Counter == 0 across all iterations. W2.3's
per-iteration trace fires for `r1` / `r2` even when WCOJ is
gated out — predicate-level update, not promoter-level.

## Workspace Tally

```
cargo test --workspace --release --tests --exclude pyxlog
  Pre-W2.3 (main @ d10bb72a):     PASS=1914 FAIL=0
  Post-W2.3, default features:    PASS=1914 FAIL=0  (W2.3 tests skipped — required-features off)

cargo test --workspace --release --tests --exclude pyxlog \
  --features xlog-runtime/recursive-stats-trace
  Post-W2.3, trace feature on:    PASS=1924 FAIL=0  (+10 W2.3 tests)

cargo test -p xlog-cuda-tests --release --test certification_suite
  test run_full_certification ... ok
  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch
  6/6 PASS  (slice-4 cert preserved bit-identical)

cargo fmt --all -- --check
  clean
```

Slice 1–5 + W2.4 + W2.2 + W2.1 regression preserved
bit-identically under default features. Test delta:
**+10 W2.3 tests** under `recursive-stats-trace`, **0 tests
added** under default features (production zero overhead).

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-runtime/Cargo.toml` | New feature `recursive-stats-trace` (default OFF). New `[[test]]` block with `required-features = ["recursive-stats-trace"]` for `test_w23_recursive_stats`. New dev-dependency `xlog-logic` (no cycle; verified via grep). |
| `crates/xlog-runtime/src/executor/mod.rs` | New private `Executor::name_to_rel_id(name) -> Option<RelId>` accessor. New `RecursiveStatsTrace` + `RecursiveStatsTraceEntry` + `RecursiveStatsPhase` types, `Executor` field + accessor, all gated on `recursive-stats-trace` feature. |
| `crates/xlog-runtime/src/executor/recursive.rs` | `execute_recursive_scc` reset trace on entry (feature-gated). Seed pass: `update_cardinality(full_rel, full_new_rows)` + `update_cardinality(delta_rel, buffer_row_count(delta_initial))` with the actual delta_initial row count (NOT full). Fixpoint Phase 2: `update_cardinality(delta_rel, delta_new_rows)`. Fixpoint Phase 4: `update_cardinality(full_rel, full_new_rows_phase4)`. Trace pushes at Seed / Phase2Delta / Phase4Full sites, all feature-gated. `binary_est_for_variant` populated inline at Phase 2 for `pred == "e1"` via `estimate_join_cardinality(delta_e1, e2, &[1], &[0])` (must be inline; delta_rel is unregistered at fixpoint exit). |
| `crates/xlog-runtime/tests/test_w23_recursive_stats.rs` | New cert file: 10 acceptance tests (Part A 3 + Part B 2 + Part C 4 + Part D 1). |

## Decisions / Limitations

* **`recursive-stats-trace` Cargo feature, default OFF.** The
  trace seam (types + Executor field + populating call sites +
  accessor + the variable bindings the trace consumes) is gated
  on this feature. Production builds carry zero overhead — no
  field, no populating call site, no symbol. Tests that need
  the trace declare the feature in `required-features`.

* **Per-iteration update is unconditional** (direction (b)).
  Updates fire whether or not the iteration's body ran via
  WCOJ. W2.4's WCOJ-specific selectivity recording (`f586ce34`)
  remains complementary and untouched.

* **Stats writes survive the fixpoint exit** for the **full**
  RelId; the **delta** RelId is unregistered at fixpoint exit
  per `recursive.rs` cleanup. Tests that need to inspect the
  delta's cost-model contribution **must** read the trace
  (populated inline at Phase 2) or `binary_est_for_variant`
  (computed inline at Phase 2). Post-`execute_plan`
  recomputation against `delta_rel` is impossible.

* **`binary_est_for_variant` non-constancy is NOT asserted.**
  The slice-4 fixtures' tiny inputs (`delta_e1.cardinality ≤ 1`,
  `e2.cardinality == 2`) make `estimate_join_cardinality`'s
  `min == 1` clamp dominate, producing a constant `1` across
  iterations. W2.3's responsibility is at the **input** layer
  (cost model reads W2.3-updated `delta_rel.cardinality`),
  which Part A's `delta_rows`-evolves test pins. The output
  layer's behavior on small inputs is a property of the
  formula, not of W2.3.

* **No new D2H on data plane.** `Executor::buffer_row_count`
  (`mod.rs:855-872`) is the existing primitive that returns
  cached row count or falls back to `dtoh_scalar_untracked`
  (metadata-plane). W2.3 reuses this — no new metadata
  primitive, no new data-plane D2H.

* **`name_to_rel_id` returns None defensively.** Production
  callers register IDB heads before `execute_plan`; tests that
  omit registration get a no-op `update_cardinality` (which
  is itself a no-op for unregistered RelIds). Bit-identical
  to pre-W2.3 behavior under default features.

* **W2.6 unblocking.** With W2.3 DONE (pending user approval),
  no further blockers stand for W2.6 (W2.1 + W2.4 already DONE).
  The recursive-arm stats integration is the prerequisite for
  W2.6's heat/selectivity feedback into variable ordering.

## Process Rule Compliance

* **Process rule #1**: this slice does NOT self-mark W2.3 DONE.
  End-of-slice commit proposes the OPEN → DONE transition; user
  reviews + explicitly approves; a separate follow-up commit
  applies the board update.
* **Process rule #2**: every commit references W2.3. Commit
  list (chronological, `git log d10bb72a..HEAD` on the
  `feat/w23-recursive-scc-stats-integration` branch):
  1. `77f3b843` — W2.3 steps 1-6 (audit + name_to_rel_id +
     trace seam + seed pass + Phase 2 + Phase 4 wiring).
  2. `2b6caff7` — W2.3 step 7+8 (10 acceptance tests +
     `recursive-stats-trace` feature + warnings cleanup +
     fmt).
  3. *(this commit)* — W2.3 step 9 evidence README.

* **Process rule #3**: plan header opens with "Closes W2.3."
* **Process rule #5**: no `v0.6.6` references.
* **Process rule #6**: no push, no tag.

## Closure Board Update Proposal

After user review and explicit "mark W2.3 DONE" approval, a
follow-up commit applies:

* `docs/v065-closure-board.md` — W2.3 status `OPEN → DONE`,
  status tally updated (DONE: 3 → 4; OPEN: 16 → 15).
* `docs/v065-closure-board.md` "Completed" section gets a W2.3
  entry referencing the W2.3 commit list (plan + steps 1-6 +
  step 7+8 + this evidence README + the board-update commit
  itself).
* W2.5's `Blocked by` set narrows further: was
  `{W2.3, W3.2, W4.1, W5.1, W5.2}` (post-W2.1); becomes
  `{W3.2, W4.1, W5.1, W5.2}` (post-W2.3).
