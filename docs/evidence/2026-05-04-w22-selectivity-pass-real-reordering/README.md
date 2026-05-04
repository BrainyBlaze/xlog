# W2.2 Evidence — Real `selectivity_pass` Join Reordering

**Closes board item: W2.2.**
**Date:** 2026-05-04
**Branch:** `feat/w22-selectivity-pass-real-reordering`
**Base:** `main` at `eea74612` (W2.2 plan commit).
**Plan:** `docs/plans/2026-05-04-w22-selectivity-pass-real-reordering-plan.md`

## Summary

Replaces slice 3's no-op `xlog_logic::optimizer::selectivity_pass`
with a real selectivity-driven join reordering pass for canonical
lowered triangle and 4-cycle bodies. Variable-graph deduction
identifies semantic atom roles regardless of positional layout;
pair-derived join keys (NOT fixed `[1]/[0]`) drive the
cardinality lookups. Promoter (`try_promote_triangle` /
`try_promote_4cycle`) extended to recognize alternative
inner-key shapes via the same variable-graph approach, emitting
`MultiWayJoin.inputs` and `slot_vars` in canonical semantic
order regardless of body positional layout.

## Acceptance Properties

### Part A — Compile-time RIR shape (8 tests, all pass)

```
cargo test -p xlog-logic --release --lib selectivity_pass
running 8 tests
test optimizer::selectivity_pass_tests::selectivity_pass_picks_y_shared_inner_when_e1_e2_smallest ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_picks_x_shared_inner_when_e1_e3_smallest ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_picks_z_shared_inner_when_e2_e3_smallest ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_two_snapshots_produce_different_inner_pairs ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_with_only_relation_cards_may_pick_arbitrary_pair ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_is_noop_for_triangle_plan ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_is_noop_for_4cycle_plan ... ok
test optimizer::selectivity_pass_tests::selectivity_pass_is_noop_for_recursive_scc ... ok
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured
```

* The three "picks_*_inner" tests verify all three triangle
  inner-pair choices (Y / X / Z shared) are driven by stats.
* The "two_snapshots" test pins "stats drive the order, not
  deterministic canonicalization." Deterministic canonicalization
  CANNOT pass this gate.
* The "only_relation_cards" test documents the 10% default-
  fallback edge case explicitly (tolerant by design).

### Part A' — Promoter semantic-slot inference (26 tests, all pass)

The W2.2 promoter extension (step 2a) is exercised by 26
tests in `crates/xlog-logic/src/promote.rs::tests`, including
3 new alternative-shape tests:

  * `promotes_triangle_with_x_shared_inner_pair` — inner keys
    `[0]/[0]`, outer keys `[1, 3]/[0, 1]`, project `[0, 1, 3]`.
    Asserts inputs reordered to canonical semantic order
    `[XY, YZ, XZ]` and shape-fixed `slot_vars`.
  * `promotes_triangle_with_z_shared_inner_pair` — inner keys
    `[1]/[1]`, outer keys `[0, 2]/[0, 1]`, project `[0, 2, 3]`.
  * `promotes_4cycle_with_alternative_inner_grouping` —
    `(e2⋈e3 on Y) + (e4⋈e1 on W)`, project `[5, 0, 1, 3]`.
    Asserts inputs reordered to `[WX, XY, YZ, ZW]`.

### Part B — End-to-end row-set parity (1 test, pass)

```
cargo test -p xlog-integration --release --test test_selectivity_pass_reordering
test selectivity_pass_triangle_two_snapshots_produce_same_row_set ... ok
```

Same source compiled twice via
`Compiler::compile_with_stats_snapshot` with two distinct
stats snapshots favoring different inner pairings. Row sets
after `execute_plan` are IDENTICAL — reordering preserves
rule semantics.

### Part C — WCOJ dispatch survives W2.2 changes (2 tests, pass)

```
test selectivity_pass_changes_do_not_break_canonical_triangle_dispatch ... ok
test selectivity_pass_changes_do_not_break_canonical_4cycle_dispatch ... ok
```

Force-WCOJ on canonical left-deep / bushy bodies — counter ≥ 1
AND row-set match vs binary-join reference. W2.2 changes
(promoter extension + selectivity_pass real logic) don't break
dispatch on the canonical case.

**Honest scope note** (also in test file's header): Part C as
originally framed wanted to drive selectivity_pass to an alt
shape end-to-end and confirm WCOJ dispatch on the alt. The
optimizer can emit right-deep `Project { Join { Scan, Join } }`
when stats favor it; right-deep is explicitly OUT of W2.2
scope per plan ("Right-deep input handling … is a separate
slice's input"), so the W2.2 promoter doesn't recognize it.
Reordering itself is exercised end-to-end by Part A (compile-
time direct plan synthesis) and Part A' (promoter-extension
unit tests on alt shapes); the canonical-case Part C cert is
a regression-style guard that W2.2 changes don't break the
existing dispatch path.

## Workspace Tally

| Crate | PASS | FAIL |
|-------|------|------|
| `xlog-runtime` | 135 | 0 |
| `xlog-cuda` | 507 | 0 |
| `xlog-logic` | 512 | 0 |
| `xlog-integration` | 131 | 0 |
| `xlog-cuda-tests` (cert) | 1 (full pass) | 0 |

Slice 1–5 + W2.4 regression preserved. xlog-logic count went
503 → 512 (+9 from W2.2: 3 step-2a promoter alt-shape tests +
5 step-3 selectivity_pass tests + 1 renamed test from "rejects
non-triangle proj" to "promotes triangle with rotated proj"
under W2.2's relaxed contract).

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-logic/src/optimizer.rs` | `selectivity_pass::run` signature `(plan, stats, rel_ids)`; module-level docs; new `mod reorder` with triangle + 4-cycle rewriters using variable-graph deduction + pair-derived keys; build helpers for 3 triangle shapes + 2 4-cycle groupings; safety floor on missing/zero card; 5 new unit tests + helpers. |
| `crates/xlog-logic/src/compile.rs` | Caller passes `self.lowerer.rel_ids()`. |
| `crates/xlog-logic/src/promote.rs` | `try_promote_triangle` / `try_promote_4cycle` extended via `infer_triangle_semantics` / `infer_4cycle_semantics` — variable-graph deduction recognizes any valid key combination, emits canonical-order `MultiWayJoin.inputs` + shape-fixed `slot_vars`. 3 new tests + 1 reframed test for alternative shapes. |
| `crates/xlog-integration/Cargo.toml` | Added `xlog-stats` dep for cert. |
| `crates/xlog-integration/tests/test_selectivity_pass_reordering.rs` | New cert file — 3 integration tests (Part B + Part C ×2). |

## Decisions / Limitations

* **Right-deep optimizer output is out of W2.2 scope** (per
  plan). When the optimizer emits right-deep, selectivity_pass
  is a no-op on it and the W2.2-extended promoter doesn't
  recognize it either. Pre-existing slice 1 / slice 2 behavior
  on right-deep stays unchanged.
* **10% default-fallback case** in
  `StatsManager::estimate_join_cardinality` is documented and
  pinned by a tolerant unit test. The pass cannot detect when
  the fallback is in use; row-set parity gates correctness
  regardless of selectivity quality.
* **No-op detection** uses Debug-string comparison
  (`format!("{:?}", body)`) because `RirNode` doesn't impl
  `PartialEq`. Bodies are tiny so cost is negligible.
* **`rel_ids` parameter** on `selectivity_pass::run` is
  reserved for future shape-extension work; current
  rewriters operate on RelIds directly.

## Process Rule Compliance

* Process rule #1: this slice does NOT self-mark W2.2 DONE.
  End-of-slice commit proposes the OPEN → DONE transition
  in the commit message; user reviews and explicitly
  approves; a separate follow-up commit applies the board
  update.
* Process rule #2: every commit references W2.2.
* Process rule #3: plan header opens with "Closes W2.2."
* Process rule #5: no `v0.6.6` references in this slice.
* Process rule #6: no push, no tag.

## Closure Board Update Proposal

After user review and explicit "mark W2.2 DONE" approval, a
follow-up commit applies:

* `docs/v065-closure-board.md` — W2.2 status `OPEN → DONE`,
  status tally updated (DONE: 1 → 2; OPEN: 17 → 16).
* `docs/v065-closure-board.md` "Completed" section gets a
  W2.2 entry.
