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

### Part A — Compile-time RIR shape (12 tests, all pass)

```
cargo test -p xlog-logic --release --lib selectivity_pass
running 12 tests
... (3 triangle inner-pair choices + triangle 2-snapshots-differ
     + triangle fallback-tolerant + 3 pre-existing no-ops
     + 4 4-cycle: 2 grouping choices + 2-snapshots-differ + missing-card)
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured
```

Triangle (5 tests):
* `selectivity_pass_picks_y_shared_inner_when_e1_e2_smallest`
* `selectivity_pass_picks_x_shared_inner_when_e1_e3_smallest`
* `selectivity_pass_picks_z_shared_inner_when_e2_e3_smallest`
* `selectivity_pass_two_snapshots_produce_different_inner_pairs`
* `selectivity_pass_with_only_relation_cards_may_pick_arbitrary_pair`

4-cycle (4 tests):
* `selectivity_pass_4cycle_picks_default_grouping_when_corners_smallest`
* `selectivity_pass_4cycle_picks_alt_grouping_when_diagonals_smallest`
* `selectivity_pass_4cycle_two_snapshots_produce_different_groupings`
* `selectivity_pass_4cycle_skips_when_card_missing`

Pre-existing no-ops preserved (3): empty stats → safety floor leaves
bodies unchanged.

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

### Part B — End-to-end row-set parity (2 tests, pass)

```
test selectivity_pass_triangle_two_snapshots_produce_same_row_set ... ok
test selectivity_pass_4cycle_two_snapshots_produce_same_row_set ... ok
```

Same source compiled twice via
`Compiler::compile_with_stats_snapshot` with two distinct
stats snapshots favoring different inner pairings (triangle)
or different bushy groupings (4-cycle). Row sets after
`execute_plan` are IDENTICAL — reordering preserves rule
semantics.

### Part C — Force-WCOJ on synthesized post-selectivity bodies (4 tests, pass)

```
test selectivity_pass_synthesized_x_shared_triangle_dispatches_wcoj ... ok
test selectivity_pass_synthesized_alt_grouping_4cycle_dispatches_wcoj ... ok
test selectivity_pass_changes_do_not_break_canonical_triangle_dispatch ... ok
test selectivity_pass_changes_do_not_break_canonical_4cycle_dispatch ... ok
```

The "synthesized" Part C tests **directly close the
acceptance gate** for non-default reordered bodies. They
build a hand-crafted alt-shape lowered RIR (X-shared
triangle / Alt-grouping 4-cycle), feed through
`xlog_logic::promote::promote_multiway` (W2.2 step 2a
extension), then run the executor with force-WCOJ. Counter
≥ 1 AND row set equals binary-join reference.

This path is the W2.2 plan's explicit fallback: "If
compile_with_stats_snapshot currently drives the optimizer
into right-deep output, … build the integration cert from
a synthesized post-selectivity plan."

The two "do_not_break_canonical_*" tests are regression
guards on the canonical case — slice 1 / slice 2 dispatch
behavior unchanged after W2.2.

A dispatch matcher relaxation in `wcoj_dispatch.rs` accepts
the alt output_columns layouts:
* Triangle: `[Column(0), Column(c), Column(3)]` where `c ∈
  {1, 2}` (Y/X-shared = `1`, Z-shared = `2`).
* 4-cycle: `[Column(0), Column(1), Column(3), Column(5)]`
  (Default) or `[Column(5), Column(0), Column(1), Column(3)]`
  (Alt).

## Workspace Tally

| Crate | PASS | FAIL |
|-------|------|------|
| `xlog-runtime` | 135 | 0 |
| `xlog-cuda` | 507 | 0 |
| `xlog-logic` | 516 | 0 |
| `xlog-integration` | 134 | 0 |
| `xlog-cuda-tests` (cert) | 1 (full pass) | 0 |

Slice 1–5 + W2.4 regression preserved. xlog-logic count went
503 → 516 (+13 from W2.2: 3 step-2a promoter alt-shape tests
+ 5 step-3 triangle selectivity_pass tests + 4 step-3 4-cycle
selectivity_pass tests + 1 renamed test). xlog-integration
went 128 → 134 (+6: 2 Part B + 2 Part C synthesized + 2
Part C canonical-regression).

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-logic/src/optimizer.rs` | `selectivity_pass::run` signature `(plan, stats, rel_ids)`; module-level docs; new `mod reorder` with triangle + 4-cycle rewriters using variable-graph deduction + pair-derived keys; build helpers for 3 triangle shapes + 2 4-cycle groupings; safety floor on missing/zero card; 5 new unit tests + helpers. |
| `crates/xlog-logic/src/compile.rs` | Caller passes `self.lowerer.rel_ids()`. |
| `crates/xlog-logic/src/promote.rs` | `try_promote_triangle` / `try_promote_4cycle` extended via `infer_triangle_semantics` / `infer_4cycle_semantics` — variable-graph deduction recognizes any valid key combination, emits canonical-order `MultiWayJoin.inputs` + shape-fixed `slot_vars`. 3 new tests + 1 reframed test for alternative shapes. |
| `crates/xlog-integration/Cargo.toml` | Added `xlog-stats` dep for cert. |
| `crates/xlog-integration/tests/test_selectivity_pass_reordering.rs` | New cert file — 6 integration tests: Part B triangle + 4-cycle (2); Part C synthesized post-selectivity X-shared triangle + Alt-grouping 4-cycle (2); Part C canonical-regression triangle + 4-cycle (2). |

## Decisions / Limitations

* **Right-deep optimizer output is out of W2.2 scope** (per
  plan). When the optimizer emits right-deep, selectivity_pass
  is a no-op on it and the W2.2-extended promoter doesn't
  recognize it either. Pre-existing slice 1 / slice 2 behavior
  on right-deep stays unchanged. Part C uses the W2.2 plan's
  explicit fallback path (synthesized post-selectivity body)
  to close the acceptance gate end-to-end.
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
