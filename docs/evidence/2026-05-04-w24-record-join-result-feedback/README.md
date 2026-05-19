# W2.4 Evidence — `record_join_result` Feedback from WCOJ Dispatch

**Closes board item: W2.4 only.**
**Date:** 2026-05-04
**Branch:** `feat/w24-record-join-result-feedback`
**Base:** `main` at `55702bb8` (W1.1 commit).
**Plan:** `docs/plans/2026-05-04-w24-record-join-result-feedback-plan.md`

## Summary

Successful WCOJ dispatches (triangle + 4-cycle) now call
`xlog_stats::StatsManager::record_join_result(...)` so observed
selectivity flows back into the stats cache. The slice 5
cardinality cost model already consumes stats; W2.4 closes the
loop by writing them back.

## Acceptance Properties (per the corrected gate)

| # | Property | How proven |
|---|----------|------------|
| 1 | Dispatch path actually calls `record_join_result` | `triangle_dispatch_records_join_result_into_stats_manager` and `cycle4_dispatch_records_join_result_into_stats_manager` assert `get_join_selectivity(slot_a, slot_b)` transitions `None → Some(_)` after `execute_plan`. The transition cannot be a false positive from fixture seeding because test-side `update_cardinality` doesn't touch the selectivity cache. |
| 2 | `binary_est` differs between runs | Both certs read `executor.stats().estimate_join_cardinality(slot_a, slot_b, &[1], &[0])` — the same call the cardinality cost model uses. After two consecutive `execute_plan` runs on the same executor, `assert_ne!(binary_est_run1, binary_est_run2)`. EMA averaging in `record_join_result` moves the cached selectivity, which moves the estimate. |
| 3 | Row-set parity unchanged across runs | `download_triples` / `download_quads` from the recursive store after run 1 equals run 2 — the recursive fixpoint converges to the same set on warm vs. cold stats. |
| (negative) | Helper skips when input cards are missing | `wcoj_dispatch_does_not_record_when_input_cards_missing` — force-mode dispatch (counter advances) but `get_join_selectivity` stays `None` because `record_wcoj_feedback` early-returns on missing input cardinalities. |

## Cert Test Results

```
cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback
running 3 tests
test wcoj_dispatch_does_not_record_when_input_cards_missing ... ok
test triangle_dispatch_records_join_result_into_stats_manager ... ok
test cycle4_dispatch_records_join_result_into_stats_manager ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured
```

## Workspace Tally

| Crate | PASS | FAIL | IGN |
|-------|------|------|-----|
| `xlog-runtime` (full) | 135 | 0 | 2 |
| `xlog-cuda` (full) | 507 | 0 | 6 |
| `xlog-integration` (full) | 128 | 0 | 0 |
| `xlog-cuda-tests` (cert suite) | 1 (cert pass) | 0 | 0 |

Slice 1–5 regression preserved bit-identical. The
xlog-integration count went 123 → 128 (+5) by adding 3 W2.4
certs and the existing slice 5 cardinality file's tests now
share helpers; both unaffected.

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | Two new `pub(super)` helpers on `Executor`: `wcoj_output_rows(buf) -> Option<u64>` (widens `cached_row_count()`'s `Option<u32>`; never invents `Some(0)` from `None`) and `record_wcoj_feedback(slot_rels, output_rows)` (early-returns on missing output count or missing input cards; calls `stats.record_join_result` with owned `vec![1]/vec![0]` keys). |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | Triangle and 4-cycle success arms wire `record_wcoj_feedback` BEFORE the counter increment. Single line per call site. |
| `crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs` | New cert file — 3 tests (triangle, 4-cycle, missing-cards-no-record). |

## Decision Mapping (documented at the call sites)

The triangle / 4-cycle output is a strict subset of the inner-
join intermediate (the third / additional atoms further filter
it). So `output_rows ≤ true_intermediate_rows`, which means
the recorded selectivity is **≤** the true binary selectivity.
For the cardinality cost model, a too-low selectivity gives a
too-low `binary_est`, which makes WCOJ less likely on the next
call — the correct conservative direction (don't over-claim
the kernel's win).

Recording an observed-empty output (`Some(0)`) IS correct: the
EMA tightens future estimates toward zero, so WCOJ becomes
less likely on the same inputs (the kernel produced nothing
useful). Recording an unknown output (`None` from
`cached_row_count`) is forbidden by the `record_wcoj_feedback`
early-return — unknown row counts must never become a `0`
selectivity record.

## Process Rule Compliance

* Process rule #1: this slice does NOT self-mark W2.4 DONE.
  The end-of-slice commit proposes the OPEN → DONE transition
  in the commit message; the user reviews and explicitly
  approves; a separate follow-up commit applies the board
  update.
* Process rule #2: every commit references W2.4.
* Process rule #3: plan header opens with "Closes W2.4 only."
* Process rule #5: no `v0.6.6` references in this slice.
* Process rule #6: no push, no tag.

## Closure Board Update Proposal

After user review and explicit "mark W2.4 DONE" approval, a
follow-up commit applies:

* `docs/v065-closure-board.md` — W2.4 status `OPEN → DONE`,
  status tally updated (DONE: 0 → 1; OPEN: 18 → 17).
* `docs/v065-closure-board.md` "Completed" section gets a W2.4
  entry with the commit hashes that delivered it.
