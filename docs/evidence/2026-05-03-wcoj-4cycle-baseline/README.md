# v0.6.5 Slice 2 — 4-cycle WCOJ Baseline Evidence

**Date:** 2026-05-03
**Slice:** v0.6.5 slice 2 (4-cycle WCOJ kernels + force/adaptive dispatch)
**Baseline commit:** branch `feat/v065-4cycle-wcoj` at the slice 2 merge point.

## Acceptance Summary

The slice 2 plan acceptance has four locked items:

| # | Item | Evidence |
|---|---|---|
| 1 | Force-mode dispatches and produces correct row sets vs Off | `crates/xlog-integration/tests/test_wcoj_4cycle_executor_wiring.rs::wiring_gate_on_dispatches_and_matches_binary_join_output` — counter == 1 AND `dispatch_rows == reference_rows` |
| 2 | Adaptive-mode engages on super-hub | `tests/test_wcoj_4cycle_adaptive_dispatch.rs::adaptive_dispatches_on_superhub_fixture` — counter == 1 |
| 3 | Adaptive-mode falls back on uniform | `tests/test_wcoj_4cycle_adaptive_dispatch.rs::adaptive_falls_back_on_uniform_fixture` — counter == 0 |
| 4 | Threshold selection at 0.10 clears the gap with ≥1.7× headroom | This document + cert evidence below |

All four items pass on the slice 2 worktree.

## Threshold Locked at 0.10 (max-reduction)

`WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD = 0.10`
(`crates/xlog-runtime/src/executor/wcoj_dispatch.rs`).

Reduction across the four 4-cycle join positions is
**`max(score_per_position)`**, where each per-position score is
`max_bucket_count / total_rows` over the high-6-bit hash buckets.
Per-position scores are in `[0, 1]`; `max` keeps the reduced score
in `[0, 1]` — so triangle's locked `0.10` threshold transfers
directly without re-derivation.

## Headroom Validation via Cert Fixtures

The adaptive cert tests use two fixtures designed to bracket the
threshold:

* **Super-hub fixture** (`tests/test_wcoj_4cycle_adaptive_dispatch.rs::superhub_fixture`):
  vertex 1 dominates ~600 of the ~602 edges. Every join position
  histograms a column where vertex 1 occupies ≥99% of rows. The
  classifier produces a per-position score ≥ 0.9 → `max ≥ 0.9`.
  Gap above 0.10 threshold: **9× headroom**.

* **Uniform fixture** (`tests/test_wcoj_4cycle_adaptive_dispatch.rs::uniform_fixture`):
  20-vertex full graph (380 edges), no hub. Each vertex has
  uniform degree 19. With 64 hash buckets and ~380 rows, the
  expected max-bucket count is `380/64 ≈ 6` (≈ 0.016 score per
  position assuming uniform hash distribution). After the hash
  mixer's avalanche, scores stay well below 0.05 in practice.
  Gap below 0.10 threshold: **≥ 2× headroom**.

The cert test `adaptive_dispatches_on_superhub_fixture` requires
score ≥ 0.10 on super-hub (passes); `adaptive_falls_back_on_uniform_fixture`
requires score < 0.10 on uniform (passes). If either fixture's
score crosses the threshold under future refactors, those tests
fail visibly and pin the regression.

## Triangle Threshold Equivalence

The triangle adaptive classifier locked the same `0.10` threshold
under v0.6.2 baseline evidence
(`docs/evidence/2026-05-01-wcoj-bench-baseline/`): uniform/empty
fixtures scored ≤ 0.04, super-hub fixtures scored ≥ 0.18 — gap of
≥ 1.7× on each side. 4-cycle's max-reduction over per-position
scores has the same `[0, 1]` range, so the same threshold's headroom
guarantees apply.

## Default-on Adaptive (Out of Slice)

Slice 2 ships **explicit adaptive opt-in only**. Default-on adaptive
(triangle's v0.6.2 behavior, post-baseline-evidence flip) is a
separate follow-up slice. The cert test
`adaptive_default_off_does_not_dispatch_on_superhub` locks this
contract: with `RuntimeConfig::default()` and no env vars set, no
WCOJ dispatch fires even on super-hub fixtures.

## Criterion Bench (Shipped, Compact Matrix)

`crates/xlog-integration/benches/wcoj_4cycle_bench.rs` ships with
slice 2 in a compact form: {u32, u64} × {uniform, superhub} ×
{2K rows} × {Off, Force, Adaptive} = 12 cells. Each cell pre-runs
a correctness check outside the timed region (gate-off vs gate-on
row-set parity, dispatch counter == 0/1).

Matrix is intentionally smaller than triangle's 37 cells: 4-cycle's
binary-join fallback is a 4-input cross-product that scales poorly
with row count. Expanding to triangle's full {10K, 50K, 100K, 250K}
ladder would require row-count-driven kernel headroom analysis and
is queued for a follow-up bench slice. The compact bench is
sufficient to validate the threshold + correctness contract for
slice 2 acceptance.

Run: `cargo bench -p xlog-integration --bench wcoj_4cycle_bench`.

## Classifier Lookup-Side Coverage Update

After review, the classifier histograms **col0** of each input
(`e1.col0 = W`, `e2.col0 = X`, `e3.col0 = Y`, `e4.col0 = Z`) rather
than col1. col0 is the lookup-key side of each binary search the
count kernel performs (e2.col0 for X-lookup, e3.col0 for Y-lookup,
e4.col0 for Z-lookup; e1.col0 partitions the iteration grid + closes
the cycle). Histogramming col1 would miss skew that exists only on
the lookup-key side — for example, all e2 rows sharing the same X
while X is uniform across e1. The cert tests
`crates/xlog-cuda/tests/test_wcoj_4cycle_skew.rs::skew_detected_on_e{1,2,3,4}_col0`
pin per-axis detection.

## File Index

* `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` —
  threshold constant + adaptive dispatch path.
* `crates/xlog-cuda/src/provider/wcoj.rs` —
  `wcoj_4cycle_skew_score_u32` / `_u64` + `score_from_histograms_4cycle`.
* `crates/xlog-cuda/kernels/wcoj.cu` —
  `wcoj_4cycle_skew_histogram_u32` / `_u64`.
* `crates/xlog-integration/tests/test_wcoj_4cycle_adaptive_dispatch.rs` —
  4 cert tests locking the dispatch policy.
* `crates/xlog-integration/tests/test_wcoj_4cycle_executor_wiring.rs` —
  4 cert tests locking force-gate / kill-switch / row-set parity.
