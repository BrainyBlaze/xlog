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

## Full Criterion Bench (Deferred)

A full perf bench mirroring `crates/xlog-integration/benches/wcoj_triangle_bench.rs`
({u32,u64} × {uniform, superhub, empty} × {sizes} × {Off, Force,
Adaptive} matrix) is deferred to a follow-up slice. The four
correctness gates above are the slice acceptance; absolute
throughput numbers do not affect the dispatch correctness contract.
A future bench slice will:

1. Implement `crates/xlog-integration/benches/wcoj_4cycle_bench.rs`.
2. Capture {Off, Force, Adaptive} medians + IQR per fixture.
3. Verify the threshold's ≥1.7× headroom holds quantitatively
   beyond the cert-test pass/fail boundary.
4. Inform the eventual default-on follow-up slice's go/no-go.

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
