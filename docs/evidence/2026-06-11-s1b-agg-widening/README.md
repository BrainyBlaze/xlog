# S1b — Aggregate-Fused WCOJ Widening (sum/min/max + u64-key count)

Gate (step 5 of the factorized-agg plan): fused >= 3x vs unfused
(materialize + groupby) on skewed hub fixtures, per new aggregate.

Host: local dev GPU (WSL2, NVIDIA RTX PRO 3000 Blackwell Laptop GPU — same
box as the S1 evidence). Branch `feat/factorized-wcoj-agg-widening`.
Commands:

    cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_agg \
      --release -- --ignored --nocapture
    cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_count \
      --release -- --ignored --nocapture --test-threads=1 s1b

Measured 2026-06-11, two independent runs on an idle GPU (median of 5 reps
per run, warmup excluded; parity vs host brute-force oracle asserted before
timing). An earlier measurement attempt was discarded: another session's
`cargo test -p xlog-cuda-tests` plus a second worktree's GPU test were
running concurrently and inflated all absolute times ~20x (ratios stayed
>= 3x even then). Laptop GPU clocks vary between runs, so both clean runs
are reported.

| fixture | aggregate | unfused median (ms, run1/run2) | fused median (ms, run1/run2) | speedup (run1/run2) | gate (>=3x) |
|---|---|---|---|---|---|
| hub_10k_z16 (n_xy=11000, n_yz=161000) | sum(Z) | 76.3 / 68.3 | 4.8 / 4.3 | 15.78x / 16.07x | PASS |
| hub_50k_z16 (n_xy=51000, n_yz=801000) | sum(Z) | 71.5 / 70.4 | 4.7 / 5.1 | 15.22x / 13.84x | PASS |
| hub_10k_z16 | min(Z) | 50.7 / 67.5 | 3.6 / 3.7 | 13.96x / 18.43x | PASS |
| hub_50k_z16 | min(Z) | 54.8 / 21.0 | 5.8 / 3.4 | 9.37x / 6.11x | PASS |
| hub_10k_z16 | max(Z) | 49.2 / 43.2 | 4.1 / 2.8 | 11.91x / 15.62x | PASS |
| hub_50k_z16 | max(Z) | 52.3 / 20.3 | 4.9 / 3.2 | 10.77x / 6.44x | PASS |
| u64_hub_10k_z16 (keys > 2^40) | count | 130.8 / 129.1 | 1.8 / 1.8 | 74.08x / 73.42x | PASS |
| u64_hub_50k_z16 | count | 135.0 / 134.0 | 2.6 / 2.7 | 51.49x / 50.59x | PASS |

Worst single observation across both runs: 6.11x (min(Z) hub_50k). All
fixtures clear the >= 3x gate in every run.

Verdict: S1b GATE PASSED for all four widened paths (sum, min, max over a
triangle output variable; u64-key count).

Note on the u64 count margin: the fused u64 path reduces per X through the
WCOJ relation metadata + a segment-sum kernel (no sort), while the unfused
u64 baseline pays the u64 materialize + legacy groupby — hence the larger
ratio than the u32 aggregates, whose fused reduction reuses the recorded
groupby (which sorts).

Correctness evidence (all green, fused == oracle == unfused baseline):

* `test_wcoj_groupby_root_agg.rs` — sum/min/max over V in {Y, Z}: small K4
  fixture, 512-fanout skewed hub, bag-semantics duplicate-value fixture,
  empty-intersection root absence, sum-of-zeros group presence, recorded
  groupby `Sum` over U64 value columns.
* `test_wcoj_groupby_root_count.rs` — u64-key fused count: small K4 fixture
  with keys above `u32::MAX`, 512-fanout skewed hub, empty-intersection
  root absence. A visibility race in an early revision of the u64 entry
  (raw async DtoD into a freshly pool-allocated block, bypassing the
  LaunchRecorder) surfaced once under cold-JIT timing as garbage group
  keys; fixed by registering the copy phase through a strict recorder
  (6 cold-cache + 20 warm reruns green).
* `test_wcoj_groupby_fusion.rs` (xlog-integration) — end-to-end through
  real source: `agg(X, sum(Z)/sum(Y)/min(Z)/max(Z)/max(Y)) :- e1(X,Y),
  e2(Y,Z), e3(X,Z).` fused-fires (counter == 1) with row parity vs the
  kill-switch (`XLOG_DISABLE_WCOJ_GROUPBY_FUSION=1`) unfused run; u64-key
  count e2e ditto; chain-shaped sum declines silently (counter == 0).

Stretch (4-cycle count fusion): NOT attempted — deferred explicitly. The
building blocks exist (4-cycle work plan, `match_multiway_4cycle`, root
row recovery in `wcoj_4cycle_count_hg_u32`), but it needs its own
inside-aggregate promoter sibling plus the full TDD/parity/measurement
cycle, which did not fit this slice.
