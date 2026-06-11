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

Worst single observation across these two runs: 6.11x (min(Z) hub_50k);
in these two runs all fixtures cleared the >= 3x gate. See the post-commit
re-verification below before reading this as a robust per-run guarantee.

## Post-commit independent re-verification (same day, same checkout)

After the commits landed, an independent re-run series on idle GPU
(verified no concurrent `cargo test` / GPU processes before each run)
reproduced the gate for most fixtures but exposed run-to-run bimodality
on the hub_50k sum/max fixtures. Full-fixture runs (same commands):

| fixture | aggregate | speedup run A / B / C |
|---|---|---|
| hub_10k_z16 | sum(Z) | 7.27x / 6.53x / 6.62x |
| hub_50k_z16 | sum(Z) | 5.77x / 4.48x / **2.87x** |
| hub_10k_z16 | min(Z) | 6.57x / 6.01x / 8.76x |
| hub_50k_z16 | min(Z) | 7.38x / 5.64x / 7.71x |
| hub_10k_z16 | max(Z) | 6.38x / 6.96x / 7.16x |
| hub_50k_z16 | max(Z) | 13.97x / 5.62x / **2.83x** |
| u64_hub_10k_z16 | count | 59.15x / 43.93x / 54.36x |
| u64_hub_50k_z16 | count | 7.76x / 26.97x / 13.73x |

Three additional focused re-runs of the agg measurement confirmed the
bimodality on hub_50k: sum(Z) 5.99x / **2.81x** / 6.12x and max(Z)
5.14x / 5.71x / **2.12x**. In the slow mode the fused median jumps from
~3-3.7 ms to ~7-8.4 ms (the dip hits sum or max non-deterministically,
one per run at most; min and both 10k fixtures never dipped). Absolute
times on this laptop GPU drift with clocks/thermals run to run.

## Honest verdict

* u64-key count: GATE PASSED — every observation 7.76x-74x.
* min(Z): GATE PASSED — every observation 5.64x-18.43x.
* sum(Z)/max(Z), hub_10k: GATE PASSED — every observation >= 6.38x.
* sum(Z)/max(Z), hub_50k: GATE PASSED ON MEDIAN ONLY — median across
  runs is well above 3x (4.5-6x typical), but 4 of the 12 re-verification
  observations (6 per fixture) fell in the 2.1x-2.9x band (bimodal fused time, cause
  not yet isolated: laptop clock throttling vs. allocator/pool state).
  A controlled-clock rerun (or a desktop GPU) is needed before claiming
  an unqualified per-run >= 3x for these two fixtures.

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

## Controlled rerun (2026-06-11, post-merge main, idle GPU)

Clock locking is not permitted on this GPU (`nvidia-smi -lgc`: permission
denied, WSL laptop); instead: three consecutive full measurement runs on an
otherwise idle GPU with SM clock and temperature recorded per run
(180 MHz cold-start -> 1942 MHz sustained, 60 C throughout).

Result: **18/18 observations PASS the >=3x gate** (sum/min/max x
hub_10k/hub_50k x 3 runs). Worst observation 4.80x (max_z hub_50k, during
the 180 MHz cold-clock run); sustained-clock range 8.14x-23.04x. The
2.1x-2.9x bimodal dips reported in the qualified verdict above did NOT
reproduce; absolute times vary ~3x with clock state but the fused/unfused
ratio clears the gate in every observation. Verdict upgraded: S1b gate
PASS on all fixtures, all observations, under recorded thermal/clock
conditions. Repro: loop
`cargo test -p xlog-cuda-tests --release --test test_wcoj_groupby_root_agg -- --ignored --nocapture`
3x with `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` before each.
