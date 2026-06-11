# S1c — Aggregate-Fused WCOJ: 4-Cycle Count + u64 Sum/Min/Max + Symbol Locks

Gate per slice: fused >= 3x vs unfused (materialize + groupby) on skewed
hub fixtures. Parity (fused == host brute-force oracle == unfused
baseline) is asserted before every timing rep; the unfused result is the
definition of correct.

Host: local dev GPU (WSL2, NVIDIA RTX PRO 3000 Blackwell Laptop GPU —
same box as the S1/S1b evidence). Branch
`feat/factorized-agg-4cycle-widths`. Clock locking is not permitted on
this GPU (`nvidia-smi -lgc`: permission denied, WSL laptop), so SM clock
and temperature are recorded per run
(`nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader`).
Timing = median of 5 reps per run, warmup excluded.

## Slice 1 — 4-cycle group-by-root count (u32)

`deg(W, count(V)) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)` through
`wcoj_4cycle_groupby_root_count_hg_u32` (per-e1-row atomicAdd counters,
compact count>0 roots, recorded groupby Sum) — the 4-cycle rows are never
materialized. Fixtures: hub W=0 with n_x X-fanout, 16-wide shared Y and Z
bands, every Z closing back to W=0 (256 completions per e1 hub row;
materialized row count = n_x * 256), plus 1000-cycle uniform background.

Repro:

    cargo test -p xlog-cuda-tests --test test_wcoj_4cycle_groupby_root_count \
      --release -- --ignored --nocapture

| run | clocks.sm / temp | cycle4_hub_5k (unfused / fused ms, speedup) | cycle4_hub_10k (unfused / fused ms, speedup) |
|---|---|---|---|
| 1 | 180 MHz, 61 C | 194.6 / 9.53 = **20.42x** | 202.0 / 10.29 = **19.64x** |
| 2 | 352 MHz, 59 C | 163.5 / 8.55 = **19.13x** | 569.9 / 11.98 = **47.58x** |
| 3 | 352 MHz, 60 C | 162.1 / 7.94 = **20.42x** | 165.3 / 9.37 = **17.64x** |

GATE (>= 3x): **PASS in all 6 observations** (worst 17.64x). Honest
caveats: run 1 happened at idle clocks (180 MHz) and `nvidia-smi
--query-compute-apps` listed 4 other processes holding GPU contexts
(other sessions on this box) — absolute times are therefore noisy
(e.g. the 569.9 ms unfused outlier in run 2), but every fused/unfused
ratio stays >= 5.8x above the gate.

Correctness evidence (all green): `test_wcoj_4cycle_groupby_root_count.rs`
small K-shape / skewed hub / empty-intersection root absence, each
asserting unfused-vs-oracle AND fused-vs-oracle; e2e
`test_wcoj_groupby_fusion.rs::groupby_fusion_4cycle_count_fires_end_to_end_with_parity`
(fused counter == 1, kill-switch row parity) and
`groupby_fusion_4cycle_sum_declines_to_unfused` (Count-only fusion lock).

Gating decision (documented per the brief): the fused 4-cycle path
mirrors the triangle fusion — enabled by default behind the shared
`XLOG_DISABLE_WCOJ_GROUPBY_FUSION` kill switch. The opt-in
`XLOG_USE_WCOJ_4CYCLE*` envs keep governing only the NON-aggregate
4-cycle materialize dispatch, which is exactly the path a declined or
kill-switched fusion falls back to.

## Slice 2 — u64-key sum/min/max

`wcoj_triangle_groupby_root_{sum,min,max}_hg_u64` + metadata-driven
segment reduction (`wcoj_groupby_root_segment_{sum,min,max}_values_u64`;
the recorded groupby is U32/Symbol-key only). Prerequisite baseline
widening: the legacy groupby (u64-key route) gained u64-value
sum/min/max (`groupby_min_u64`/`groupby_max_u64`, min/max result schema
preserves the value width) — locked by
`groupby_legacy_agg_u64_values_matches_host` with keys > 2^40 and values
> u32::MAX.

Fixture: u64_hub_10k_z16 (n_xy=11000, n_yz=161000, keys above 2^40),
value column Z. Repro:

    cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_agg \
      --release -- --ignored s1c_measurement_u64_agg --nocapture

| run | clocks.sm / temp | sum(Z) | min(Z) | max(Z) |
|---|---|---|---|---|
| 1 | 1942 MHz, 58 C | 27.79 / 0.78 = **35.51x** | 28.53 / 0.78 = **36.65x** | 23.42 / 0.77 = **30.28x** |
| 2 | 1942 MHz, 55 C | 24.79 / 0.71 = **34.71x** | 24.68 / 0.78 = **31.49x** | 21.57 / 0.70 = **30.94x** |
| 3 | 1942 MHz, 55 C | 19.79 / 0.56 = **35.43x** | 19.45 / 0.55 = **35.43x** | 25.21 / 0.70 = **36.10x** |

GATE (>= 3x): **PASS in all 9 observations** (worst 30.28x; sustained
clocks throughout).

Correctness evidence (all green): `test_wcoj_groupby_root_agg.rs` u64
section — all six (op, value in {Y, Z}) combinations on small K4
(keys > 2^33) and skewed-hub (keys > 2^40) fixtures, fused == oracle ==
unfused baseline; empty-intersection root absence per op; e2e u64
sum(Z)/min(Z)/max(Y) fused-fires (counter == 1) with kill-switch parity.

## Slice 3 — Symbol-valued aggregates (type-gating verification)

Honest scope statement: NO production code change was needed — the
existing gates already produce exactly-matching fused/unfused semantics.
This slice is locked behavior, not new capability:

* `count` over Symbol-keyed/valued triangles FUSES (Symbol is
  u32-physical and count never reads the value column); the fused output
  preserves the Symbol key type. Locked by
  `groupby_fusion_count_symbol_fires_end_to_end_with_parity`
  (counter == 1, kill-switch parity, schema check).
* `sum`/`min`/`max` over Symbol VALUES: semantic, not physical — symbol
  ids carry no arithmetic order. The unfused groupby REJECTS Symbol
  values with its value-type error; the fused hook therefore DECLINES
  (counter == 0) and the query fails through the same unfused gate with
  a byte-identical error in fused-enabled and kill-switch runs. Locked by
  `groupby_fusion_symbol_valued_min_declines_and_unfused_rejects` (e2e)
  and `groupby_root_agg_rejects_symbol_value_columns` (provider level,
  all three ops).

## Stretch — k-clique (K=5,6) count fusion

NOT attempted — deferred explicitly. The non-aggregate clique kernels and
leader work plans exist (`wcoj_clique{5..8}_count_hg_*`,
`KCliqueVariableOrder`), but a fused variant needs its own
inside-aggregate promoter, leader-edge row-recovery in the templated
kernel, executor params plumbing, and a full TDD/parity/measurement
cycle, which did not fit this slice after slices 1-3.

Also deferred (stated in the architecture guide): u64-key 4-cycle count
fusion, 4-cycle sum/min/max fusion, recorded-groupby u64-value min/max
(only the legacy path was widened — u64 keys route there).

## Final verification on the completed tree

See the commit history of `feat/factorized-agg-4cycle-widths` for
per-slice TDD reds. Full-suite runs on the final tree (same checkout,
same day):

* `cargo test -p xlog-cuda-tests --release` — all binaries green
  (unittests 69, category_isolation 24, certification_suite 1 (the 206
  cert wrapper), the three groupby-root test files 3+11+6 with 5 ignored
  measurement harnesses, remaining ILP/properties/smoke binaries all 0
  failed).
* `cargo test -p xlog-integration --release` — exit code 0, 42 test
  binaries, totals: 409 passed / 0 failed / 3 ignored.
* Touched crates (`xlog-cuda`, `xlog-logic`, `xlog-runtime`,
  `xlog-integration`, `xlog-cuda-tests`) build warning-free in release.

Flake observed and root-caused during the work: one early run of
`groupby_legacy_agg_u64_values_matches_host` printed
`skipping ...: no CUDA device` and passed vacuously — `CudaDevice::new(0)`
failed transiently while 4 other processes held GPU contexts. Reruns with
`--nocapture` confirmed the genuine red (then green after the widening).
GPU-skip-on-missing-device is the suite's established pattern; the lesson
recorded here is to check for the skip line before trusting a green.
