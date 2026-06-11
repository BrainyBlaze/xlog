# S1d — Aggregate-Fused WCOJ: 4-Cycle Agg Variants

Slices: (1) 4-cycle sum/min/max fusion (u32), (2) u64-key 4-cycle count
fusion, (3) recorded-groupby u64-value min/max, (4) float/LogSumExp
design note (no implementation — by directive).

Gate for slices 1-2: fused >= 3x vs unfused (materialize + groupby) on
the skewed 4-cycle hub fixture. Parity (fused == host brute-force oracle
== unfused baseline) is asserted before the timing reps; the unfused
result is the definition of correct.

Host: local dev GPU (WSL2, NVIDIA RTX PRO 3000 Blackwell Laptop GPU —
same box as the S1/S1b/S1c evidence). Branch
`feat/factorized-4cycle-agg-variants`. Clock locking is not permitted on
this GPU (WSL laptop), so SM clock and temperature are recorded per run
(`nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader`).
Timing = median of 5 reps per run, warmup excluded. Honest caveat:
`nvidia-smi --query-compute-apps` listed 2 other processes holding GPU
contexts during the slice-1 runs (other sessions on this box), so
absolute times are noisy — the per-run fused/unfused ratios are the
evidence, and every one clears the gate.

## Slice 1 — 4-cycle sum/min/max fusion (u32 keys/values)

`q(W, agg(V)) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)`,
`agg ∈ {sum, min, max}`, `V ∈ {X, Y, Z}` through
`wcoj_4cycle_groupby_root_{sum,min,max}_hg_u32` (per-e1-row integer-atomic
match counter + aggregate partial, compact count>0 roots, recorded
groupby reduce) — the 4-cycle rows are never materialized. Executor
branch widened from Count-only with the same value-column mapping rules
as the triangle: X = output col 1 (e1.col1), Y = col 2 (e2.col1),
Z = col 3 (e3.col1), plain U32 only.

Fixture: hub W=0 with 10k X-fanout, 16-wide shared Y and Z bands, every Z
closing back to W=0 (256 completions per e1 hub row; materialized row
count = n_x * 256), plus 1000-cycle uniform background; value column Z.

Repro:

    cargo test -p xlog-cuda-tests --test test_wcoj_4cycle_groupby_root_agg \
      --release -- --ignored --nocapture

| run | clocks.sm / temp | sum(Z) | min(Z) | max(Z) |
|---|---|---|---|---|
| 1 | 1942 MHz, 58 C | 51.70 / 6.91 = **7.49x** | 49.74 / 5.48 = **9.08x** | 48.64 / 6.78 = **7.17x** |
| 2 | 1942 MHz, 61 C | 58.92 / 7.16 = **8.23x** | 52.81 / 8.30 = **6.37x** | 26.22 / 7.97 = **3.29x** |
| 3 | 1942 MHz, 60 C | 74.25 / 5.82 = **12.76x** | 51.17 / 6.65 = **7.70x** | 62.03 / 5.55 = **11.17x** |

GATE (>= 3x): **PASS in all 9 observations** (worst 3.29x — the max(Z)
run-2 unfused median dropped to 26.2 ms under the shared-GPU noise noted
above; the ratio still clears the gate).

Correctness evidence (all green):
`test_wcoj_4cycle_groupby_root_agg.rs` — all 9 (op, value in {X, Y, Z})
combinations on the small K-shape and skewed-hub fixtures, each asserting
unfused-vs-oracle AND fused-vs-oracle; bag-semantics duplicate-value
fixture; empty-intersection root absence per op; zero-valued sum group
presence; provider-level Symbol value rejection (all 9 combos). E2e
`test_wcoj_groupby_fusion.rs`:
`groupby_fusion_4cycle_{sum_z,sum_x,min_x,max_y}_fires_end_to_end_with_parity`
(fused counter == 1, kill-switch row parity) and
`groupby_fusion_4cycle_symbol_valued_min_declines_and_unfused_rejects`
(counter == 0, byte-identical rejection fused-enabled vs kill-switch).
The S1c Count-only lock test was superseded by the fires tests.

## Slice 2 — u64-key 4-cycle count fusion

`wcoj_4cycle_groupby_root_count_hg_u64` + metadata-driven segment
reduction (`wcoj_build_metadata_u64_recorded` on e1.col0 +
`wcoj_groupby_root_segment_sum_counts_u32`; the recorded groupby is
U32/Symbol-key only), mirroring the triangle u64 count machinery. Output
schema (W: U64, count: U64). u64-key 4-cycle sum/min/max stays deferred:
the executor agg branch declines 8-byte keys.

Fixture: same hub shape with keys above 2^40 (u64_hub_10k). Repro:

    cargo test -p xlog-cuda-tests --test test_wcoj_4cycle_groupby_root_count \
      --release -- --ignored s1d_measurement_4cycle_count_u64 --nocapture

| run | clocks.sm / temp | count (unfused / fused ms, speedup) |
|---|---|---|
| 1 | 352 MHz, 57 C | 87.24 / 5.33 = **16.37x** |
| 2 | 1942 MHz, 58 C | 149.73 / 6.51 = **23.01x** |
| 3 | 1942 MHz, 58 C | 144.98 / 5.56 = **26.06x** |

GATE (>= 3x): **PASS in all 3 observations** (worst 16.37x; run 1
started at idle clocks — ratio unaffected).

Correctness evidence (all green): `test_wcoj_4cycle_groupby_root_count.rs`
u64 section — small (keys > 2^33) and skewed hub (keys > 2^40), each
asserting unfused(materialize + legacy groupby)-vs-oracle AND
fused-vs-oracle; empty-intersection root absence; e2e
`groupby_fusion_4cycle_count_u64_fires_end_to_end_with_parity`
(counter == 1, kill-switch parity).

## Slice 3 — recorded-groupby u64-value min/max

No measurement gate (correctness widening, not a perf slice).
`groupby_multi_agg_recorded` now accepts U64 value columns for Min/Max
(previously Sum-only): u64 values reduce through
`groupby_min_u64`/`groupby_max_u64` (min identity `u64::MAX` via
`arith_fill_const_u64`, max identity 0) and the result column preserves
the value width — matching the legacy path widened in S1c. Recorded Sum
over u64 values was verified already-working (pre-existing lock
`groupby_recorded_sum_u64_values_matches_host`, green before and after).

Locked by `groupby_recorded_min_max_u64_values_matches_host_and_legacy`
(values above `u32::MAX` plus `u64::MAX`/0 identity pins; recorded ==
host == legacy row sets; U64 result schema). TDD red watched: the
pre-change run failed with
`Kernel error: Min currently requires U32 values, got U64`.

## Slice 4 — float/LogSumExp aggregates: design note only

NO implementation, per the brief. The architecture guide's deferred list
gained a design-decision section: float atomics are non-associative in
IEEE-754 and would break the fused kernels' deterministic-values
contract; recorded candidates are (1) per-block deterministic tree
reduction with ordered block combine (preferred) and (2) fixed-point
encoding onto the existing exact integer atomics — baseline (unfused
groupby float support) first either way.

## Still deferred after S1d

u64-key 4-cycle sum/min/max fusion, k-clique (K=5..8) count fusion,
LogSumExp/float aggregates (design note above).

## Final verification on the completed tree

See the commit history of `feat/factorized-4cycle-agg-variants` for
per-slice TDD reds (4-cycle agg fused-counter 0 vs 1 e2e reds, u64 count
counter red, recorded min/max `requires U32 values, got U64` red).
Full-suite runs on the final code tree (same checkout, same day):

* `cargo test -p xlog-cuda-tests --release` — all 17 binaries green
  (unittests 69, category_isolation 24, certification_suite 1 (the 206
  cert wrapper), the five groupby-root test files 6+6+12+7 passed with
  the ignored measurement harnesses, remaining ILP/properties/smoke
  binaries all 0 failed).
* `cargo test -p xlog-integration --release` — exit code 0, 43 test
  binaries, 417 passed / 0 failed.
* Touched crates (`xlog-cuda`, `xlog-runtime`, `xlog-cuda-tests`,
  `xlog-integration`) build warning-free in release (the two
  pre-existing `unused_mut` warnings in the fused staging copies were
  fixed first, in their own commit).

Measurement-harness note: one early full run of the new agg test file
aborted with `stream: Capacity { max: 16 }` — the per-call
`StreamPool::acquire` pattern exceeded the 16-stream pool at 18
acquisitions per case; fixed by acquiring one stream per case (root
cause understood, not a flake). A separate first attempt at counting the
integration totals through the lean-ctx shell wrapper compressed the
`test result:` lines and reported a nonsense total; the recorded numbers
above come from the raw uncompressed log.
