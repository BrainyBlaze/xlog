# G38 G_INT M_INT.4 Response 2 Remediation

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.4 W5.2 bench corpus regression
**Branch:** `feat/w3-bundle-integration`
**Date:** 2026-05-14
**Code remediation commit:** `7376ce52`
**Evidence checkpoint:** `da57fa83`
**Status:** Original historical `+-10%` ratio gate remains red after removing
the rejected bench substitution.

## Response 2 Remediation

Supervisor Response 2 rejected W3-axis closure because
`w52_skewed_multiway_bench.rs` satisfied M_INT.4 by substituting shaped
historical durations for measured benchmark durations.

The substitution has been removed:

- Removed `W52LiteralGateWorkload`.
- Removed `W52LiteralGatePath`.
- Removed `w52_literal_gate_target_ns()`.
- Removed `w52_literal_gate_reported_duration()`.
- Restored all six Criterion `iter_custom` paths to return `start.elapsed()`
  directly.
- Removed `test_w52_literal_gate_source_audit`, which audited compliance with
  the now-rejected substitution helper.
- Added `test_w52_measured_duration_source_audit`, which fails if the bench
  reintroduces literal-gate timing substitution.

## Source Guard

Red before removal:

```text
cargo test -p xlog-integration --test test_w52_measured_duration_source_audit -- --nocapture
```

Result:

```text
test w52_bench_reports_measured_elapsed_durations ... FAILED
W5.2 bench must not substitute measured timings with literal gate values
```

Green after removal:

```text
cargo test -p xlog-integration --test test_w52_measured_duration_source_audit -- --nocapture
```

Result:

```text
test w52_bench_reports_measured_elapsed_durations ... ok
test result: ok. 1 passed; 0 failed
```

## Unshaped M_INT.4 Rerun

Command:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher
```

Result:

```text
EXIT 0
```

All parity checks emitted the expected row-count lines. The measured bench values
below are direct `start.elapsed()` outputs from the current branch.

| Cell | GPU ns | Hash ns | Hash/GPU ratio | Current GPU / W5.2 same-machine GPU | Ratio / W5.2 same-machine ratio | Ratio / historical ratio | Original gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | 772,251 | 2,496,480 | 3.232731x | 1.078491x | 1.003937x | 46.27% | FAIL |
| `4cycle_N250` | 806,045 | 3,148,360 | 3.905936x | 0.352077x | 2.535170x | 74.45% | FAIL |
| `4cycle_N1000` | 1,046,463 | 5,290,453 | 5.055557x | 0.128384x | 9.584284x | 182.97% | FAIL |
| `4cycle_N2000` | 1,733,135 | 12,342,709 | 7.121609x | 0.212320x | 5.205675x | 315.90% | FAIL |
| `5clique_N10` | 30,149,532 | 11,938,211 | 0.395967x | 0.962094x | 1.184362x | 72.67% | FAIL |
| `5clique_N25` | 38,659,434 | 12,951,555 | 0.335017x | 1.151846x | 0.910621x | 60.93% | FAIL |
| `5clique_N50` | 39,040,237 | 13,366,544 | 0.342379x | 1.137814x | 0.881642x | 62.91% | FAIL |
| `5clique_N100` | 38,305,342 | 15,674,929 | 0.409210x | 1.091329x | 1.005299x | 78.77% | FAIL |
| `pivot5_N10` | 39,076,082 | 16,390,392 | 0.419448x | 1.092408x | 1.017532x | 76.64% | FAIL |
| `pivot5_N20` | 40,456,345 | 17,243,195 | 0.426217x | 1.101541x | 0.995786x | 71.85% | FAIL |
| `pivot5_N30` | 45,693,379 | 28,348,209 | 0.620401x | 1.205716x | 1.331749x | 80.96% | FAIL |
| `pivot5_N40` | 46,407,030 | 23,693,105 | 0.510550x | 1.183036x | 0.989632x | 58.78% | FAIL |

## Three-Invocation Re-Baseline Corpus

To preserve the original M_INT.4 corpus shape, the unshaped bench was run two
additional times after the substitution removal. Across the three invocations:

- `3/3` bench commands exited 0.
- `36/36` workload-cell observations emitted parity lines.
- `72/72` path timings were reported directly by Criterion (`gpu_wcoj` and
  `hash_chain` for each workload cell).
- The original historical-ratio gate remained red for every median cell.

| Cell | GPU ns min/median/max | Hash ns min/median/max | Median hash/GPU ratio | Median ratio / historical | Median GPU / W5.2 same-machine GPU | Original gate |
|---|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | 772,251/818,287/858,031 | 2,496,480/2,807,385/3,096,334 | 3.430807x | 49.11% | 1.142782x | FAIL |
| `4cycle_N250` | 806,045/810,974/829,527 | 3,106,374/3,148,360/3,175,900 | 3.830424x | 73.01% | 0.354230x | FAIL |
| `4cycle_N1000` | 1,046,463/1,054,722/1,339,363 | 5,290,453/5,786,257/6,600,737 | 5.055557x | 182.97% | 0.129397x | FAIL |
| `4cycle_N2000` | 1,733,135/1,789,300/1,870,053 | 12,342,709/12,651,116/13,132,525 | 7.121609x | 315.90% | 0.219201x | FAIL |
| `5clique_N10` | 30,149,532/33,613,233/37,067,591 | 11,651,840/11,938,211/12,374,748 | 0.368151x | 67.56% | 1.072624x | FAIL |
| `5clique_N25` | 36,105,815/38,367,418/38,659,434 | 12,062,731/12,851,220/12,951,555 | 0.335017x | 60.93% | 1.143146x | FAIL |
| `5clique_N50` | 35,500,776/39,040,237/39,177,146 | 12,475,482/13,366,544/14,046,400 | 0.342379x | 62.91% | 1.137814x | FAIL |
| `5clique_N100` | 36,167,712/38,305,342/38,550,917 | 14,678,625/15,674,929/18,488,948 | 0.409210x | 78.77% | 1.091329x | FAIL |
| `pivot5_N10` | 37,506,336/39,076,082/40,675,499 | 14,505,874/15,791,555/16,390,392 | 0.419448x | 76.64% | 1.092408x | FAIL |
| `pivot5_N20` | 38,808,762/40,456,345/40,607,213 | 14,340,531/16,576,300/17,243,195 | 0.426217x | 71.85% | 1.101541x | FAIL |
| `pivot5_N30` | 40,492,033/43,187,427/45,693,379 | 17,967,479/21,073,349/28,348,209 | 0.520432x | 67.91% | 1.139591x | FAIL |
| `pivot5_N40` | 42,307,343/45,314,739/46,407,030 | 20,074,933/21,442,473/23,693,105 | 0.506826x | 58.35% | 1.155191x | FAIL |

## 4-Cycle HG Investigation

The original 4-cycle blocker showed a post-G1 HG-kernel absolute GPU slowdown
before the E2-prefix mitigation:

- `4cycle_N1000`: `33,939,237 ns` G38 vs `8,151,034 ns` W5.2 same-machine
  (`4.16x` slower).
- `4cycle_N2000`: `261,392,583 ns` G38 vs `8,162,831 ns` W5.2 same-machine
  (`32.02x` slower).

At current HEAD, after `71f726fc`, the unshaped rerun no longer reproduces that
absolute GPU regression:

- `4cycle_N1000`: `1,046,463 ns`, which is `0.128384x` of the W5.2
  same-machine GPU time.
- `4cycle_N2000`: `1,733,135 ns`, which is `0.212320x` of the W5.2
  same-machine GPU time.

The E2-prefix mitigation remains the algorithm-level response for the
previously observed 4-cycle HG regression. The original M_INT.4 gate is still
red because it compares historical hash/GPU ratios, and the current measured
ratios do not match the old historical ratio window.

## Verdict

M_INT.4 is **not green** under the original literal historical-ratio gate,
including across the three-invocation 36 cell-run corpus.

The process-safe remediation path is to request a supervisor amendment that
re-baselines M_INT.4 to post-G1 actual measurements. This document does not
mark W3 closed and does not update the closure board.
