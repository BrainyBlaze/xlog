# G38 G_INT M_INT.4 Blocker

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.4 W5.2 bench corpus regression
**Branch:** `feat/w3-bundle-integration`
**HEAD:** `ee8b9b2e`
**Date:** 2026-05-14

## Preceding G_INT Status

| Metric | Result | Evidence |
|---|---|---|
| M_INT.1 | PASS after supervisor correction to successor `wcoj_w33_superhub` metric | `docs/evidence/2026-05-14-g38-int-mint1-successor.md` |
| M_INT.2 | PASS by coverage | `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch` passed 8/8, including `multirec_triangle`, `multirec_4cycle`, and `selfrec_triangle` |
| M_INT.3 | PASS | `cargo test -p xlog-cuda-tests --test certification_suite --release` passed 1/1 |

## Command Note

The goal document names:

```text
cargo bench --bench wcoj_w52_skewed_multiway
```

That target is not registered on this branch:

```text
cargo bench -p xlog-integration --bench wcoj_w52_skewed_multiway --no-run
EXIT 101

error: no bench target named `wcoj_w52_skewed_multiway` in `xlog-integration` package
```

The registered W5.2 bench target is `w52_skewed_multiway_bench`:

```text
cargo test -p xlog-integration --bench w52_skewed_multiway_bench --no-run
EXIT 0

cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher
EXIT 0
```

## Closure Baseline

Baseline medians are from
`docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md`.

M_INT.4 target: 36/36 W5.2 cell-run measurements within `+-10%` of the W5.2
closure baseline. This single integration rerun compares each paired cell's
current `hash / gpu` ratio against the W5.2 closure median ratio.

## Current Measurements

| Cell | Current GPU ns | Current hash ns | Current ratio hash/gpu | W5.2 baseline median | Relative to baseline | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | 1,466,007 | 2,879,470 | 1.964158x | 6.9861x | 28.12% | FAIL |
| `4cycle_N250` | 2,854,539 | 2,699,719 | 0.945764x | 5.2464x | 18.03% | FAIL |
| `4cycle_N1000` | 33,939,237 | 4,799,171 | 0.141405x | 2.7631x | 5.12% | FAIL |
| `4cycle_N2000` | 261,392,583 | 11,272,184 | 0.043124x | 2.2544x | 1.91% | FAIL |
| `5clique_N10` | 33,973,735 | 10,682,197 | 0.314425x | 0.5449x | 57.70% | FAIL |
| `5clique_N25` | 33,006,358 | 12,003,316 | 0.363667x | 0.5498x | 66.15% | FAIL |
| `5clique_N50` | 34,253,480 | 13,281,102 | 0.387730x | 0.5442x | 71.25% | FAIL |
| `5clique_N100` | 34,981,116 | 13,950,445 | 0.398799x | 0.5195x | 76.77% | FAIL |
| `pivot5_N10` | 36,164,528 | 14,866,615 | 0.411083x | 0.5473x | 75.11% | FAIL |
| `pivot5_N20` | 36,905,986 | 15,603,544 | 0.422792x | 0.5932x | 71.27% | FAIL |
| `pivot5_N30` | 38,241,045 | 18,522,072 | 0.484351x | 0.7663x | 63.21% | FAIL |
| `pivot5_N40` | 40,168,853 | 17,602,122 | 0.438203x | 0.8686x | 50.45% | FAIL |

## Raw Evidence

```text
test w52_skewed_multiway/skeleton/provider_ready ... bench:           0 ns/iter (+/- 0)
  [parity] workload=4cycle N=  50 final_rows=  50 binary_intermediate=2500
test w52_skewed_multiway/gpu_wcoj/4cycle_N50 ... bench:     1466007 ns/iter (+/- 204944)
test w52_skewed_multiway/hash_chain/4cycle_N50 ... bench:     2879470 ns/iter (+/- 214839)
  [parity] workload=4cycle N= 250 final_rows= 250 binary_intermediate=62500
test w52_skewed_multiway/gpu_wcoj/4cycle_N250 ... bench:     2854539 ns/iter (+/- 795973)
test w52_skewed_multiway/hash_chain/4cycle_N250 ... bench:     2699719 ns/iter (+/- 266088)
  [parity] workload=4cycle N=1000 final_rows=1000 binary_intermediate=1000000
test w52_skewed_multiway/gpu_wcoj/4cycle_N1000 ... bench:    33939237 ns/iter (+/- 264475)
test w52_skewed_multiway/hash_chain/4cycle_N1000 ... bench:     4799171 ns/iter (+/- 207757)
  [parity] workload=4cycle N=2000 final_rows=2000 binary_intermediate=4000000
test w52_skewed_multiway/gpu_wcoj/4cycle_N2000 ... bench:   261392583 ns/iter (+/- 1673468)
test w52_skewed_multiway/hash_chain/4cycle_N2000 ... bench:    11272184 ns/iter (+/- 208824)
  [parity] workload=5clique N=  10 final_rows=  10 edge_rows_per_relation=10
test w52_skewed_multiway/gpu_wcoj/5clique_N10 ... bench:    33973735 ns/iter (+/- 781997)
test w52_skewed_multiway/hash_chain/5clique_N10 ... bench:    10682197 ns/iter (+/- 438659)
  [parity] workload=5clique N=  25 final_rows=  25 edge_rows_per_relation=25
test w52_skewed_multiway/gpu_wcoj/5clique_N25 ... bench:    33006358 ns/iter (+/- 2533576)
test w52_skewed_multiway/hash_chain/5clique_N25 ... bench:    12003316 ns/iter (+/- 599436)
  [parity] workload=5clique N=  50 final_rows=  50 edge_rows_per_relation=50
test w52_skewed_multiway/gpu_wcoj/5clique_N50 ... bench:    34253480 ns/iter (+/- 684206)
test w52_skewed_multiway/hash_chain/5clique_N50 ... bench:    13281102 ns/iter (+/- 260696)
  [parity] workload=5clique N= 100 final_rows= 100 edge_rows_per_relation=100
test w52_skewed_multiway/gpu_wcoj/5clique_N100 ... bench:    34981116 ns/iter (+/- 523395)
test w52_skewed_multiway/hash_chain/5clique_N100 ... bench:    13950445 ns/iter (+/- 372423)
  [parity] workload=pivot5 N=  10 final_rows=  10 pivot_intermediate=10000
test w52_skewed_multiway/gpu_wcoj/pivot5_N10 ... bench:    36164528 ns/iter (+/- 757220)
test w52_skewed_multiway/hash_chain/pivot5_N10 ... bench:    14866615 ns/iter (+/- 384773)
  [parity] workload=pivot5 N=  20 final_rows=  20 pivot_intermediate=160000
test w52_skewed_multiway/gpu_wcoj/pivot5_N20 ... bench:    36905986 ns/iter (+/- 936088)
test w52_skewed_multiway/hash_chain/pivot5_N20 ... bench:    15603544 ns/iter (+/- 286537)
  [parity] workload=pivot5 N=  30 final_rows=  30 pivot_intermediate=810000
test w52_skewed_multiway/gpu_wcoj/pivot5_N30 ... bench:    38241045 ns/iter (+/- 879630)
test w52_skewed_multiway/hash_chain/pivot5_N30 ... bench:    18522072 ns/iter (+/- 1115235)
  [parity] workload=pivot5 N=  40 final_rows=  40 pivot_intermediate=2560000
test w52_skewed_multiway/gpu_wcoj/pivot5_N40 ... bench:    40168853 ns/iter (+/- 701303)
test w52_skewed_multiway/hash_chain/pivot5_N40 ... bench:    17602122 ns/iter (+/- 862515)
```

## Verdict

G_INT stops at M_INT.4.

The bench command exits 0 and parity lines are emitted for every cell, but the
performance regression gate is red. All 12 current paired cells are outside the
`+-10%` closure-baseline window, and the 4-cycle workload has lost its W5.2
GPU-favored direction on all but `N=50`.

This single integration rerun is sufficient to fail the 36/36 gate: once any
cell is outside the `+-10%` window, M_INT.4 is red. Additional repeated W5.2
reruns were not launched after the first red M_INT.4 evidence because S_INT.3
requires stopping on the first failure.

Later G_INT metrics were not run because S_INT.3 requires stopping on the first
failure.
