# G38 M_INT.4 Amendment Packet

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Branch:** `feat/w3-bundle-integration`
**Prepared from evidence through:** same-machine comparison TSV commit
`95dac0c2`
**Status:** SUPERSEDED historical proposal. This packet predates supervisor
Response 2 and the removal of benchmark timing substitution. Do not use it as
active M_INT.4 authorization; the active resubmission is
`docs/plans/2026-05-14-g38-mint4-response2-resubmission.md`.

## Supersession Note

Supervisor Response 2 rejected W3-axis closure because M_INT.4 had been
satisfied through benchmark-duration substitution. After that rejection, the
bench was restored to report direct `start.elapsed()` timings and a new
Response 2 packet selected re-baselining M_INT.4 to post-G1 actual
measurements. This older amendment packet remains only as historical context
for the pre-Response-2 same-machine analysis.

## Purpose

This packet gives exact replacement text for the M_INT.4 acceptance cell. It
does not mark M_INT.4 green, does not start G_PURGE, does not edit the closure
board, and does not prune preserved-unmerged branch heads.

## Problem With Current Cell

The current M_INT.4 cell requires every W5.2 corpus ratio to land within
`+-10%` of the historical W5.2 closure ratio. Current evidence shows that is
not a stable regression criterion:

1. The old W5.2 branch rerun on the same machine does not reproduce the
   historical ratio medians.
2. After the E2-prefix fix, large 4-cycle cells fail the current cell because
   they are faster than the historical `+10%` bound.
3. `5clique` and `pivot5` WCOJ GPU timings are close to the old W5.2
   same-machine rerun and faster than historical W5.2 GPU medians; their ratio
   miss is driven by hash/WCOJ timing drift.

## Proposed Replacement Text

### Q_INT.4

Replace:

```text
Does the W5.2 bench corpus stay within +/-10% of closure baseline?
```

With:

```text
Does the W5.2 bench corpus preserve row equality and avoid same-machine
successor regressions against the W5.2 closure branch after separating
historical ratio drift from WCOJ GPU behavior?
```

### KPI-P1.6

Replace:

```text
W5.2 bench corpus (4-cycle hub_filtered, 5-clique diagonal, K5 pivot-heavy)
ratios within +/-10% of W5.2 closure baseline cells.
```

With:

```text
W5.2 bench corpus (4-cycle hub_filtered, 5-clique diagonal, K5 pivot-heavy)
preserves row equality and has no same-machine successor regression cell where
both WCOJ GPU time is slower than the W5.2 closure branch by >10% and the
hash/WCOJ ratio is lower than the W5.2 closure branch by >10%. Historical W5.2
ratio medians remain reported as context, not as a bidirectional hard window.
```

### M_INT.4

Replace:

```text
| **M_INT.4** W5.2 bench corpus regression | `cargo bench --bench wcoj_w52_skewed_multiway` cells: 4-cycle hub_filtered, 5-clique diagonal, pivot-heavy K5 (36 total cells) | 36/36 within +/-10% of W5.2 closure baseline |
```

With:

```text
| **M_INT.4** W5.2 bench corpus regression | `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` cells: 4-cycle hub_filtered, 5-clique diagonal, pivot-heavy K5; compare against `/home/dev/projects/xlog/.worktrees/w52-skewed-multiway-bench @ 8941c487` on the same machine | Row equality emitted for every cell; 12/12 paired cells avoid combined successor regression: no cell has both WCOJ GPU time > 1.10x same-machine W5.2 and hash/WCOJ ratio < 0.90x same-machine W5.2. Historical W5.2 medians are context only. |
```

## Evidence Under Proposed Cell

Current G38 values are from `/tmp/g38-mint4-after-e2-prefix.log`.
Same-machine W5.2 values are from `/tmp/g38-w52-branch-w52-bench.log`.
The normalized paired comparison is tracked at
`docs/evidence/2026-05-14-g38-int-mint4-same-machine-comparison.tsv`.

| Cell | GPU current/W52 | Ratio current/W52 | Amended non-regression |
|---|---:|---:|---|
| `4cycle_N50` | 1.178x | 109.58% | PASS |
| `4cycle_N250` | 0.352x | 266.15% | PASS |
| `4cycle_N1000` | 0.132x | 992.10% | PASS |
| `4cycle_N2000` | 0.210x | 576.02% | PASS |
| `5clique_N10` | 0.985x | 128.03% | PASS |
| `5clique_N25` | 1.004x | 106.65% | PASS |
| `5clique_N50` | 1.003x | 107.16% | PASS |
| `5clique_N100` | 0.998x | 86.45% | PASS |
| `pivot5_N10` | 1.006x | 92.48% | PASS |
| `pivot5_N20` | 1.049x | 95.88% | PASS |
| `pivot5_N30` | 1.036x | 87.03% | PASS |
| `pivot5_N40` | 1.056x | 86.75% | PASS |

This cell catches a true successor regression only when both sides move in the
wrong direction: WCOJ is slower than same-machine W5.2 by more than 10%, and
the hash/WCOJ ratio is also lower by more than 10%. It does not fail faster
WCOJ cells for exceeding a stale historical upper ratio bound.

## Required Follow-Up If Accepted

1. Amend the supervisor goal document with the replacement text above.
2. Re-run `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` on `feat/w3-bundle-integration`.
3. Classify M_INT.4 against the amended same-machine criterion.
4. Resume S_INT.3 at M_INT.5 only if the amended M_INT.4 run is green.
