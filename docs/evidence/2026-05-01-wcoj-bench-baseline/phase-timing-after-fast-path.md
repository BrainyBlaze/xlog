# WCOJ Phase Timing Report — After Layout Fast-Path

**Date**: 2026-05-01
**Measured at**: head of `feat/v0.6.2-wcoj-layout-fast-path` (off `99f72bb3`)
**Iters per cell**: 30 samples on a single `Executor` (default-on adaptive); first 5 discarded as warmup.

## Context

The prior phase report at commit `4a83ab4e` showed `wcoj_layout_*_recorded` (sort + dedup of each input) was **91–97% of wall clock** on super-hub fixtures, dwarfing every other phase. This slice adds a proof-based fast-path: if the input is already strictly lex-sorted AND full-row unique, skip `dedup_full_row_recorded` and emit a recorded device-side clone instead. Per the orient, the bench's host-deduped fixtures should hit fast-path 3/3 per dispatch.

## Results (post-fast-path)

### `superhub-u32-10K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.092 | 12.6% |
| layout_xy | 0.108 | 14.8% |
| layout_yz | 0.104 | 14.3% |
| layout_xz | 0.104 | 14.2% |
| **layout_total** | **0.304** | **41.6%** |
| triangle_count | 0.076 | 10.4% |
| triangle_scan | 0.014 | 2.0% |
| triangle_total | 0.006 | 0.8% |
| triangle_materialize | 0.101 | 13.8% |
| **triangle_gpu_total** | **0.214** | **29.2%** |
| residual_overhead | 0.124 | 17.0% |
| **wall** | **0.731** | **100%** |

### `superhub-u32-50K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.082 | 5.3% |
| layout_xy | 0.113 | 7.3% |
| layout_yz | 0.106 | 6.8% |
| layout_xz | 0.110 | 7.1% |
| **layout_total** | **0.344** | **22.2%** |
| triangle_count | 0.411 | 26.5% |
| triangle_scan | 0.014 | 0.9% |
| triangle_total | 0.006 | 0.4% |
| triangle_materialize | 0.506 | 32.7% |
| **triangle_gpu_total** | **0.938** | **60.5%** |
| residual_overhead | 0.188 | 12.2% |
| **wall** | **1.550** | **100%** |

### `superhub-u64-10K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.081 | 10.5% |
| layout_xy | 0.112 | 14.6% |
| layout_yz | 0.103 | 13.3% |
| layout_xz | 0.102 | 13.3% |
| **layout_total** | **0.321** | **41.6%** |
| triangle_count | 0.097 | 12.6% |
| triangle_scan | 0.014 | 1.9% |
| triangle_total | 0.006 | 0.8% |
| triangle_materialize | 0.117 | 15.2% |
| **triangle_gpu_total** | **0.235** | **30.5%** |
| residual_overhead | 0.128 | 16.6% |
| **wall** | **0.772** | **100%** |

### `superhub-u64-50K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.081 | 4.6% |
| layout_xy | 0.115 | 6.6% |
| layout_yz | 0.114 | 6.5% |
| layout_xz | 0.108 | 6.2% |
| **layout_total** | **0.357** | **20.4%** |
| triangle_count | 0.508 | 29.0% |
| triangle_scan | 0.014 | 0.8% |
| triangle_total | 0.006 | 0.4% |
| triangle_materialize | 0.595 | 34.0% |
| **triangle_gpu_total** | **1.123** | **64.1%** |
| residual_overhead | 0.182 | 10.4% |
| **wall** | **1.752** | **100%** |

## Absolute reductions vs `4a83ab4e`

| Cell | Wall before → after | Reduction | Layout before → after | Reduction |
|---|---:|---:|---:|---:|
| u32 10K | 13.24 → 0.73 ms | **-94.5%** (18×) | 12.78 → 0.30 ms | -97.6% |
| u32 50K | 15.08 → 1.55 ms | **-89.7%** (9.7×) | 13.77 → 0.34 ms | -97.5% |
| u64 10K | 20.77 → 0.77 ms | **-96.3%** (27×) | 20.27 → 0.32 ms | -98.4% |
| u64 50K | 22.64 → 1.75 ms | **-92.3%** (12.9×) | 21.22 → 0.36 ms | -98.3% |

## Verdict shift

* u32 10K, u64 10K: still `PipelineOverheadWarranted`, but the absolute pipeline time is now ~0.4 ms (was ~13 / 20 ms). The verdict is now driven by classifier + clone + residual overhead rather than the prior layout-pipeline cost.
* u32 50K, u64 50K: `Inconclusive` — no single bucket clears 50%. Triangle GPU phases dominate (60–64%) but no individual phase hits the threshold.

**B1 remains NOT warranted.** Materialize is 32–34% of wall in 50K cells (was 2.6–3.3% pre-fast-path — the percentage went up because everything else got proportionally smaller). Absolute materialize time is unchanged at 0.5–0.6 ms; even a 10× speedup buys 0.5 ms on a 1.5–1.75 ms wall, ~30% improvement, which is the right order of magnitude for a future kernel-internals slice but no longer the obvious next thing.

## What this means

The bench's host-deduped fixtures hit fast-path 3/3 per dispatch (assertion verified by `three_sorted_unique_inputs_yield_three_fast_path_hits`). Production callers whose relations arrive sorted+unique (typical: outputs of prior CSM joins, GroupBy, dedup) will get the same speedup.

Production callers whose relations arrive unsorted will silently fall through to the existing `dedup_full_row_recorded` path. Correctness is preserved by construction: the checker's strict-lex-increase test exactly matches the contract `dedup_full_row_recorded` would produce.

## Reproducing

```bash
cargo run -p xlog-integration --bin wcoj_phase_report \
    --features wcoj-phase-timing --release
```
