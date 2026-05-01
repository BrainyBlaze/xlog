# WCOJ Phase Timing Report — v0.6.2

**Date**: 2026-05-01
**Measured at**: head of `feat/v0.6.2-wcoj-phase-timing` (off `9c7c3fb0`)
**GPU**: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU (SM120, driver 591.59)
**Iters per cell**: 30 samples on a single long-lived `Executor` (default-on adaptive); first 5 discarded as warmup; median reported.

## Purpose

Decide whether to invest in B1 heavy-row offload before any kernel work, by measuring per-phase wall-clock breakdown of WCOJ adaptive dispatch on the bench's super-hub fixtures.

Locked decision rule:

| Bucket dominant ≥ 50% of wall | Action |
|---|---|
| `triangle_materialize_ms` | Proceed with B1 (count ≥ 4, dual-grid deterministic materialization). |
| `triangle_count_ms + scan_ms + total_ms` | Design count/materialize work scheduling instead. |
| `layout_total_ms + classifier_ms + residual_overhead_ms` | Optimize pipeline/control-plane before kernel scheduling. |

Buckets:

* `classifier_ms` — `wcoj_triangle_skew_score_*` provider call (Instant)
* `layout_xy_ms` / `layout_yz_ms` / `layout_xz_ms` / `layout_total_ms` — `wcoj_layout_*_recorded` per slot, summed (Instant)
* `triangle_count_ms` / `scan_ms` / `total_ms` / `materialize_ms` — CUDA events inside `wcoj_triangle_*_recorded`
* `triangle_gpu_total_ms` — sum of the four GPU buckets
* `execute_plan_wall_ms` — `Instant` from `try_dispatch_wcoj_triangle` entry to dispatch success
* `residual_overhead_ms` = `wall - classifier - layout_total - triangle_gpu_total`

## Results

### `superhub-u32-10K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.088 | 0.7% |
| layout_xy | 4.698 | 35.5% |
| layout_yz | 3.740 | 28.2% |
| layout_xz | 4.987 | 37.7% |
| **layout_total** | **12.776** | **96.5%** |
| triangle_count | 0.076 | 0.6% |
| triangle_scan | 0.014 | 0.1% |
| triangle_total | 0.006 | 0.0% |
| triangle_materialize | 0.100 | 0.8% |
| **triangle_gpu_total** | **0.197** | **1.5%** |
| residual_overhead | 0.171 | 1.3% |
| **wall** | **13.238** | **100%** |

### `superhub-u32-50K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.115 | 0.8% |
| layout_xy | 4.728 | 31.3% |
| layout_yz | 4.838 | 32.1% |
| layout_xz | 5.077 | 33.7% |
| **layout_total** | **13.771** | **91.3%** |
| triangle_count | 0.412 | 2.7% |
| triangle_scan | 0.014 | 0.1% |
| triangle_total | 0.006 | 0.0% |
| triangle_materialize | 0.505 | 3.3% |
| **triangle_gpu_total** | **0.937** | **6.2%** |
| residual_overhead | 0.227 | 1.5% |
| **wall** | **15.082** | **100%** |

### `superhub-u64-10K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.088 | 0.4% |
| layout_xy | 6.906 | 33.3% |
| layout_yz | 6.738 | 32.4% |
| layout_xz | 7.054 | 34.0% |
| **layout_total** | **20.268** | **97.6%** |
| triangle_count | 0.097 | 0.5% |
| triangle_scan | 0.014 | 0.1% |
| triangle_total | 0.006 | 0.0% |
| triangle_materialize | 0.117 | 0.6% |
| **triangle_gpu_total** | **0.236** | **1.1%** |
| residual_overhead | 0.170 | 0.8% |
| **wall** | **20.765** | **100%** |

### `superhub-u64-50K`

| bucket | ms | % wall |
|---|---:|---:|
| classifier | 0.110 | 0.5% |
| layout_xy | 7.641 | 33.7% |
| layout_yz | 6.535 | 28.9% |
| layout_xz | 6.837 | 30.2% |
| **layout_total** | **21.222** | **93.7%** |
| triangle_count | 0.509 | 2.2% |
| triangle_scan | 0.014 | 0.1% |
| triangle_total | 0.006 | 0.0% |
| triangle_materialize | 0.594 | 2.6% |
| **triangle_gpu_total** | **1.124** | **5.0%** |
| residual_overhead | 0.227 | 1.0% |
| **wall** | **22.644** | **100%** |

## Verdict (per locked decision rule)

| Cell | Layout share | Triangle GPU share | Materialize share | Verdict |
|---|---:|---:|---:|---|
| superhub u32 10K | 96.5% | 1.5% | 0.8% | **`PipelineOverheadWarranted`** |
| superhub u32 50K | 91.3% | 6.2% | 3.3% | **`PipelineOverheadWarranted`** |
| superhub u64 10K | 97.6% | 1.1% | 0.6% | **`PipelineOverheadWarranted`** |
| superhub u64 50K | 93.7% | 5.0% | 2.6% | **`PipelineOverheadWarranted`** |

**Unanimous: 4/4 cells route to "optimize pipeline overhead before kernel scheduling".**

## Implications

**B1 heavy-row offload is NOT warranted.** It targets the materialize bucket (0.6–3.3% of wall). A 10× speedup on materialize buys ~0.05–0.5 ms back, on a 13–23 ms wall. The slice spec's locked min for super-hub speedups (e.g. 1.6×) cannot be met by optimizing a 3% bucket.

**Count/materialize work scheduling is also NOT warranted.** Same problem: count + scan + total combined are 0.6–6.2% of wall.

**The actual cost is `wcoj_layout_*_recorded` — sort + dedup of each input — running 3× per dispatch and consuming 91–97% of wall clock.** This was invisible before phase timing because it executes inside the dispatcher between the classifier call and the triangle kernel, not under any of the three "WCOJ kernel" buckets the prior baseline measured.

## Recommended next slice (post phase-timing)

Three candidate directions, in order of likely impact:

1. **Layout caching / elision**. If a given input buffer participates in multiple triangle dispatches without mutation, the second-and-later dispatches re-sort + re-dedup data that's already sorted+deduped from the first. A buffer-fingerprint or "already-sorted" flag could let the dispatcher skip layout when the input is known-good. Bench fixtures fit this pattern (same uploaded buffers on every iter); production workloads do too whenever a relation is reused across rules.

2. **Layout pre-pass at relation upload**. If WCOJ adaptive is the default, run layout once on `put_relation` for any relation that'll see triangle dispatch. This pushes the ~14 ms cost to a one-time event at relation registration, off the hot path. Tradeoff: we pay layout for relations that *don't* hit a triangle dispatch.

3. **Layout fast-path for already-sorted inputs**. Detect via a single-block scan whether the input is already sorted+unique; if so, skip the full sort_recorded + dedup_full_row_recorded pipeline. Free for inputs that arrive sorted (common for outputs of prior CSM joins, GroupBy, etc.).

All three are pipeline-overhead optimizations. None require kernel-internals work. None invalidate the existing A2-lite + default-on adaptive guarantees.

**B1 is shelved unless and until layout overhead is reduced enough that materialize becomes a meaningful share of wall clock.**

## Reproducing

```bash
cargo run -p xlog-integration --bin wcoj_phase_report \
    --features wcoj-phase-timing --release
```

Production builds (no `wcoj-phase-timing` feature) cannot build this binary, by design — phase timing is diagnostic-only with zero runtime surface in production.
