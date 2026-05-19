# WCOJ Triangle Evidence Index (v0.6.2)

**Date**: 2026-05-01
**GPU**: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
**Driver**: 591.59
**Compute capability**: 12.0 (SM120)
**CUDA Toolkit**: 13.x
**Memory budget**: 8 GB per provider

## Current Release Conclusion

v0.6.2 ships default-on adaptive WCOJ triangle dispatch for eligible, non-recursive, 3-atom triangle rules over `u32`, `Symbol`, and `u64` keys. The dispatch path is guarded by a GPU skew classifier, falls back silently to the existing binary-join chain when the shape or workload is not eligible, and can be disabled globally with `XLOG_DISABLE_WCOJ_TRIANGLE=1`.

The final layout fast-path changed the performance profile materially: on the super-hub fixture, wall time dropped by 89-96% and layout time dropped by 97-98% versus the pre-fast-path phase report. The current bottleneck is no longer layout sort/dedup, and B1 heavy-row materialization remains deferred because materialize is a sub-millisecond bucket on the validated fixtures.

## Artifact Map

| Artifact | Purpose |
|---|---|
| This README | Release-level index and summary of the evidence chain. |
| `phase-timing-report.md` | Pre-fast-path phase timing. It invalidated B1 heavy-row materialization as the immediate next slice because layout dominated 91-97% of wall time. |
| `phase-timing-after-fast-path.md` | Post-fast-path phase timing. It shows layout time reduced by 97-98% and super-hub wall time reduced by 89-96%. |
| `crates/xlog-integration/benches/wcoj_triangle_bench.rs` | Criterion harness for Off / Force / Adaptive WCOJ triangle modes. |
| `crates/xlog-integration/src/bin/wcoj_phase_report.rs` | Feature-gated phase report binary for per-phase GPU and dispatcher timing. |

## Phase 1 - Initial Force-On Baseline

**Measured at**: `127090cc` for the runtime stream-cache fix, with the bench harness and report committed at `4576e53f`.

The first baseline compared `Mode::Off` (existing binary-join chain) with `Mode::Force` (always run WCOJ) on the same host-deduped fixtures. The goal was to determine whether a skew-aware policy was justified before adding any scheduler work.

| Fixture | u32 10K off / force | u32 50K off / force | u64 10K off / force | u64 50K off / force |
|---|---:|---:|---:|---:|
| uniform | 2.16 / 15.06 ms | 3.84 / 11.52 ms | 2.54 / 21.34 ms | 15.81 / 19.59 ms |
| super-hub | 16.26 / 10.14 ms | 55.53 / 12.98 ms | 30.35 / 19.19 ms | 94.63 / 19.44 ms |
| empty | 1.62 / 10.49 ms | 1.87 / 10.73 ms | 1.70 / 41.01 ms | 2.02 / 18.81 ms |

The result was clear: force-on WCOJ won strongly on high-skew super-hub workloads, but regressed uniform and empty-result workloads. That established the locked targets for an adaptive policy:

- Uniform and empty adaptive cells must be <= 2x the corresponding binary `Mode::Off` median.
- Super-hub adaptive speedups must preserve at least 90% of the force-on baseline speedups: u32 10K >= 1.44x, u32 50K >= 3.85x, u64 10K >= 1.42x, u64 50K >= 4.38x.

## Phase 2 - A2-Lite Adaptive Classifier

**Measured through**: `70ca8494` plus the commit-C bench harness at `8a177e87`, with the median-gate clarification finalized at `90792a2f`.

A2-lite added a combined GPU histogram classifier over the three skew-sensitive columns: `e1.col1`, `e2.col0`, and `e3.col0`. The classifier computes `max_bucket / row_count` with 64 hash-mixed buckets and dispatches WCOJ when the max score is at least `0.10`.

The classifier met the locked targets by median:

| Fixture class | Acceptance result |
|---|---|
| uniform / empty | All 8 adaptive/off ratios were in the accepted <= 2.0x range. |
| super-hub u32 10K | 1.61x speedup, locked minimum 1.44x. |
| super-hub u32 50K | 4.88x speedup, locked minimum 3.85x. |
| super-hub u64 10K | 1.65x speedup, locked minimum 1.42x. |
| super-hub u64 50K | 4.93x speedup, locked minimum 4.38x. |

This justified making adaptive dispatch the default rather than keeping WCOJ purely opt-in.

## Phase 3 - Default-On Adaptive

**Finalized at**: `9c7c3fb0`.

Default-on adaptive changed the resolver policy without changing the WCOJ kernels. With no config or env overrides, eligible non-recursive triangle rules now run the adaptive classifier. The explicit precedence is:

```text
disable > force > explicit-off > adaptive
```

Controls:

| Control | Meaning |
|---|---|
| `RuntimeConfig::with_wcoj_triangle_dispatch_disabled(Some(true))` | Hard-off kill switch. |
| `XLOG_DISABLE_WCOJ_TRIANGLE=1` | Hard-off kill switch for production ops. |
| `RuntimeConfig::with_wcoj_triangle_dispatch(Some(true))` | Force WCOJ and bypass the classifier unless disabled. |
| `XLOG_USE_WCOJ_TRIANGLE_U32=1` | Env force-on, historical name retained for compatibility. |
| `RuntimeConfig::with_wcoj_triangle_dispatch_adaptive(Some(bool))` | Programmatic adaptive override. |
| `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE=1` | Env adaptive-on override. |

Acceptance rerun:

| Cell | Default/adaptive super-hub speedup | Locked minimum |
|---|---:|---:|
| u32 10K | 1.56x | 1.44x |
| u32 50K | 4.45x | 3.85x |
| u64 10K | 1.60x | 1.42x |
| u64 50K | 4.78x | 4.38x |

Uniform and empty adaptive cells stayed within the <= 2x median gate under focused reruns. `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release` passed 13/13 under default-on adaptive.

## Phase 4 - Phase Timing and Layout Fast-Path

**Pre-fast-path timing**: `phase-timing-report.md`, finalized at `99f72bb3`.
**Post-fast-path timing**: `phase-timing-after-fast-path.md`, layout fast-path finalized at `59a5bf77`.

The phase report showed layout sort/dedup dominated the WCOJ adaptive path before any kernel-internal materialization work mattered:

| Cell | Layout share before fast-path | Triangle GPU share before fast-path |
|---|---:|---:|
| u32 10K super-hub | 96.7% | 1.3% |
| u32 50K super-hub | 92.0% | 6.1% |
| u64 10K super-hub | 97.7% | 1.0% |
| u64 50K super-hub | 93.9% | 4.8% |

The layout fast-path then added a strict sorted-and-unique adjacent-pair checker. When the proof succeeds, layout emits a recorded device-side clone instead of running the full sort/dedup pipeline.

| Cell | Wall before -> after | Layout before -> after |
|---|---:|---:|
| u32 10K super-hub | 13.24 -> 0.73 ms | 12.78 -> 0.30 ms |
| u32 50K super-hub | 15.08 -> 1.55 ms | 13.77 -> 0.34 ms |
| u64 10K super-hub | 20.77 -> 0.77 ms | 20.27 -> 0.32 ms |
| u64 50K super-hub | 22.64 -> 1.75 ms | 21.22 -> 0.36 ms |

Post-fast-path super-hub adaptive speedups versus `Mode::Off`:

| Cell | Speedup | Locked minimum |
|---|---:|---:|
| u32 10K | 10.5x | 1.44x |
| u32 50K | 24.7x | 3.85x |
| u64 10K | 19.4x | 1.42x |
| u64 50K | 33.8x | 4.38x |

## Reproducing

```bash
# Main WCOJ triangle benchmark matrix.
cargo bench -p xlog-integration --bench wcoj_triangle_bench

# Full matrix, slow.
WCOJ_BENCH_FULL=1 cargo bench -p xlog-integration --bench wcoj_triangle_bench

# Phase timing report, diagnostic feature only.
cargo run -p xlog-integration --bin wcoj_phase_report \
    --features wcoj-phase-timing --release
```

Criterion HTML output lands under `target/criterion/`.

## v0.6.2 Scope Boundaries

- Shipped: default-on adaptive triangle WCOJ for eligible non-recursive rules over `u32`, `Symbol`, and `u64`.
- Shipped: runtime kill switch, WCOJ phase timing diagnostics, skew classifier, layout fast-path, and planner/provider/runtime certification.
- Deferred: general-arity WCOJ, recursive/SCC WCOJ, cost-based join ordering, histogram-guided heavy-row scheduling, real-graph imports, and default WCOJ for non-triangle shapes.
