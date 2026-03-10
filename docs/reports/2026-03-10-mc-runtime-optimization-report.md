# MC Runtime Optimization Report (v0.6.0)

**Date:** 2026-03-10
**Branch:** `feat/v060-mc-optimization`
**Commits:** 8 (358a3030..ecd25e3e)

## Summary

Optimized the MC evaluation hot loop by removing per-sample D2H synchronization from sample materialization, replacing full store clone/restore with a targeted reset plan, and reducing clamped-mode count-path overhead. Separately, trimmed behavior-only test budgets to reduce test-suite turnaround time. All changes preserve MC semantics (Rejection and EvidenceClamping modes) with no API changes.

## Results

### Runtime optimization (Tasks 1–6)

Apples-to-apples measurement: same test, same sample count (1000), same machine, sequential runs:

| | main | optimized |
|---|---|---|
| Wall clock | 14.11s | 12.90s |
| **Speedup** | — | **8.6%** |

The improvement is moderate because `evaluate_program_gpu` (fixpoint evaluation) still takes 72–83% of total time. The optimization attacks only the non-eval slice.

### Test-suite turnaround (Task 7)

Behavior-only MC tests reduced from 50K–80K samples to 2K–5K (one 20K accuracy guard retained). This is a test-budget change, not a runtime-engine improvement.

| | main | optimized |
|---|---|---|
| mc.rs suite | >10 min (18 tests, 50K–80K) | ~6 min (21 tests, 2K–20K) |

## Changes by Task

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `358a3030` | Added `McTimingBreakdown` struct with env-gated profiling (`XLOG_MC_PROFILE=1`) |
| 2 | `4385447c` | Removed hot-loop row-count D2H synchronizations from `build_sample_buffers` |
| 3 | `dc73dc30` | Added `McCountStrategy` enum — skips evidence-side allocations/uploads in clamped mode |
| 4 | `938f8c3a` | Added `Executor::reset_for_mc_relations()` — targeted preserve/clear reset helper |
| 5 | `ae33e30a` | Replaced `snapshot_store`/`restore_store` with `McSampleResetPlan` — avoids full store clone per sample |
| 6 | `43717e27` | Pre-allocated pointer buffers outside closure — avoids per-sample Vec heap allocation |
| 7 | `1acefc2b` | Trimmed behavior-only test budgets (test-suite change, not runtime) |

## Timing Profile (optimized branch, 1000 samples, clamped)

| Phase | Time (µs) | % of total |
|-------|-----------|-----------|
| sampler | 193 | 0.0% |
| reset | 1,172,318 | 9.5% |
| build | 1,924,832 | 15.6% |
| eval | 8,928,998 | 72.3% |
| count | 319,715 | 2.6% |
| **total** | **12,346,056** | 100% |

## What was optimized

1. **Hot-loop row-count D2H removal** (Task 2): Removed the `device_row_count_u32()` calls from `build_sample_buffers()` — these performed synchronous GPU→CPU transfers to read scalar row counts. Replaced with host-side `num_rows()` (capacity) checks. Note: final result download and other codepaths outside this hot loop are unchanged.

2. **Store clone elimination** (Task 5): Replaced per-sample full store clone/restore with `McSampleResetPlan`. Pure deterministic relations are preserved in-place; only dynamic relations are cleared to empty buffers each sample.

3. **Clamped count-path cleanup** (Task 3): In clamped mode, evidence pointer uploads and evidence buffer allocations are reduced to 1-element sentinels. The truth/accumulate kernel path is still used (with `effective_evidence_count = 0` making the evidence loop a no-op), because the truth kernel is still needed to set query flags.

4. **Allocation reuse** (Task 6): Pre-allocated pointer vectors (`query_ptrs_buf`, `evidence_ptrs_buf`) avoid per-sample heap allocation in the count callback.

## Next performance frontier

- **eval (72–83%)**: Fixpoint evaluation dominates. Meaningful speedup requires GPU kernel-level work (fused join/project, better scheduling) — v0.7+ scope.
- **build (11–16%)**: `build_sample_buffers` still runs sort+dedup per relation per sample.
- **reset (4–10%)**: `reset_for_mc_relations` still allocates empty GPU buffers per dynamic relation per sample.
- **count (2–3%)**: Per-sample truth+accumulate kernel launches could be batched.

## Validation

```
cargo test -p xlog-prob --release --features host-io --test mc          # 21/21 PASS
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts  # 4/4 PASS
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native        # 6/6 PASS
Python pyxlog smoke test (evidence_clamping 200 [0.715])                          # PASS
CUDA certification suite                                                          # PASS
```
