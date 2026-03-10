# MC Runtime Optimization Report (v0.6.0)

**Date:** 2026-03-10
**Branch:** `feat/v060-mc-optimization`
**Commits:** 7 (358a3030..1acefc2b)

## Summary

Reduced MC evaluation overhead by eliminating per-sample D2H synchronization barriers, replacing full store clone/restore with a targeted reset plan, and trimming test sample budgets. All changes preserve MC semantics (Rejection and EvidenceClamping modes) with no API changes.

## Changes by Task

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `358a3030` | Added `McTimingBreakdown` struct with env-gated profiling (`XLOG_MC_PROFILE=1`) |
| 2 | `4385447c` | Removed `device_row_count_u32()` — eliminated 3 D2H reads/relation/sample from `build_sample_buffers` |
| 3 | `dc73dc30` | Added `McCountStrategy` enum — skips evidence-side allocations/uploads in clamped mode |
| 4 | `938f8c3a` | Added `Executor::reset_for_mc_relations()` — targeted preserve/clear reset helper |
| 5 | `ae33e30a` | Replaced `snapshot_store`/`restore_store` with `McSampleResetPlan` — avoids full store clone per sample |
| 6 | `43717e27` | Pre-allocated pointer buffers outside closure — avoids per-sample Vec heap allocation |
| 7 | `1acefc2b` | Trimmed behavior-only test budgets (80K→2K-5K samples, one 20K accuracy guard) |

## Timing Profile (1000 samples, clamped evidence)

Test: `test_evidence_clamping_all_samples_count`

| Phase | Time (µs) | % of total |
|-------|-----------|-----------|
| sampler | 356 | 0.01% |
| reset | 131,699 | 3.7% |
| build | 404,788 | 11.4% |
| eval | 2,966,822 | 83.4% |
| count | 54,503 | 1.5% |
| **total** | **3,558,168** | 100% |

Eval dominates at 83%. Reset is now 3.7% (down from ~5% pre-optimization). Build is 11.4%.

## Timing Profile (100 samples, rejection mode)

Test: `test_sampling_method_in_result_metadata` (runs both clamped + rejection)

| Mode | Total (µs) | eval % | reset % | build % | count % |
|------|-----------|--------|---------|---------|---------|
| Clamped | 121,920 | 77.7% | 4.6% | 13.4% | 3.5% |
| Rejection | 314,904 | 83.4% | 3.6% | 11.3% | 1.6% |

## Optimizations Achieved

1. **D2H elimination** (Task 2): Removed 3 synchronous D2H reads per relation per sample from `build_sample_buffers`. Uses `num_rows()` (host-side capacity) instead of `device_row_count_u32()` (D2H sync).

2. **Store clone elimination** (Task 5): Replaced O(N_relations × buffer_size) per-sample cloning with O(N_dynamic_relations × empty_alloc) targeted reset. Pure deterministic relations are preserved in-place.

3. **Evidence-side skip** (Task 3): In clamped mode, evidence pointer uploads and evidence buffer allocations are minimized. Evidence count accumulates correctly through the loop (evidence_ok is always 1 when effective_evidence_count = 0).

4. **Allocation reuse** (Task 6): Pre-allocated pointer vectors (`query_ptrs_buf`, `evidence_ptrs_buf`) avoid per-sample heap allocation in the count callback.

## Residual Hotspots

- **eval (83%)**: Fixpoint evaluation dominates. Further optimization requires GPU kernel-level work (fused join/project, better scheduling), which is v0.7+ scope.
- **build (11%)**: `build_sample_buffers` still runs sort+dedup per relation per sample. Could benefit from a fused sample-expand kernel that skips the compact→sort→dedup pipeline.
- **reset (3.7%)**: `reset_for_mc_relations` still allocates empty GPU buffers per dynamic relation per sample. A buffer pool or pre-allocated empty sentinel could eliminate this.
- **count (1.5%)**: Per-sample truth kernel launch + accumulate kernel launch. Could be batched across multiple samples in future.

## Test Suite Impact

- All 31 MC tests pass (21 mc.rs + 4 gpu_mc_device_counts.rs + 6 mc_gpu_native.rs)
- Python smoke test passes (`evidence_clamping 2000 [0.6975]`)
- mc.rs test suite runtime reduced via sample budget trimming (Task 7)

## Validation

```
cargo test -p xlog-prob --release --features host-io --test mc          # 21/21 PASS
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts  # 4/4 PASS
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native        # 6/6 PASS
Python pyxlog smoke test                                                          # PASS
```
