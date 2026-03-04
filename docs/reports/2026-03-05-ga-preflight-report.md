# v0.4.0-ga Preflight Report

**Date:** 2026-03-05
**HEAD:** `50756a4e` (main)
**Branch:** main (clean working tree)

## Test Matrix

| Suite | Tests | Result | Duration | Notes |
|-------|-------|--------|----------|-------|
| Rust workspace | all targets (excl. pyxlog) | PASS | ~45s | 0 failures |
| CUDA cert suite | run_full_certification | PASS | ~10s | 206 sub-tests |
| ILP reliability | 20/20 (4 stages x 5 seeds) | **20/20 PASS** | 1462s | reach, grandparent, colleague, plus2 |
| GA reliability | 50 seeds (attempts=2) | **PASS** | 1086s | Clopper-Pearson lower95 >= 0.929 |
| SLO scaling (enforced) | N=20/50/100/150 | **6/6 PASS** | 51s | `ILP_PERF_ENFORCE_SLO=1`, isolated run |

## SLO Detail

Run in isolation (no concurrent GPU tests). First attempt with concurrent reliability/GA tests showed ~30% wall-clock regression due to GPU contention.

| Chain N | Wall (s) | Wall SLO | Fwd p95 (us) | Fwd SLO |
|---------|----------|----------|---------------|---------|
| 20 | <15.0 | 15.0 | <250,000 | 250,000 |
| 50 | <20.0 | 20.0 | <550,000 | 550,000 |
| 100 | <35.0 | 35.0 | <900,000 | 900,000 |
| 150 | <50.0 | 50.0 | <1,500,000 | 1,500,000 |

## Changes Since Beta

### Fixed
- **Typed batch upload**: `batch_fact_membership` and `batch_tagged_credit` use schema-aware
  typed packing for all column types (I32, I64, U64, Bool, Symbol). Previously cast to `u32`.
  F32/F64 explicitly rejected. (commits `0fb9e529`..`50756a4e`)

### Changed
- **GA runtime profile**: `max_attempts` 7→2 (1447s → 436s, 3.3x speedup). (commit `803865af`)

### Added
- **SLO harness**: Parametrized `test_slo_scaling[N]` for N=20/50/100/150. (commit `d1b06577`)
- **Per-step phase timing**: 6 timed phases with p95 telemetry. (commit `2c93b2d9`)

## Known Limitations

- **SLO tests require GPU isolation**: Cannot run concurrently with other GPU tests under enforcement.
  Advisory mode (default) is safe for concurrent runs.
- **U64 columns**: Batch APIs accept `[0, i64::MAX]` range only (Python i64 input limitation).
- **F32/F64 columns**: Not supported in batch APIs (explicitly rejected).

## Verdict

All GA-blocking test suites pass. Preflight is **CLEAN** — ready for v0.4.0-ga tag.
