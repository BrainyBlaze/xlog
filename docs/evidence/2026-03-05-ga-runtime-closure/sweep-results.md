# GA Budget Sweep Results

**Date:** 2026-03-05
**Baseline:** max_attempts=7, step_budget=150, 1447s wall clock, 200/200 (CI=0.982)

## Phase 1: max_attempts sweep (step_budget=150 fixed)

### Quick Screen (GA_RELIABILITY_SEEDS=10)

| max_attempts | success | rate | wall_s (3-way GPU contention) |
|:---:|:---:|:---:|:---:|
| 5 | 40/40 | 1.0 | 470 |
| 3 | 40/40 | 1.0 | 378 |
| 2 | 40/40 | 1.0 | 271 |

All configs: 100% convergence at 10 seeds.

### Full Confirm (GA_RELIABILITY_SEEDS=50, solo GPU run)

| max_attempts | success | rate | clopper_pearson_95_lower | wall_s |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 200/200 | 1.0 | 0.981725 | 445.30 |

**Winner:** max_attempts=2, step_budget=150

## Phase 2: step_budget sweep

Skipped — max_attempts=2 already meets both constraints:
- CI lower95 (0.982) >= 0.929 gate
- Wall clock (445s) <= 600s target

## Summary

- **3.25x speedup** (1447s → 445s)
- **69% wall clock reduction**
- **Zero convergence regressions** (200/200 maintained)
- **CI preserved** (0.982 vs 0.982 baseline)

## Explanation

With 100% per-stage convergence rate, the baseline max_attempts=7 was running
5 unnecessary attempts per stage. Each attempt re-compiles, re-evaluates, and
re-converges independently. Reducing to max_attempts=2 eliminates this waste
while preserving a safety margin of 1 retry if the first attempt doesn't converge.
