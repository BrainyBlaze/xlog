# SLO Scaling Harness Design

**Date:** 2026-03-05
**Goal:** Parametrized performance SLO test across chain lengths N=20/50/100/150

---

## Approach

Extend `test_ilp_performance.py` with a parametrized `test_slo_scaling[N]` test that
replaces the existing `test_forward_performance_slo_n50_smoke`. Uses the same
`_build_reach_chain_source()` and `_chain_positives()` helpers.

## SLO Targets

Based on empirical baseline (3 seeds: 7, 42, 99) with ~2x headroom:

| N | Wall-clock SLO | Forward p95 SLO |
|---|---|---|
| 20 | 15s | 250,000 us |
| 50 | 20s | 550,000 us |
| 100 | 35s | 900,000 us |
| 150 | 50s | 1,500,000 us |

## Enforcement

- Advisory by default: prints results table, does not assert
- Hard-assert when `ILP_PERF_ENFORCE_SLO=1` (matches existing pattern)
- Convergence is always asserted (must converge regardless of enforcement mode)

## Config

Standard profile: `max_attempts=2, step_budget=150, seed=7, deterministic=True, max_active_rules=16`

## Changes

1. Replace `test_forward_performance_slo_n50_smoke` with parametrized `test_slo_scaling`
2. Add `SLO_TARGETS` dict with wall-clock and forward p95 thresholds per N
3. Print summary table with pass/fail indicators
