# GA Runtime Closure Report

**Date:** 2026-03-05
**Plan:** `docs/plans/2026-03-04-ga-runtime-closure.md`
**Evidence:** `docs/evidence/2026-03-05-ga-runtime-closure/`

---

## Objective

Reduce `test_ga_reliability_50` wall-clock runtime from ~1447s to <=600s
while preserving statistical reliability (Clopper-Pearson lower95 >= 0.929).

## Result

**Target achieved.** Wall clock reduced from 1447s to 436s (3.3x speedup, 70% reduction).

| Metric | Before | After |
|:---|:---:|:---:|
| Wall clock (s) | 1447 | 436 |
| Success rate | 200/200 (1.0) | 200/200 (1.0) |
| CI lower95 | 0.982 | 0.982 |
| max_attempts | 7 | 2 |
| step_budget | 150 | 150 |

## Approach

### Phase 1: Measurement (Task 1)

Added per-step phase timing instrumentation to the trainer step loop covering
6 phases: apply_mask, loss_credit, loss_reduce, backward_step, membership,
convergence. Measured p95 and total across all phases.

**Finding:** All Python-side phases combined accounted for <2% of wall clock.
The forward pass (GPU evaluate) dominated at 55-75% per run. The remaining
cost was in repeated attempts: with 100% convergence and max_attempts=7,
6 of 7 attempts were wasted work.

### Phase 2: Budget Sweep (Tasks 7-8)

Based on the measurement insight, skipped Tasks 2-6 (Python dedup and
preloaded buffer optimizations) and jumped directly to the budget sweep.

**Sweep protocol:**
1. Fix step_budget=150, sweep max_attempts: 7, 5, 3, 2
2. Quick screen: 10 seeds per config
3. Full confirm: 50 seeds for most aggressive passing config

**Quick screen results (10 seeds each):**

| max_attempts | success | rate |
|:---:|:---:|:---:|
| 5 | 40/40 | 1.0 |
| 3 | 40/40 | 1.0 |
| 2 | 40/40 | 1.0 |

All configs passed with 100% convergence. Promoted max_attempts=2 directly
to full confirm.

**Full confirm (50 seeds):** 200/200, CI=0.982, 436s. PASSED.

Phase 2 step_budget sweep was skipped since the target was already met.

## Regression Testing

| Suite | Result | Time |
|:---|:---:|:---:|
| trainer + reset + sparse | 27/27 PASS | 225s |
| reliability (20/20) | 20/20 PASS | 985s |
| performance | 3/3 PASS | 52s |
| GA reliability (att=2, 50 seeds) | 200/200 PASS | 436s |

Zero regressions across all test suites.

## Changes Made

1. **Phase timing instrumentation** (`trainer.py`): 6 phase timers with p95/total telemetry.
   Commit: `2c93b2d9`

2. **GA test update** (`test_ilp_ga_reliability.py`): Default max_attempts changed from 7 to 2.
   Added wall-clock and config output to summary. Commit: `803865af`

## Deferred Work

Tasks 2-6 from the original plan (Python dedup, preloaded fact buffers) were
deferred based on measurement evidence showing <2% potential savings.
These remain available if future workloads introduce per-step Python overhead.

## Conclusion

The GA reliability test now runs in 436s (under the 600s target) with identical
statistical quality (200/200, CI=0.982). The key insight was that the dominant
cost was redundant attempts, not per-step overhead. Reducing max_attempts
from 7 to 2 captured the full available speedup without any loss in reliability.
