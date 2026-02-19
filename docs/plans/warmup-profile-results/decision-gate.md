# Warmup Decision Gate Results

**Date:** 2026-02-19
**Hardware:** NVIDIA RTX PRO 3000 Blackwell (SM120, 12GB)
**CUDA:** 13.1, Driver 570.86.10

## Cold vs Warm Timing

| Component | Cold (no NV cache) | Warm (NV cache) | Delta |
|-----------|-------------------|-----------------|-------|
| PTX load | 1.804s | 0.012s | 1.791s |
| D4 compile | 28.668s | 18.128s | 10.540s |
| Verify | 53.600s | 54.350s | -0.749s (noise) |
| Smooth | 0.006s | 0.009s | - |
| Cache store | 0.001s | 0.000s | - |
| Free-var mask | 0.000s | 0.000s | - |
| **Total warmup** | **83.005s** | **72.857s** | **10.148s** |

## Analysis

- **PTX JIT cost: 1.8s** — NVIDIA's ComputeCache handles most JIT. Cubins save <2s.
- **D4 compile: 18-29s** — Substantial, but varies between cold/warm (d4 binary may benefit from OS page cache).
- **Verify: 54s** — Single largest cost. `validate_equivalence_gpu_gated()` dominates.
- **Circuit total: ~72s** — Overwhelmingly the bottleneck.

## Decision

**Phase 2 (disk cache) first.** Circuit compilation is 97% of warmup.
Phase 1 (cubin) deferred — saves <2s, not worth prioritizing.

Proceed: Tasks 10-12 (disk cache), then Tasks 6-9 (cubin) if needed.
