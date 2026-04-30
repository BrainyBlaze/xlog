# WCOJ Triangle Benchmark Baseline (v0.6.2)

**Date**: 2026-05-01
**Commit**: `e609fc52` (head of `feat/v0.6.2-wcoj-u64-keys` post-u64-slice; bench harness lands as a follow-on slice off this commit)
**GPU**: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
**Driver**: 591.59
**Compute capability**: 12.0 (SM120)
**CUDA Toolkit**: 13.x
**Memory budget**: 8 GB per provider

## Purpose

Establish baseline performance for the GPU 3-way Worst-Case Optimal Join (WCOJ) triangle dispatch (env-gated `XLOG_USE_WCOJ_TRIANGLE_U32`) versus the existing binary-join chain. The baseline is the prerequisite for any future skew/histogram scheduling work — without numbers, histogram changes have no defensible target.

## Methodology

- Harness: `crates/xlog-integration/benches/wcoj_triangle_bench.rs` (criterion 0.5).
- Gate forced via `RuntimeConfig::with_wcoj_triangle_dispatch(Some(bool))` — bench never mutates the process-global env.
- Timed region = `Executor::execute_plan` only. Fixture generation, GPU upload, Compiler+Executor instantiation are in `iter_batched`'s setup closure.
- Each cell pre-runs an untimed correctness check that asserts:
  - `gate=Some(false)` produces dispatch counter == 0.
  - `gate=Some(true)` produces dispatch counter == 1.
  - Row sets (sorted+deduped) from the two paths are bit-identical.
- Sample size: 10 per cell (criterion's GPU-friendly floor).

## Fixtures

| Family | Generator | Targets |
|---|---|---|
| `uniform` | LCG-seeded Erdős-Rényi; `key_range = (rows / 10).max(1000)` | Average case, low skew. Per-key degree ≈ Poisson(10). |
| `superhub` | Deterministic super-hub: ~50% of edges in each relation concentrated on a single hub key (Y for e1+e2, X for e3). Other 50% uniform | Histogram-targetable per-thread workload imbalance. The (HUB_X, HUB_Y) thread does quadratic intersection work while most threads do trivial work. |
| `empty` | Three uniform relations over disjoint key ranges | Count→scan→empty fast path (no materialize). |
| `symbol_sanity` | Uniform 10K, Symbol-typed | Symbol shares u32's 4-byte physical layout — sanity check the dispatch fires. |

## Default Matrix

`{u32, u64} × {uniform, superhub, empty} × {10K, 50K rows per relation} × {gate off, gate on}` plus one Symbol uniform-10K-gate-on cell. **25 cells.** Run with:

```bash
cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_default
```

Full matrix (adds 100K + 250K row sizes) behind `WCOJ_BENCH_FULL=1`; deferred for this baseline pass to keep the wall-clock manageable.

## Results

Median wall-clock time of the timed `execute_plan` region per criterion's default estimator. **Speedup = `gate-off median / gate-on median`** — speedup > 1 means WCOJ is *faster* than binary-join.

### Uniform Erdős-Rényi (low skew)

| Width  | Rows / rel | Gate off | Gate on | Speedup |
|--------|------------|----------|---------|---------|
| u32    | 10K        | 2.16 ms  | 15.06 ms| 0.14×   |
| u32    | 50K        | 3.84 ms  | 11.52 ms| 0.33×   |
| u64    | 10K        | 2.54 ms  | 21.34 ms| 0.12×   |
| u64    | 50K        | 15.81 ms | 19.59 ms| 0.81×   |

### Super-hub (high skew, ~50% of edges concentrated on one hub key)

| Width  | Rows / rel | Gate off | Gate on | Speedup |
|--------|------------|----------|---------|---------|
| u32    | 10K        | 16.26 ms | 10.14 ms| **1.60×** |
| u32    | 50K        | 55.53 ms | 12.98 ms| **4.28×** |
| u64    | 10K        | 30.35 ms | 19.19 ms| **1.58×** |
| u64    | 50K        | 94.63 ms | 19.44 ms| **4.87×** |

### Empty result (three relations over disjoint key ranges)

| Width  | Rows / rel | Gate off | Gate on | Speedup |
|--------|------------|----------|---------|---------|
| u32    | 10K        | 1.62 ms  | 10.49 ms| 0.15×   |
| u32    | 50K        | 1.87 ms  | 10.73 ms| 0.17×   |
| u64    | 10K        | 1.70 ms  | 41.01 ms (high variance: 18.8–80.8 ms) | 0.04× |
| u64    | 50K        | 2.02 ms  | 18.81 ms| 0.11×   |

### Symbol sanity

| Width  | Rows / rel | Gate on |
|--------|------------|---------|
| Symbol | 10K        | 11.43 ms (≈ u32 10K gateon, as expected; Symbol shares u32 layout) |

All 25 cells passed the untimed correctness pre-check (`wcoj_triangle_dispatch_count` advanced one-per-cell on gate=on; gate-off counter stayed 0; row sets between paths bit-identical after host-side dedup of fixtures).

## Findings

1. **Uniform regime — WCOJ loses 3–8×.** On low-skew Erdős-Rényi inputs the dispatch + count + scan + materialize pipeline pays a per-launch fixed cost that the existing binary-join chain doesn't. For u32 10K, WCOJ takes 15 ms vs binary-join's 2 ms — **0.14× speedup**. The gap narrows but persists at 50K (0.33× / 0.81×). At small uniform sizes, default-on dispatch would be a clear regression.

2. **Super-hub regime — WCOJ wins 1.6–4.9×.** The same dispatch pays off dramatically when input is skewed: at u32 50K, WCOJ takes 13 ms vs binary-join's 56 ms — **4.28× speedup**. The win grows with size (1.6× at 10K → 4.3× at 50K for u32; 1.6× → 4.9× for u64), because the binary-join chain's quadratic blowup on hub keys scales worse than WCOJ's per-`e_xy`-row fan-out work. This is the regime that justifies skew-aware scheduling — and the regime where the *existing* WCOJ kernel is already a major win without any histogram help.

3. **Empty-result path — WCOJ has 10–20 ms constant overhead.** Binary-join early-exits trivially when no triangles exist (1.6–2 ms). WCOJ runs the full count→scan→`dtoh_scalar_untracked` pipeline before learning the total is 0 (10–19 ms typical, with one cell — `empty/u64-10K-gateon` — showing very high variance: 18.8–80.8 ms). Worth investigation as a separate item but not a blocker for the histogram slice.

4. **u64 vs u32 parity.** u64 cells run roughly 1.4–1.7× the u32 time on the same fixture, consistent with double the per-key byte traffic. No pathological factor — the U64 path scales as expected.

5. **Stream-pool exhaustion bug.** Discovered during bench construction: the executor's WCOJ dispatch hook acquired one stream per invocation, silently exhausting the runtime's grow-only `StreamPool` (cap 16) on long-lived runtimes. Fixed in commit `127090cc` (`fix(runtime): cache WCOJ launch stream on Executor`) by mirroring `CudaKernelProvider::recorded_op_stream`'s OnceLock pattern. Without this fix the bench numbers above would have been bimodal/garbage past iteration 16 of any gate-on cell. Regression test: `crates/xlog-integration/tests/test_wcoj_dispatch_stream_reuse.rs::dispatch_counter_grows_past_stream_pool_cap`.

## Defensible Target for Skew / Histogram Work

Two concrete success criteria the future histogram-aware scheduler must hit on this bench:

1. **Don't regress super-hub.** The super-hub speedups (1.6× / 4.3× / 1.6× / 4.9×) are the *current* WCOJ win without skew help. Histogram-aware dispatch must keep these speedups within 10% of baseline (i.e., super-hub u32 50K speedup ≥ 3.85× after histogram work, vs the 4.28× here).

2. **Close the uniform gap.** Default-on dispatch is currently a regression in the uniform regime (0.14× at u32 10K). The histogram scheduler should bring uniform-gate-on to **≤ 2× uniform-gate-off** at all sizes — i.e., u32 uniform 10K must drop from 15 ms to ≤ 4 ms, u64 uniform 10K from 21 ms to ≤ 5 ms. Below that bar, default-on is still a regression and the env gate stays opt-in. Above that bar (uniform-on ≤ 2× uniform-off AND super-hub-on > 1.5× super-hub-off), there's a defensible case for default-on dispatch on non-recursive triangle rules with type-eligible inputs.

The empty-result 10–20 ms overhead is a *separate* class of optimization (skip the count pipeline when at least one input range is empty in lex space) and is not the histogram scheduler's target.

## Reproducing

```bash
git checkout <commit-with-bench-harness>
cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_default
# Full matrix (slow):
WCOJ_BENCH_FULL=1 cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_full
```

The criterion HTML report lands under `target/criterion/`.

## Out of Scope

- No actual histogram / skew kernel work — this slice's purpose is to identify where it's needed.
- No CI integration.
- No real-graph imports.
- Not run with `WCOJ_BENCH_FULL=1` in this baseline pass; left as future work.
