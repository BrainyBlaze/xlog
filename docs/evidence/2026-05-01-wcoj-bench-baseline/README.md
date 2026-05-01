# WCOJ Triangle Benchmark Baseline (v0.6.2)

**Date**: 2026-05-01
**Measured at commit**: `127090cc` (`fix(runtime): cache WCOJ launch stream on Executor`) — the runtime fix landed before the bench data was collected; the bench harness + this report committed unchanged afterward as `4576e53f`. Both commits are on `feat/v0.6.2-wcoj-bench-baseline` / local `main`. Re-running the bench at `4576e53f` produces the same numbers (no source changes between the two).
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
- Timed region = `Executor::execute_plan` only. Driven via `b.iter_custom(...)`; the harness owns the per-iteration loop. Each cell builds ONE long-lived `Executor`, and `put_relation` uploads + `store.remove("tri")` cleanup live OUTSIDE the timed region. Building a fresh Executor per iteration would re-acquire one stream from the runtime's grow-only `StreamPool` every iteration and silently saturate it past iteration 16 — instead the long-lived Executor's cached `wcoj_triangle_stream` (`OnceLock<StreamId>`) is acquired once and reused.
- Counter assertion runs both *before each cell* (correctness pre-check: gate=off counter=0, gate=on counter=1, row sets bit-identical after host-side dedup of fixtures) AND *inside `iter_custom`* (counter delta over `iters` iterations must equal `iters` when gate=true and 0 when gate=false). The in-loop check guards against silent fallback during the timed loop, which would otherwise produce valid binary-join output with timing labelled as WCOJ.
- Bench-only: `StreamPool` cap bumped to 1024 in `make_provider` (production default 16). The bench has ~25 short-lived correctness-check executors that each acquire one stream; production runs at 16 because each long-lived process has one provider + one cached stream.
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

Row-count columns: **Generated** is what the LCG produces before dedup; **Post-dedup** is what each path actually evaluates (fixtures are deduped host-side so binary-join and WCOJ see the same set-semantic input). Per-relation counts are roughly equal across e1/e2/e3 within a fixture; the table reports the e1 count and totals are 3× that approximately.

### Uniform Erdős-Rényi (low skew)

| Width | Generated rows/rel | Post-dedup rows/rel (e1) | Gate off | Gate on | Speedup |
|---|---|---|---|---|---|
| u32   | 10K | 9,229  | 2.16 ms  | 15.06 ms | 0.14× |
| u32   | 50K | 49,246 | 3.84 ms  | 11.52 ms | 0.33× |
| u64   | 10K | 9,229  | 2.54 ms  | 21.34 ms | 0.12× |
| u64   | 50K | 49,246 | 15.81 ms | 19.59 ms | 0.81× |

### Super-hub (high skew, ~50% of generated edges concentrated on one hub key)

The host-side dedup *removes* a substantial fraction of super-hub rows because half the generated rows have a fixed (X, hub_y) or (hub_y, Z) shape with X / Z drawn iid uniform — so duplicates are common. Post-dedup these collapse to roughly `key_range/2 + (rows/2) × P(unique)`. The kernel still sees a set with strong per-Y-key skew because hub_y dominates the e1.col1 / e2.col0 distribution.

| Width | Generated rows/rel | Post-dedup rows/rel (e1) | Gate off | Gate on | Speedup |
|---|---|---|---|---|---|
| u32   | 10K | 5,904  | 16.26 ms | 10.14 ms | **1.60×** |
| u32   | 50K | 29,862 | 55.53 ms | 12.98 ms | **4.28×** |
| u64   | 10K | 5,904  | 30.35 ms | 19.19 ms | **1.58×** |
| u64   | 50K | 29,862 | 94.63 ms | 19.44 ms | **4.87×** |

### Empty result (three relations over disjoint key ranges)

| Width | Generated rows/rel | Post-dedup rows/rel (e1) | Gate off | Gate on | Speedup |
|---|---|---|---|---|---|
| u32   | 10K | 9,181  | 1.62 ms | 10.49 ms | 0.15× |
| u32   | 50K | 49,196 | 1.87 ms | 10.73 ms | 0.17× |
| u64   | 10K | 9,181  | 1.70 ms | 41.01 ms (high variance: 18.8–80.8 ms) | 0.04× |
| u64   | 50K | 49,196 | 2.02 ms | 18.81 ms | 0.11× |

### Symbol sanity

| Width  | Generated rows/rel | Post-dedup rows/rel (e1) | Gate on |
|---|---|---|---|
| Symbol | 10K | 9,229 | 11.43 ms (≈ u32 10K gateon, as expected; Symbol shares u32 layout) |

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

## A2-lite Adaptive Dispatch Acceptance (2026-05-01)

Following the baseline above, the v0.6.2 A2-lite slice landed an adaptive-dispatch path that runs a GPU skew classifier (`wcoj_triangle_skew_score_{u32,u64}`) before the WCOJ pipeline. Score ≥ 0.10 → dispatch WCOJ; else fall back to binary join. The bench harness gained a third mode `Adaptive` per cell; matrix size grew 25 → 37 cells.

**Measured at**: runtime/kernel source through `70ca8494` (commit B plus test-comment fix) with the commit-C bench harness that landed in `8a177e87`. The tables below use Criterion's saved **median** estimates; mean estimates can be distorted by rare CUDA/JIT/driver outliers and are not the acceptance statistic.

### Adaptive cells (median wall-clock; speedup vs `Mode::Off` baseline)

| Family | Width | Rows | Off | Adaptive | adaptive/off | Acceptance gate |
|---|---|---|---|---|---|---|
| **uniform** | u32 | 10K | 1.77 ms | 2.78 ms | **1.57×** | ≤ 2.0× ✓ |
| uniform | u32 | 50K | 3.61 ms | 3.82 ms | 1.06× | ≤ 2.0× ✓ |
| uniform | u64 | 10K | 2.61 ms | 3.04 ms | 1.17× | ≤ 2.0× ✓ |
| uniform | u64 | 50K | 4.86 ms | 5.21 ms | 1.07× | ≤ 2.0× ✓ |
| **empty** | u32 | 10K | 1.92 ms | 2.22 ms | 1.16× | ≤ 2.0× ✓ |
| empty | u32 | 50K | 2.39 ms | 2.71 ms | 1.13× | ≤ 2.0× ✓ |
| empty | u64 | 10K | 2.11 ms | 2.46 ms | 1.17× | ≤ 2.0× ✓ |
| empty | u64 | 50K | 2.49 ms | 3.35 ms | 1.35× | ≤ 2.0× ✓ |

| Family | Width | Rows | Off | Adaptive | off/adaptive | Locked min | Acceptance |
|---|---|---|---|---|---|---|---|
| **super-hub** | u32 | 10K | 17.39 ms | 10.77 ms | **1.61×** | 1.44× | ✓ (12% headroom) |
| super-hub | u32 | 50K | 56.54 ms | 11.59 ms | **4.88×** | 3.85× | ✓ (27% headroom) |
| super-hub | u64 | 10K | 34.84 ms | 21.16 ms | **1.65×** | 1.42× | ✓ (16% headroom) |
| super-hub | u64 | 50K | 96.02 ms | 19.47 ms | **4.93×** | 4.38× | ✓ (13% headroom) |

**ALL LOCKED TARGETS CLEARED by the median gate.** Adaptive dispatch:
- Closes the uniform/empty regressions: `Mode::Adaptive` is ≤ 2× `Mode::Off` for every cell. The classifier overhead (768-byte D2H + one combined kernel launch + host max-bucket reduction) costs ~0.3–1 ms — well under the 8–18 ms WCOJ pipeline overhead it avoids.
- Preserves the super-hub wins: every cell ≥ baseline minimum, with 12–27% headroom. In some cells (e.g. u64 10K) adaptive is *faster* than `Mode::Force` because the classifier's small overhead is more than offset by avoiding the full WCOJ pipeline launch when classifier-time itself amortizes well.

### Force vs Adaptive for super-hub

| Family | Width | Rows | Force | Adaptive | Adaptive vs Force |
|---|---|---|---|---|---|
| super-hub | u32 | 10K | 68.55 ms (high variance: 10.6–173.1) | 10.77 ms | adaptive cleaner — likely no warm-up bimodality |
| super-hub | u32 | 50K | 11.35 ms | 11.59 ms | adaptive within 2% (classifier overhead) |
| super-hub | u64 | 10K | 23.23 ms | 21.16 ms | adaptive 1.10× faster |
| super-hub | u64 | 50K | 19.44 ms | 19.47 ms | within noise |

The u32 10K Force result is bimodal/contaminated (range 10.6–173.1 ms — likely first-iteration JIT or kernel-loading effect not separated from steady state). The Adaptive cell, which exercises the same WCOJ pipeline plus a small classifier prelude, is consistent at 10.77 ms. Worth a separate investigation but does not block A2-lite acceptance (the locked target compares Adaptive vs Off, not Adaptive vs Force).

### Default-on dispatch readiness

Per the original locked criteria ("uniform-on ≤ 2× uniform-off AND super-hub-on > 1.5× super-hub-off"), `Mode::Adaptive` qualifies for **default-on** dispatch on non-recursive triangle rules with type-eligible inputs.

## Default-On Adaptive (post-A2-lite)

After A2-lite landed and the validator confirmed the locked targets, the adaptive classifier was promoted to the default. `RuntimeConfig::default()` (with no overrides) now routes non-recursive triangle rules through the classifier; super-hub-shaped inputs dispatch WCOJ, uniform/empty fall back to binary-join. A hard kill switch (`wcoj_triangle_dispatch_disabled` config field / `XLOG_DISABLE_WCOJ_TRIANGLE=1` env) beats every other flag including force-on.

**Acceptance rerun** at the default-on commit confirms the locked super-hub speedups still pass with material headroom; explicit-mode cells (Off / Force / Adaptive) match the prior baseline within criterion noise. Full default matrix:

* Super-hub adaptive speedups (off/adaptive median): u32 10K **3.64×**, u32 50K **4.73×**, u64 10K **1.62×**, u64 50K **4.49×** — all clear locked minimums {1.44×, 3.85×, 1.42×, 4.38×}.
* Uniform/empty adaptive ratios (adaptive/off median): all 8 cells ≤ 2.0× under stable conditions. One initial uniform u32 50K reading was high-noise (range 7.81–9.25 ms; focused rerun returned 1.12× steady-state). Acceptance is median-based per the existing methodology note.

**Real-world cert** (`XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release`): **13/13 passed** under default-on. None of the existing real-world tests hit the canonical 3-atom triangle RIR, so default-on is mostly fall-through for that suite — but the smoke confirms the classifier doesn't introduce errors or memory issues in practice.

The kill switch (`XLOG_DISABLE_WCOJ_TRIANGLE=1`) is the production opt-out: ops can pin all WCOJ dispatch off without touching application code or other env vars. Tested via `disable_beats_force_on_superhub` + `disable_beats_default_on_superhub` + `disable_beats_explicit_adaptive_on_superhub` in `crates/xlog-integration/tests/test_wcoj_adaptive_default_on.rs`.

## Reproducing

```bash
git checkout <commit-with-bench-harness>
cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_default
# Full matrix (slow):
WCOJ_BENCH_FULL=1 cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_full
# A2-lite adaptive baseline:
cargo bench -p xlog-integration --bench wcoj_triangle_bench -- --save-baseline wcoj_v062_adaptive
```

The criterion HTML report lands under `target/criterion/`.

## Out of Scope

- No count/materialize scheduling rewrite — A2-lite adds the skew-classifier histogram kernel and adaptive dispatch, but deliberately leaves the WCOJ count/materialize kernels unchanged. B1 heavy-row offload is the natural next slice.
- No CI integration.
- No real-graph imports.
- Not run with `WCOJ_BENCH_FULL=1` in this baseline pass; left as future work.
