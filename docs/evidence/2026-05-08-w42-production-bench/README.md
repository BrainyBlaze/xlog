# W4.2 Production-Kernel Benchmark — Nested-Loop vs Hash

**Branch:** `feat/w42-nested-loop-join` (Step 12).
**Bench file:** `crates/xlog-integration/benches/w42_production_nested_loop_bench.rs`.
**Date:** 2026-05-08.
**Predecessor evidence:** `docs/evidence/2026-05-07-w42-bench-spike/README.md` (spike on `bench-spike/w42-nested-loop` HEAD `9c0cefc6`, unmerged).

## Purpose

Validate that the production W4.2 nested-loop join — `nested_loop_join_v2_inner_u32_1key` in `crates/xlog-cuda/src/provider/relational.rs` — wins ≥ 2× vs `hash_join_v2` on the production eligibility envelope. This is D7 acceptance criterion #6 from the iteration-4 plan.

## Why a separate bench from the spike

The spike at `bench-spike/w42-nested-loop` measured 4–6× wins with a 1-col-no-payload kernel, intentionally minimal to falsify the hypothesis "does nested-loop ever beat hash". F-W42-3 caveat: production has multi-col arity + payload + the `gather_buffer_by_indices` materialization step after the kernel. Per-row cost can compress the speedup. The plan set the production acceptance bar at ≥ 2× to absorb that compression.

## Methodology

* **Provider:** built once. 8 GiB device budget. 1024-stream pool. Same setup as `wcoj_*_bench`.
* **Fixtures:** 3-col U32 buffers (key at col 0; two payload columns). Left covers keys `[0..num_left)`. Right covers keys `[num_left/2..num_left/2 + num_right)` so output has 50%-match rate at the canonical alignment.
* **Pre-cell parity** (outside timed region): `provider.nested_loop_join_v2_inner_u32_1key(left, right, 0, 0)` and `provider.hash_join_v2(left, right, &[0], &[0], JoinType::Inner)` both produce 6-col `combine_schemas` output. Comparison via `BTreeSet<[u32; 6]>` equality. Panic on mismatch.
* **Timed region:** the provider call only. Same uploaded buffers across both paths within a cell. `b.iter_custom` so allocation/upload/cleanup live outside the measurement.
* **Criterion config:** `sample_size=20`, `measurement_time=3s`, `warm_up_time=500ms`.

## Matrix

**Eligible cells** (Cartesian ≤ `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000`): nested-loop AND hash both benched.

| `(L, R)` | Cartesian | Matched output |
|----------|-----------|----------------|
| (100, 100) | 10K | 50 |
| (500, 500) | 250K | 250 |
| (1000, 1000) | 1M | 500 |
| (2000, 2000) | 4M (threshold ceiling) | 1000 |

**Above-threshold cells** (Cartesian > 4M): hash-only — the production `nested_loop_join_v2_inner_u32_1key` returns `Err` when the caller violates the eligibility threshold, and the W4.2 dispatcher routes these to hash. Above-threshold rows establish hash's scaling baseline for comparison.

| `(L, R)` | Cartesian | Notes |
|----------|-----------|-------|
| (5000, 5000) | 25M | symmetric, 6.25× over threshold |
| (10000, 1000) | 10M | asymmetric, 2.5× over threshold |

## Results

### Eligible cells (D7 acceptance #6 surface)

| Cell | Nested-loop median | Hash v2 median | NL/Hash | Speedup |
|------|--------------------|----------------|---------|---------|
| L=100 R=100 | 262.34 µs | 726.33 µs | 0.361 | **2.77×** |
| L=500 R=500 | 288.37 µs | 828.90 µs | 0.348 | **2.87×** |
| L=1000 R=1000 | 323.82 µs | 901.33 µs | 0.359 | **2.78×** |
| L=2000 R=2000 | 428.18 µs | 899.74 µs | 0.476 | **2.10×** |

**Minimum speedup across eligible cells:** 2.10× (at the threshold ceiling `L=R=2000`).
**D7 acceptance #6 (≥ 2× on eligible cells):** ✅ MET on all 4 cells.

### Above-threshold (hash-only baseline)

| Cell | Hash v2 median |
|------|----------------|
| L=5000 R=5000 | 971.93 µs |
| L=10000 R=1000 | 980.25 µs |

Hash's wall time stays near 1 ms across the entire 10K–25M Cartesian range — consistent with the spike's F2 finding that hash's cost is dominated by fixed per-launch overhead (the multi-phase pipeline `compute_composite_hash → bucket_count → exclusive_scan → scatter → probe_count → exclusive_scan → probe_materialize`) rather than work proportional to input size.

### Parity (correctness witness)

All 4 eligible cells passed `BTreeSet<[u32; 6]>` row-set equality between nested-loop and hash outputs. Output row counts match the expected 50% match rate (50 / 250 / 500 / 1000).

## Findings

### F1. ≥ 2× win confirmed across the eligible envelope.

The minimum measured speedup (2.10× at `L=R=2000`) clears the D7 acceptance bar. The maximum (2.87× at `L=R=500`) is roughly half the spike's headline of 4–6×, consistent with F-W42-3's caveat that production multi-col + gather imposes higher per-row cost than the 1-col-spike kernel. The compression is real but bounded.

### F2. Speedup compresses at the threshold ceiling.

Speedup curve across `L=R=N` cells:
* L=R=100: 2.77×
* L=R=500: 2.87×
* L=R=1000: 2.78×
* L=R=2000: 2.10×

The plateau holds steady from N=100 to N=1000 then drops at N=2000. The drop matches the algorithmic crossover model: nested-loop scales as `O(L · R)` while hash's per-launch cost is roughly constant. As L · R grows, nested-loop's kernel time grows; hash's wall time stays near floor; speedup compresses.

The threshold value (4M Cartesian) lands exactly where the speedup is reaching its lower limit on this fixture, confirming the iteration-4 plan's threshold choice. Pushing the threshold higher would risk dropping below 2× in adjacent untested cells.

### F3. Hash's per-launch overhead dominates at this scale.

Hash medians range from 726 µs (10K Cartesian) to 980 µs (25M Cartesian) — only ~35% increase across a 2500× input-size range. The work-proportional component is small relative to the launch-overhead floor (~700 µs in the production environment). This is a different regime from the spike's 2.7 ms hash floor, likely due to environment differences (the spike ran on a freshly-warmed CUDA context; this bench ran after extensive prior workload). The relative comparison (NL vs hash within a single bench run) is what matters for D7 #6.

### F4. Production crossover is OUTSIDE the eligible range, as expected.

At the threshold ceiling (L=R=2000 = 4M Cartesian), nested-loop is 2.10×. Extrapolating from the slope (NL ~= 215 µs + 0.05 µs × Cartesian-millions; Hash ~= 700-1000 µs constant), the algorithmic crossover lands somewhere around L=R=4000-5000 (16M-25M Cartesian), well above the threshold. Hash's hash-only data at L=R=5000 (972 µs) suggests nested-loop at the same size would take ~1.4 ms — about 1.4× slower than hash. So the 4M threshold sits comfortably below the actual crossover.

### F5. Schema concatenation works correctly across both paths.

`provider.hash_join_v2 Inner` and `provider.nested_loop_join_v2_inner_u32_1key` both produce 6-col `combine_schemas(left, right) = [k_l, p1_l, p2_l, k_r, p1_r, p2_r]` output. `BTreeSet<[u32; 6]>` parity holds at every cell. F-W42-1's drop-in semantics (output schema matches hash's full concatenation, no key-column dropping) is empirically witnessed.

## Reproduction

```bash
cd /home/dev/projects/xlog/.worktrees/w42-nested-loop-join
cargo bench --bench w42_production_nested_loop_bench -p xlog-integration
```

Total runtime ~70s (4 eligible cells × 2 paths × 3.5s + 2 above-threshold cells × 1 path × 3.5s + parity-check overhead).

## Conclusion

D7 acceptance criterion #6 (nested-loop wins ≥ 2× vs hash on eligible cells) is empirically met on all 4 cells in the production envelope, with margin (minimum 2.10× at the threshold ceiling, maximum 2.87× elsewhere). The threshold choice (`NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000`) is justified: speedups stay above 2× throughout, while the algorithmic crossover sits comfortably outside.

The W4.2 implementation is ready for the closure proposal (Step 13). No threshold adjustment, kernel tuning, or further optimization needed inside W4.2's chartered scope.
