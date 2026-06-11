# S1e — K=5/K=6 clique count fusion (u32 keys): gate evidence

Date: 2026-06-11
Branch: `feat/factorized-kclique-count-fusion`
Commit under measurement: `4e59f508` (provider + executor wiring complete)
Host GPU: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU (WSL2),
driver 591.86, max SM clock 3090 MHz.

## Gate

Fused `q(R, count(*)) :- <complete K_5 body>` grouped by the
variable-order root must be **>= 3x** faster than the unfused production
path (planned clique materialize + recorded groupby count) on a skewed
hub fixture.

## Method

`cargo test -p xlog-cuda-tests --test test_wcoj_clique_groupby_root_count \
  --release -- --ignored --nocapture`

Fixture: hub root 0 whose e01 fans to `n_x` V1 values; V2..V4 draw from
16-wide bands with all band pairs present (4096 completions per leader
row → materialized clique row count `n_x * 4096`), plus 1000 uniform
background rows per edge so the group column is not single-valued.
5 reps per path, median reported; both paths warmed once; oracle +
parity asserted on every warmup (the timing loop re-executes the same
entries). One stream per case, reused across reps.

## GPU state (recorded around the runs)

| moment | temp | SM clock | util |
|---|---|---|---|
| before run 1 | 60 C | 1942 MHz | 13 % (desktop background) |
| idle just before run 1 start | 57 C | 352 MHz | 22 % |
| after run 1 | 58 C | 1942 MHz | 15 % |

The GPU drives a WSL2 laptop desktop, so a 13–22 % background
utilization floor exists; per the measurement discipline the gate was
re-run a second time to confirm the margin is not a scheduling artifact.

## Results

Run 1:

| case | unfused median | fused median | speedup |
|---|---|---|---|
| clique5_hub_500 (n_x=500, 2.05M clique rows) | 153.701 ms | 42.776 ms | **3.59x** |
| clique5_hub_1000 (n_x=1000, 4.10M clique rows) | 353.290 ms | 111.393 ms | **3.17x** |

Run 2 (repeat, same binary, same fixtures):

| case | unfused median | fused median | speedup |
|---|---|---|---|
| clique5_hub_500 | 168.680 ms | 56.014 ms | **3.01x** |
| clique5_hub_1000 | 334.210 ms | 107.800 ms | **3.10x** |

## Verdict

- Gate **PASS**: every measured cell in both runs is >= 3x
  (3.59x / 3.17x / 3.01x / 3.10x).
- Honest margin note: the second run's hub_500 cell sits at 3.01x —
  the gate holds but without much headroom on this laptop GPU with a
  desktop-compositing background load. The dominant saving is the
  eliminated materialize phase (clique rows never hit memory); the
  fused count kernel performs the identical traversal as the unfused
  count phase, so the speedup scales with the materialize+groupby share
  of the unfused pipeline (~2.05–4.10M output rows here).
- Parity: asserted against a host brute-force K-clique oracle AND the
  unfused baseline inside the measurement test (warmup phase) and in
  the four non-ignored parity tests of the same file (small K5/K6,
  skewed hub, unsorted+duplicated input normalization).

## Boundary (what this evidence does and does not cover)

- Covered: K=5 and K=6 fused count-by-root at the u32/Symbol (4-byte)
  width-class, provider level + executor/promoter wiring (e2e cells in
  `crates/xlog-integration/tests/test_wcoj_clique_groupby_fusion.rs`:
  fused fires counter==1 with kill-switch parity for K5+K6;
  plan-dependent root (cooled-stats V3 root) fuses on V3 and declines
  on V0 with parity; incomplete clique bodies never promote or fuse).
- The >=3x gate was measured for K=5 (as specified). K=6 fused has
  parity coverage but no separate perf gate claim.
- NOT covered / deferred: u64-key clique count fusion, K=7/K=8 fusion
  (no fused kernels), clique sum/min/max fusion, and any non-count
  aggregate over clique bodies. These decline silently to the unfused
  path by design.
