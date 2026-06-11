# D1: Aggregate-Fused WCOJ (group-by-root count) — Implementation Plan

Branch: `feat/factorized-wcoj-aggregates` (worktree `.worktrees/factorized-agg-wcoj`)
Origin: `docs/plans/2026-06-11-factorized-hypergraph-research.md` §4 D1 / §5 S1.
Discipline: S1 measurement gate evaluated before executor wiring is considered
closable; all steps TDD; evidence recorded under `docs/evidence/`.

## Semantics

`q(X, C) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` with `C = count` grouped by the
variable-order ROOT (X): compute per-X counts of distinct (Y,Z) completions
WITHOUT materializing the triangle rows. Theory basis (verified): one-pass
semiring aggregate propagation over the variable order; group-by variable is
the root, satisfying the constant-delay/group-by iff-condition.

## Fused pipeline (all recorded; zero tracked transfers)

1. `wcoj_triangle_hg_work_plan_u32_recorded` (existing).
2. NEW kernel `wcoj_triangle_groupby_root_count_hg_u32`: identical traversal
   to `wcoj_triangle_count_hg_u32`, but each match does
   `atomicAdd(&out_row_counts[xy_idx], 1)` (u32, order-insensitive =>
   deterministic values; precedent: `groupby_sum` uses integer atomicAdd).
   `out_row_counts` is `n_xy` long, zero-initialized (`memset_zeros`).
3. Temp 2-col buffer (X = dtod copy of `e_xy.col0`, C = row_counts), n_xy rows.
4. `compare_const_mask_recorded::<u32>(temp, 1, 0, Gt)` → device mask.
5. `compact_buffer_by_device_mask_counted_recorded` → only contributing rows
   (still sorted by X; compaction is stable).
6. `groupby_multi_agg_recorded(compacted, [0], [(1, Sum)])` → (X: U32, C: U64).

Asymptotics: reduction runs over n_xy (input rows), never |triangles|.
Output schema matches the unfused baseline
(`wcoj_triangle_hg_u32_recorded` → `groupby_multi_agg([0],[(_,Count)])`).

## Steps (each with verification)

1. **RED**: provider parity test
   `xlog-cuda-tests/tests/test_wcoj_groupby_root_count.rs` — small fixture +
   skew fixture; fused result equals (a) host brute-force oracle and
   (b) unfused materialize+groupby baseline. Fails: entry missing.
2. **GREEN**: kernel (wcoj.cu) + manifest + const + provider entry
   `wcoj_triangle_groupby_root_count_u32_recorded` (wcoj_metadata.rs).
   Check: step-1 test green; `cargo test -p xlog-cuda-tests` green.
3. **S1 measurement**: timing test (same file, `s1_measurement_` prefix,
   prints fused vs unfused wall-clock on hub-skew fixtures at 10K/50K rows +
   small uniform). Gate: ≥5× on skew, ≤1.1× regression small-uniform.
   Evidence: `docs/evidence/2026-06-11-s1-aggregate-fused-wcoj/README.md`.
   Gate FAIL => stop, report, branch preserved as evidence.
4. **Executor wiring (production)**: RIR-level recognition in
   `xlog-runtime` — `GroupBy { input: MultiWayJoin(triangle shape),
   key_cols == [0 (root X)], aggs == [(_, Count)] }` dispatches the fused
   path; every structural mismatch declines silently to the existing
   materialize+groupby path; kernel errors via `wcoj_decline_on_error`
   (counted, XLOG_WCOJ_STRICT honored). Kill switch
   `XLOG_DISABLE_WCOJ_GROUPBY_FUSION`. Counter
   `wcoj_groupby_fusion_dispatch_count` + accessor.
   Check (TDD): integration test — fused fires (counter==1) with row-set
   parity vs forced-unfused run; decline cases (non-root key, non-count agg,
   kill switch) keep counter==0 with identical results.
5. **Full validation**: workspace tests + cert quick smoke green; no new
   warnings. Docs: wcoj-architecture-guide section + language-reference note.
6. Commits: conventional, incremental per step.

## Out of scope (deferred, stated explicitly)

u64/Symbol key variants beyond U32/Symbol-as-u32; 4-cycle/k-clique fusion;
sum/min/max aggregates; recursive-context fusion; D2-D4 directions.
