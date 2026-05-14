# G38 M_INT.1 Response Proposal

**Branch:** `feat/w3-bundle-integration`
**Current HEAD:** `cc875c64`
**Blocked gate:** G_INT M_INT.1
**Related evidence:**

- `docs/evidence/2026-05-14-g38-int-mint1-blocker.md`
- `docs/evidence/2026-05-14-g38-completion-audit.md`
- `docs/plans/2026-05-13-w34-closure-proposal.md`

This proposal does not amend `docs/plans/2026-05-14-supervisor-goal-038.md`.
It stages explicit response choices for the supervisor because M_INT.1 cannot be
honestly passed under the current goal-038 text.

## 1. Current M_INT.1 Contract

Goal-038 M_INT.1 requires:

```text
cargo bench --bench wcoj_w34_kernel_fusion
```

Target:

```text
ratio >= 1.51x
```

The contract is a W3.4 re-validation of the original `1.590x` layout+count
fusion result within a 5 percent tolerance.

## 2. Observed State

Fresh check on `feat/w3-bundle-integration`:

```text
cargo bench -p xlog-integration --bench wcoj_w34_kernel_fusion --no-run
EXIT 101

error: no bench target named `wcoj_w34_kernel_fusion` in `xlog-integration` package
```

Current tree scan:

```text
rg --files crates/xlog-integration/benches crates/xlog-cuda-tests/tests | rg 'w34|fusion'
EXIT 1
```

The old W3.4 implementation commit is an ancestor:

```text
git merge-base --is-ancestor 70d2cf5e HEAD
EXIT 0
```

But the old W3.4 surface was later retired:

```text
738ab6f2 feat(w33 G1/S1.4 M1.1): retire W3.4 fused count kernel
0754a30d feat(w33 G1/S1.4 M1.1 M1.3): retire old u32 triangle count surface
```

Current code has no W3.4 threshold route:

```text
git grep -n 'wcoj_triangle_fused_lc_u32_recorded\|W34_FUSION_THRESHOLD\|ENV_WCOJ_W34_THRESHOLD\|wcoj_triangle_fused_dispatch_count\|wcoj_triangle_unfused_dispatch_count' HEAD -- crates xlog-core
EXIT 1
```

Current triangle provider route:

```text
crates/xlog-cuda/src/provider/wcoj.rs:307-321
```

`wcoj_triangle_u32_recorded` delegates directly to the HG recorded path. The
current HG implementation is:

```text
crates/xlog-cuda/src/provider/wcoj_metadata.rs:653-760
```

It uses `wcoj_triangle_hg_u32_with_plan_recorded` and the
`wcoj_triangle_count_hg_cached_u32` kernel.

## 3. Diagnosis

M_INT.1 is not failing because the benchmark ratio regressed. It is blocked
earlier: the acceptance artifact and the production surface it measured no
longer exist on integration HEAD.

The old W3.4 result validated:

- production layout+count fusion;
- threshold auto-disable via `W34_FUSION_THRESHOLD`;
- env override `XLOG_WCOJ_W34_THRESHOLD`;
- routed fused/unfused counters;
- `wcoj_fusion_bench.rs`;
- `test_wcoj_w34_fusion.rs`.

Those are the exact surfaces later removed by W33 retirement commits. Restoring
only the benchmark target name would not revalidate W3.4 unless the retired
production route is also restored or the acceptance cell is amended to a
successor invariant.

## 4. Response Options

### Response 1 - Restore Literal W3.4 Revalidation

Restore the old W3.4 fused production route and a bench target named
`wcoj_w34_kernel_fusion`, then rerun M_INT.1 with the original `>= 1.51x` target.

Required work:

- restore or reimplement `wcoj_triangle_fused_lc_u32_recorded`;
- restore threshold selection for canonical 4-byte triangle WCOJ;
- restore `W34_FUSION_THRESHOLD` and `XLOG_WCOJ_W34_THRESHOLD`;
- restore fused/unfused routing counters and cert coverage;
- add the plan-named bench target `wcoj_w34_kernel_fusion`;
- rerun M_INT.1, then resume S_INT.3 from the first passing metric.

Tradeoff:

This keeps the current M_INT.1 acceptance text intact, but it reopens the W33
retirement decision and may conflict with process lock 5 if the restored surface
is not a single live production path.

### Response 2 - Amend M_INT.1 To A Successor Metric

Amend goal-038 M_INT.1 so G_INT validates the post-W33 replacement surface
instead of the retired W3.4 fused route.

Proposed replacement text:

```text
M_INT.1-SUCC W3.4 retired-surface resolution:

1. Verify old W3.4 implementation history is present:
   git merge-base --is-ancestor 70d2cf5e HEAD

2. Verify the W33 retirement commits are present:
   git merge-base --is-ancestor 738ab6f2 HEAD
   git merge-base --is-ancestor 0754a30d HEAD

3. Verify old W3.4 production knobs and routing counters are absent:
   git grep -n 'wcoj_triangle_fused_lc_u32_recorded\|W34_FUSION_THRESHOLD\|ENV_WCOJ_W34_THRESHOLD\|wcoj_triangle_fused_dispatch_count\|wcoj_triangle_unfused_dispatch_count' HEAD -- crates xlog-core
   expected: no matches

4. Verify the current successor triangle path builds:
   cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run

5. Verify the current successor triangle path benchmark runs:
   cargo bench -p xlog-integration --bench wcoj_w33_superhub -- --output-format bencher

Acceptance:
- commands 1, 2, 4, and 5 exit 0;
- command 3 has no matches;
- `wcoj_w33_superhub` preserves row equality for uniform and superhub cells;
- benchmark output is captured in a new M_INT.1 successor evidence artifact.
```

Tradeoff:

This does not pretend the old `1.590x` W3.4 fused route still exists. It changes
the first G_INT acceptance cell from a W3.4 ratio revalidation into a documented
retired-surface resolution plus current-successor validation. That is an
acceptance-contract amendment and requires explicit supervisor approval before
G_INT can continue.

### Response 3 - Treat G38 As STUCK

Leave goal-038 unchanged and classify G_INT as STUCK at M_INT.1.

Tradeoff:

This is the strictest reading of the current contract. It preserves the evidence
without reopening W33 retirement or changing acceptance semantics, but Phase 1
cannot reach G_PURGE or G_CLOSE under S_INT.3.

## 5. Recommendation

Response 2 is the least disruptive technical path if the supervisor accepts an
acceptance-cell amendment. It aligns the gate with current integration reality
and preserves the W33 retirement decision.

Response 1 should be chosen only if the supervisor wants the old W3.4 fused route
back as live production code.

Response 3 is correct if no acceptance-cell amendment is authorized.

## 6. Awaiting Decision

No G_INT metric after M_INT.1 should be run until one response is selected,
because goal-038 S_INT.3 requires stopping on the first failure.
