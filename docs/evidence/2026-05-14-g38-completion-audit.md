# G38 Completion Audit

**Objective:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Audited branch:** `feat/w3-bundle-integration`
**Audited HEAD:** `f244c888`
**Audit date:** 2026-05-14

## Source Note

The governing goal document was loaded from the main checkout at:

```text
/home/dev/projects/xlog/docs/plans/2026-05-14-supervisor-goal-038.md
```

That file is not present on the audited integration branch. The main checkout
currently has uncommitted edits to both goal-038 and goal-039, so this audit does
not copy the plan into the integration branch.

The main-checkout goal-038 diff was also inspected after the sibling-worktree
scan. It adds Phase-1 out-of-bounds item 15 for M18 / M37-A surface
preservation only; it does not amend M_INT.1 or the plan-named
`wcoj_w34_kernel_fusion` command.

## Objective Restatement

Goal 038 Phase 1 is complete only when all of the following hold together:

1. G_W35 is green or graceful-closed per S_W35.5.
2. G_W36 is green or graceful-closed per S_W36.3.
3. G_W39 has M_W39.1 through M_W39.9 green simultaneously.
4. G_INT has M_INT.1 through M_INT.11 green on `feat/w3-bundle-integration`.
5. G_PURGE has M_PURGE.1 through M_PURGE.9 green.
6. G_CLOSE has M_CLOSE.1 through M_CLOSE.5 green, including explicit user DONE
   approval and the closure-board update.
7. KPI-P1.1 through KPI-P1.7 all hold.
8. W7.1 release tagging remains user-gated.
9. Phase 2 has a durable ready-state with the integration HEAD referenced in
   goal-039.

The current branch does not satisfy this definition of done. It is stopped at
G_INT M_INT.1.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Result |
|---|---|---|
| G_W35 M_W35.1-7 or S_W35.5 graceful-close | `docs/evidence/2026-05-14-w35-line6-fanout-g38/README.md`, `measurements.tsv` | CLOSED-AS-GRACEFUL. Final paper-class direct speedup `1.432992x` and Criterion `1.450661x` missed the `>= 1.5x` gate; parity Criterion `0.936408x` missed the `>= 0.95x` guard. Experimental production code was reverted and the paper-citation justification is present. |
| G_W36 M_W36.1-5 or S_W36.3 graceful-close | `docs/evidence/2026-05-14-w36-line7-fanout-g38/README.md` | CLOSED-AS-GRACEFUL. G_W35 produced no accepted shared-memory predecessor baseline, so W3.6 has no accepted line-7 comparison baseline. |
| G_W39 M_W39.1-9 | `docs/evidence/2026-05-14-w39-paper-class-integration-g38/README.md`, `measurements.tsv` | PASS. Three fixtures pass row equality, 5/5 bundle paths, CV `<= 5%`, peak VRAM below 38 GB, recursive growth 0, and geomean direct ratio `28.389319x`. |
| G_INT M_INT.1 W3.4 re-validation | `docs/evidence/2026-05-14-g38-int-mint1-blocker.md`; fresh `cargo bench -p xlog-integration --bench wcoj_w34_kernel_fusion --no-run` | BLOCKED. The command exits 101 because no bench target named `wcoj_w34_kernel_fusion` exists in `xlog-integration`. |
| Sibling-worktree check for already-completed W3.4 revalidation work | `git worktree list --porcelain`; `find /home/dev/projects/xlog/.worktrees ...`; old `w34-fusion-impl` files | No completed G38 work found. Old W3.4 worktrees contain `wcoj_fusion_bench.rs`, not the plan-named target, and current integration HEAD has no `w34` or `fusion` bench/test files. The old bench depends on the retired W3.4 fused production surface. |
| Main-checkout plan edits | `git -C /home/dev/projects/xlog diff -- docs/plans/2026-05-14-supervisor-goal-038.md` | No M_INT.1 amendment found. The only goal-038 edit is the new M18 / M37-A preservation out-of-bounds item. |
| G_INT M_INT.2-M_INT.11 | G38 plan S_INT.3 and M_INT.1 result | NOT RUN. S_INT.3 requires running M_INT.1 through M_INT.11 sequentially and stopping on the first failure. |
| G_PURGE M_PURGE.1-M_PURGE.9 | File search for G38 purge artifact | NOT STARTED. No `docs/evidence/2026-05-14-g38-dead-code-followup.md` exists on this branch, and G_PURGE is downstream of G_INT. |
| G_CLOSE M_CLOSE.1-M_CLOSE.5 | File search for G38 closure proposal; `docs/v065-closure-board.md` | NOT STARTED. No G38 W3-bundle closure proposal exists, the board still lists W3.3/W3.5/W3.6/W3.7/W3.8/W3.9 as OPEN, and no explicit DONE approval has been applied. |
| KPI-P1.1 W3 axis 9/9 DONE | `docs/v065-closure-board.md` | NOT MET. W3.3, W3.5, W3.6, W3.7, W3.8, and W3.9 remain OPEN on the board. |
| KPI-P1.2 DoD items 1-7 | This audit table | NOT MET because G_INT, G_PURGE, and G_CLOSE are not green. |
| KPI-P1.3 W3.4 revalidated `>= 1.51x` | Fresh M_INT.1 command | NOT MET. The plan-named bench target is missing. |
| KPI-P1.4 W4.1 certs PASS | S_INT.3 stop rule | UNVERIFIED in G38 because M_INT.2 was not run after the M_INT.1 blocker. |
| KPI-P1.5 W5.1 cert trio PASS | S_INT.3 stop rule | UNVERIFIED in G38 because M_INT.3 was not run after the M_INT.1 blocker. |
| KPI-P1.6 W5.2 corpus within `+-10%` | S_INT.3 stop rule | UNVERIFIED in G38 because M_INT.4 was not run after the M_INT.1 blocker. |
| KPI-P1.7 VRAM budget/growth | W35/W39 evidence; S_INT.3 stop rule | PARTIAL. W35 and W39 report VRAM within budget, but M_INT.11 was not run after the M_INT.1 blocker. |
| W7.1 remains user-gated | `docs/v065-closure-board.md` | PRESERVED. W7.1 remains OPEN and tag-gated. |
| Phase-2 ready-state | Goal-039 predecessor reference not audited from this branch | NOT MET. G_CLOSE has not produced the Phase-1 hand-off proposal or board update. |

## Fresh Verification Commands

```text
git status --short --branch
## feat/w3-bundle-integration
```

```text
cargo bench -p xlog-integration --bench wcoj_w34_kernel_fusion --no-run
EXIT 101
error: no bench target named `wcoj_w34_kernel_fusion` in `xlog-integration` package
```

```text
rg --files crates/xlog-integration/benches crates/xlog-cuda-tests/tests | rg 'w34|fusion'
EXIT 1
```

```text
git merge-base --is-ancestor 70d2cf5e HEAD
EXIT 0
```

## Other Worktree Findings

The old W3.4 implementation work is present in sibling worktrees such as:

```text
/home/dev/projects/xlog/.worktrees/w34-fusion-impl
/home/dev/projects/xlog/.worktrees/w34-kernel-fusion
```

Those worktrees contain `crates/xlog-integration/benches/wcoj_fusion_bench.rs`
and `crates/xlog-cuda-tests/tests/test_wcoj_w34_fusion.rs`. They do not contain
the G38 plan-named `wcoj_w34_kernel_fusion` bench target, and the current
integration branch no longer has the W3.4 fused production API that the old
bench calls.

Current-code pointers:

```text
crates/xlog-cuda/src/provider/wcoj.rs:307-321
```

`wcoj_triangle_u32_recorded` now delegates directly to the HG recorded path.

```text
crates/xlog-cuda/src/provider/wcoj_metadata.rs:653-760
```

The current HG path uses `wcoj_triangle_hg_u32_with_plan_recorded` and the
`wcoj_triangle_count_hg_cached_u32` kernel.

```text
git grep -n 'wcoj_triangle_fused_lc_u32_recorded\|W34_FUSION_THRESHOLD\|ENV_WCOJ_W34_THRESHOLD\|wcoj_triangle_fused_dispatch_count\|wcoj_triangle_unfused_dispatch_count' HEAD -- crates xlog-core
EXIT 1
```

The W3.4 production routing API, threshold constants, env override, and routing
counters from the old closure proposal are absent on integration HEAD.

The absence is consistent with the blocker evidence: the old W3.4 implementation
commit `70d2cf5e` is an ancestor of integration HEAD, but later commits retired
the W3.4 fused-count surface and deleted the old bench/test files. Reusing the
old W3.4 bench on integration HEAD would require either restoring retired
production code or replacing the acceptance metric with a successor check.

## Verdict

G38 is not complete.

The branch has valid evidence for:

- G_W35 closed-as-graceful.
- G_W36 closed-as-graceful.
- G_W39 green after integration.

The branch is blocked at:

- G_INT M_INT.1: the required W3.4 re-validation target is missing on
  integration HEAD.

Per S_INT.3, later G_INT metrics, G_PURGE, and G_CLOSE should not be treated as
complete until M_INT.1 is restored/replaced or the supervisor explicitly amends
the acceptance cell.

## Response Options

1. Restore or replace the W3.4 re-validation bench and rerun G_INT from M_INT.1.
2. Amend M_INT.1 to a successor metric for the post-W33 replacement surface, then
   rerun G_INT from the amended first metric.
3. Treat G38 as STUCK under the current goal-038 contract.
