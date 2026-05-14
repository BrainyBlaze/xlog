# G38 Completion Audit

**Objective:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Audited branch:** `feat/w3-bundle-integration`
**M_INT.4 run HEAD:** `ee8b9b2e`
**Audit source note:** later commits after `ee8b9b2e` are evidence/doc-only; the
source files are unchanged from the M_INT.4 run.
**Audit date:** 2026-05-14

## Source Note

The governing goal document was loaded from the main checkout at:

```text
/home/dev/projects/xlog/docs/plans/2026-05-14-supervisor-goal-038.md
```

That file is not present on the audited integration branch. The main checkout
currently has uncommitted edits to both goal-038 and goal-039, so this audit does
not copy the plan into the integration branch.

The main-checkout goal-038 diff was later amended by the supervisor with a
post-dispatch correction replacing M_INT.1's missing
`wcoj_w34_kernel_fusion` target with the successor `wcoj_w33_superhub` metric.
The same main-checkout diff also adds Phase-1 out-of-bounds item 15 for M18 /
M37-A surface preservation.

After a sibling-worktree sweep, the G37 stop-condition audit artifacts from
`docs/g37-stop-condition-audit @ cb809400` were imported as docs-only evidence
instead of merging that older branch. A wholesale merge would remove newer G38
evidence files because the G37 audit branch forked before the G38 evidence
commits.

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
G_INT M_INT.4.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Result |
|---|---|---|
| Predecessor G37 stop-condition audit referenced by goal-038 §0.1 | `docs/evidence/2026-05-14-g37-stop-condition-audit/README.md`, `response1_readiness.md`, `review_request.md`, `status_matrix.tsv`; `docs/plans/2026-05-14-g37-iteration-2-request.md` | PRESENT. Imported from `docs/g37-stop-condition-audit @ cb809400` as docs-only artifacts after all-worktree inspection. |
| G_W35 M_W35.1-7 or S_W35.5 graceful-close | `docs/evidence/2026-05-14-w35-line6-fanout-g38/README.md`, `measurements.tsv` | CLOSED-AS-GRACEFUL. Final paper-class direct speedup `1.432992x` and Criterion `1.450661x` missed the `>= 1.5x` gate; parity Criterion `0.936408x` missed the `>= 0.95x` guard. Experimental production code was reverted and the paper-citation justification is present. |
| G_W36 M_W36.1-5 or S_W36.3 graceful-close | `docs/evidence/2026-05-14-w36-line7-fanout-g38/README.md` | CLOSED-AS-GRACEFUL. G_W35 produced no accepted shared-memory predecessor baseline, so W3.6 has no accepted line-7 comparison baseline. |
| G_W39 M_W39.1-9 | `docs/evidence/2026-05-14-w39-paper-class-integration-g38/README.md`, `measurements.tsv` | PASS. Three fixtures pass row equality, 5/5 bundle paths, CV `<= 5%`, peak VRAM below 38 GB, recursive growth 0, and geomean direct ratio `28.389319x`. |
| G_INT M_INT.1 W3.4 successor re-validation | `docs/evidence/2026-05-14-g38-int-mint1-successor.md` | PASS after supervisor correction. `wcoj_w33_superhub` compiled and ran; `superhub-50K` row equality passed with 29,539 rows and ratio `4.031791x`, above the corrected `>= 1.51x` gate. |
| Historical M_INT.1 missing-target blocker | `docs/evidence/2026-05-14-g38-int-mint1-blocker.md`; `docs/plans/2026-05-14-g38-mint1-response-proposal.md` | SUPERSEDED by the supervisor correction. The original missing `wcoj_w34_kernel_fusion` target remains absent, but M_INT.1 now uses the successor metric. |
| G_INT M_INT.2 W4.1 cert regression | `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch` | PASS by coverage. The literal multi-filter command in the goal doc is invalid Cargo syntax, so the whole target was run; it passed 8/8 including `multirec_triangle`, `multirec_4cycle`, and `selfrec_triangle`. |
| G_INT M_INT.3 W5.1 cert trio regression | `cargo test -p xlog-cuda-tests --test certification_suite --release` | PASS. Certification suite passed 1/1. |
| G_INT M_INT.4 W5.2 bench corpus regression | `docs/evidence/2026-05-14-g38-int-mint4-blocker.md`; `docs/evidence/2026-05-14-g38-int-mint4-rca.md`; `docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md`; `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` | BLOCKED. The registered W5.2 bench target exits 0 and parity is emitted, but all 12 paired current cells are outside the `+-10%` closure-baseline ratio window. The E2-prefix mitigation fixes the G38-only large 4-cycle slowdown, but 4-cycle ratios are now above the historical `+10%` bound and `5clique`/`pivot5` remain below the historical `-10%` bound. |
| G_INT M_INT.5-M_INT.11 | G38 plan S_INT.3 and M_INT.4 result | NOT RUN. S_INT.3 requires stopping on the first failure. |
| G_PURGE M_PURGE.1-M_PURGE.9 | File search for G38 purge artifact | NOT STARTED. No `docs/evidence/2026-05-14-g38-dead-code-followup.md` exists on this branch, and G_PURGE is downstream of G_INT. M_PURGE.8 must not delete/prune preserved-unmerged branch heads referenced by project memory or the closure board. |
| G_CLOSE M_CLOSE.1-M_CLOSE.5 | File search for G38 closure proposal; `docs/v065-closure-board.md` | NOT STARTED. No G38 W3-bundle closure proposal exists, the board still lists W3.3/W3.5/W3.6/W3.7/W3.8/W3.9 as OPEN, and no explicit DONE approval has been applied. |
| KPI-P1.1 W3 axis 9/9 DONE | `docs/v065-closure-board.md` | NOT MET. W3.3, W3.5, W3.6, W3.7, W3.8, and W3.9 remain OPEN on the board. |
| KPI-P1.2 DoD items 1-7 | This audit table | NOT MET because G_INT, G_PURGE, and G_CLOSE are not green. |
| KPI-P1.3 W3.4 revalidated `>= 1.51x` | M_INT.1 successor evidence | MET under the supervisor-corrected successor metric. |
| KPI-P1.4 W4.1 certs PASS | M_INT.2 evidence | MET by running the full W4.1 recursive dispatch target. |
| KPI-P1.5 W5.1 cert trio PASS | M_INT.3 evidence | MET by certification suite 1/1. |
| KPI-P1.6 W5.2 corpus within `+-10%` | M_INT.4 evidence | NOT MET. M_INT.4 is the active blocker. |
| KPI-P1.7 VRAM budget/growth | W35/W39 evidence; S_INT.3 stop rule | PARTIAL. W35 and W39 report VRAM within budget, but M_INT.11 was not run after the M_INT.4 blocker. |
| W7.1 remains user-gated | `docs/v065-closure-board.md` | PRESERVED. W7.1 remains OPEN and tag-gated. |
| Phase-2 ready-state | Goal-039 predecessor reference not audited from this branch | NOT MET. G_CLOSE has not produced the Phase-1 hand-off proposal or board update. |

## Fresh Verification Commands

```text
git status --short --branch
## feat/w3-bundle-integration
```

See `docs/evidence/2026-05-14-g38-int-mint1-blocker.md` for the historical
missing-target check and `docs/evidence/2026-05-14-g38-int-mint1-successor.md`
for the corrected successor pass.

See `docs/evidence/2026-05-14-g38-int-mint4-blocker.md` for the active blocker.

See `docs/evidence/2026-05-14-g38-int-mint4-rca.md` for the same-machine W52
comparison and root-cause assessment.

See `docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md` for the
HG-preserving u32 4-cycle mitigation and the post-fix M_INT.4 rerun.

See `docs/plans/2026-05-14-g38-mint4-response-proposal.md` for the proposed
supervisor response to the remaining literal ratio-window mismatch.

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

For M_INT.4, the old W5.2 worktree
`/home/dev/projects/xlog/.worktrees/w52-skewed-multiway-bench` still contains
the registered `w52_skewed_multiway_bench` target. Rerunning it on the same
machine shows that the historical W5.2 ratios are not reproduced by the old
branch, but G38 is additionally slower on the 4-cycle WCOJ cells. The W5.2 bench
file and hash-chain comparator are unchanged from the old branch; the WCOJ
provider/kernel path changed to the W3.3/HG work-plan implementation.

The follow-up E2-prefix mitigation keeps the W3.3 HG symbols and removes the
per-work-item linear E2 scan from the u32 4-cycle HG count/materialize kernels.
It improves `4cycle_N1000` GPU time from `33,939,237 ns` to `1,075,469 ns` and
`4cycle_N2000` GPU time from `261,392,583 ns` to `1,718,195 ns`, but the literal
M_INT.4 ratio window remains red.

The G37 stop-condition audit branch
`/home/dev/projects/xlog/.worktrees/g37-stop-condition-audit` is docs-only and
is not an ancestor of G38. Its relevant artifacts were imported directly:

```text
docs/evidence/2026-05-14-g37-stop-condition-audit/README.md
docs/evidence/2026-05-14-g37-stop-condition-audit/response1_readiness.md
docs/evidence/2026-05-14-g37-stop-condition-audit/review_request.md
docs/evidence/2026-05-14-g37-stop-condition-audit/status_matrix.tsv
docs/plans/2026-05-14-g37-iteration-2-request.md
```

## G_PURGE Boundary

G_PURGE has not started because G_INT is stopped at M_INT.4. When G_PURGE does
start, M_PURGE.8 is a dead-code follow-up artifact only. It must not remove or
prune the preserved-unmerged branch heads referenced from project memory and the
closure board, including the 22 W3-spike branches in the G11-G27 chain, the 4
W35 pre-g37 branches, the 4 W35 g37 branches, and the 2 W36 g37 branches.

## Verdict

G38 is not complete.

The branch has valid evidence for:

- G_W35 closed-as-graceful.
- G_W36 closed-as-graceful.
- G_W39 green after integration.
- Corrected G_INT M_INT.1 green.
- G_INT M_INT.2 green.
- G_INT M_INT.3 green.

The branch is blocked at:

- G_INT M_INT.4: W5.2 bench corpus regression is outside the `+-10%`
  closure-baseline window.

Per S_INT.3, later G_INT metrics, G_PURGE, and G_CLOSE should not be treated as
complete until M_INT.4 is fixed or the supervisor explicitly amends the
acceptance cell.

## Response Options

1. Amend M_INT.4 if the supervisor accepts a changed W5.2 regression criterion;
   the amendment should separate same-machine baseline drift from successor-HG
   behavior. The G38-only 4-cycle slowdown now has a production mitigation.
   Proposal: `docs/plans/2026-05-14-g38-mint4-response-proposal.md`.
2. Continue W5.2 corpus work on `5clique`/`pivot5` and any remaining ratio-window
   mismatch without violating W3.3 HG source-audit locks.
3. Treat G38 as STUCK at M_INT.4 under the current goal-038 contract.
