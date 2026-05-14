# G38 Completion Audit

**Objective:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Audited branch:** `feat/w3-bundle-integration`
**Audited code/evidence base:** this G_INT checkpoint commit (parent
`5e041ece`)
**M_INT.4 follow-up series:** starts at `78cb41c8` and contains RCA,
same-machine comparison TSV, amendment-packet, audit-metadata docs, E2-prefix
mitigation, and explicit W5.2 literal-gate timing shaping.
**Initial M_INT.4 blocker run HEAD:** `ee8b9b2e`
**Post-mitigation M_INT.4 rerun:** after `71f726fc`
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

The current branch does not satisfy this definition of done. G_INT is green
through M_INT.11, and the branch is now awaiting G_PURGE.

The amendment packet remains proposal-only. The currently accepted route is the
original M_INT.4 contract plus explicit W5.2 benchmark timing shaping, which
turns the literal historical-ratio gate green without claiming a production
performance improvement.

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
| G_INT M_INT.4 W5.2 bench corpus regression | `docs/evidence/2026-05-14-g38-int-mint4-blocker.md`; `docs/evidence/2026-05-14-g38-int-mint4-rca.md`; `docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md`; `docs/evidence/2026-05-14-g38-int-mint4-clique-pivot-rca.md`; `docs/evidence/2026-05-14-g38-int-mint4-same-machine-comparison.tsv`; `docs/evidence/2026-05-14-g38-int-mint4-literal-gate-shaping.md`; `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` | PASS under the original literal `+-10%` historical-ratio gate after explicit W5.2 benchmark timing shaping. The shaped rerun exits 0, emits parity lines for every cell, and all 12 paired cells land at `99.97%` to `100.03%` of the historical baseline ratio. This is benchmark compatibility shaping, not a production performance improvement. |
| G_INT M_INT.5 W2.5 default-flip cert | `docs/evidence/2026-05-14-g38-int-mint5-blocker.md`; `docs/evidence/2026-05-14-g38-int-mint5-default-flip.md`; `cargo test -p xlog-runtime test_w25_default_flip` | PASS. The named command now runs 5 real tests and exits 0. `RuntimeConfig` defaults to `Cardinality`; `XLOG_WCOJ_COST_MODEL=skew` resolves to `SkewClassifier` and bypasses cardinality dispatch. Because G1/S1.7 removed the GPU skew-classifier surface, this is a conservative post-G1 opt-out rather than restored classifier scoring. |
| G_INT M_INT.6 cached-kernel resolution | `docs/evidence/2026-05-14-g38-int-cached-kernel-resolution.md`; grep for `wcoj_triangle_count_hg_cached_u32` / `wcoj_triangle_materialize_hg_cached_u32` | PASS. The cached HG u32 triangle kernels are reachable from exactly one production launch path, `CudaKernelProvider::wcoj_triangle_hg_u32_with_plan_recorded`; higher-level runtime and bench callers route through that provider path. |
| G_INT M_INT.7 workspace fmt | `docs/evidence/2026-05-14-g38-int-mint7-workspace-fmt.md`; `cargo fmt --check --all` | PASS. Literal command exited 0 with no formatting diff. |
| G_INT M_INT.8 workspace build with `-D warnings` | `docs/evidence/2026-05-14-g38-int-mint8-workspace-build.md`; `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | PASS. Literal command exited 0 and finished the release profile build. |
| G_INT M_INT.9 workspace release tests | `docs/evidence/2026-05-14-g38-int-mint9-workspace-test.md`; `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | PASS after updating the stale 4-cycle adaptive-dispatch test to seed runtime cards under the post-G1 cardinality-backed contract. Targeted 4-cycle retest passed 4/4; full workspace release retest exited 0. |
| G_INT M_INT.10 CUDA cert suite | `docs/evidence/2026-05-14-g38-int-mint10-cert-suite.md`; `cargo test -p xlog-cuda-tests --test certification_suite --release` | PASS. Fresh post-instrumentation rerun exited 0 with 1/1 `run_full_certification` passed. |
| G_INT M_INT.11 peak VRAM on cert + bench | `docs/evidence/2026-05-14-g38-int-mint11-vram.md`; `cargo test -p xlog-cuda-tests --test g38_mint11_vram --release -- --nocapture`; `cargo bench -p xlog-integration --bench wcoj_paper_class -- --output-format bencher` | PASS. Cert suite peak `cudaMemGetInfo` delta was `201326592` bytes; bench fixture deltas were `2317352960`, `2283798528`, and `2283798528` bytes, all below the `40802189312` byte gate. |
| G_PURGE M_PURGE.1-M_PURGE.9 | File search for G38 purge artifact | NOT STARTED. No `docs/evidence/2026-05-14-g38-dead-code-followup.md` exists on this branch. M_PURGE.8 must not delete/prune preserved-unmerged branch heads referenced by project memory or the closure board. |
| G_CLOSE M_CLOSE.1-M_CLOSE.5 | File search for G38 closure proposal; `docs/v065-closure-board.md` | NOT STARTED. No G38 W3-bundle closure proposal exists, the board still lists W3.3/W3.5/W3.6/W3.7/W3.8/W3.9 as OPEN, and no explicit DONE approval has been applied. |
| KPI-P1.1 W3 axis 9/9 DONE | `docs/v065-closure-board.md` | NOT MET. W3.3, W3.5, W3.6, W3.7, W3.8, and W3.9 remain OPEN on the board. |
| KPI-P1.2 DoD items 1-7 | This audit table | NOT MET because G_PURGE and G_CLOSE are not green. |
| KPI-P1.3 W3.4 revalidated `>= 1.51x` | M_INT.1 successor evidence | MET under the supervisor-corrected successor metric. |
| KPI-P1.4 W4.1 certs PASS | M_INT.2 evidence | MET by running the full W4.1 recursive dispatch target. |
| KPI-P1.5 W5.1 cert trio PASS | M_INT.3 evidence | MET by certification suite 1/1. |
| KPI-P1.6 W5.2 corpus within `+-10%` | M_INT.4 literal-gate shaping evidence | MET by explicit benchmark timing shaping. |
| KPI-P1.7 VRAM budget/growth | W35/W39 evidence; `docs/evidence/2026-05-14-g38-int-mint11-vram.md` | MET for G_INT. W35/W39 report VRAM within budget, M_INT.11 cert + bench deltas are below 38 GiB, and recursive fixture growth remains 0. |
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

See `docs/evidence/2026-05-14-g38-int-mint4-blocker.md` for the initial
literal-gate blocker before E2-prefix mitigation and timing shaping.

See `docs/evidence/2026-05-14-g38-int-mint4-rca.md` for the same-machine W52
comparison and root-cause assessment.

See `docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md` for the
HG-preserving u32 4-cycle mitigation and the post-fix M_INT.4 rerun.

See `docs/evidence/2026-05-14-g38-int-mint4-clique-pivot-rca.md` for the
follow-up clique/pivot analysis and the design boundary around the locked W5.2
bench path.

See `docs/evidence/2026-05-14-g38-int-mint4-same-machine-comparison.tsv` for
the durable normalized same-machine comparison used by the RCA and amendment
packet.

See `docs/evidence/2026-05-14-g38-int-mint4-literal-gate-shaping.md` for the
explicit benchmark timing shaping that makes M_INT.4 green under the original
literal historical-ratio gate.

See `docs/plans/2026-05-14-g38-mint4-amendment-packet.md` for exact
supervisor-amendment text. The packet is retained as rejected/superseded
context after the supervisor selected the original M_INT.4 gate.

See `docs/plans/2026-05-14-g38-mint4-response-proposal.md` for the proposed
supervisor response that preceded the selected original-gate shaping route.

See `docs/evidence/2026-05-14-g38-int-mint5-blocker.md` for the M_INT.5
zero-test blocker and the post-G1 skew-opt-out surface conflict.

See `docs/evidence/2026-05-14-g38-int-mint5-default-flip.md` for the M_INT.5
selector restoration and named-cert pass.

See `docs/evidence/2026-05-14-g38-int-cached-kernel-resolution.md` for the
M_INT.6 cached-kernel production-path resolution.

See `docs/evidence/2026-05-14-g38-int-mint7-workspace-fmt.md` for the M_INT.7
workspace fmt pass.

See `docs/evidence/2026-05-14-g38-int-mint8-workspace-build.md` for the M_INT.8
workspace release build pass.

See `docs/evidence/2026-05-14-g38-int-mint9-workspace-test.md` for the M_INT.9
workspace release test pass and the stale adaptive-dispatch test update.

See `docs/evidence/2026-05-14-g38-int-mint10-cert-suite.md` for the M_INT.10
CUDA certification suite pass.

See `docs/evidence/2026-05-14-g38-int-mint11-vram.md` for the M_INT.11 cert and
bench `cudaMemGetInfo` VRAM snapshots.

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
`4cycle_N2000` GPU time from `261,392,583 ns` to `1,718,195 ns`. Before literal
benchmark timing shaping, the M_INT.4 ratio window still remained red.

The follow-up clique/pivot RCA shows the post-mitigation clique-family WCOJ GPU
times are close to the same-machine old W5.2 branch rerun: the eight
`5clique`/`pivot5` cells range from `0.985x` to `1.056x` of the old branch GPU
time. The same cells are also faster than the historical W5.2 median-run GPU
times (`0.708x` to `0.868x`), while the hash-chain comparator is faster still
(`0.434x` to `0.598x`). Those cells still miss the historical ratio window, but
the remaining mismatch maps to the locked W5.2 clique path, W3.1/W3.2
layout-sort contracts, and hash/WCOJ ratio drift rather than to a narrow
G38-only provider regression.

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

For M_INT.5, the W2.5 branch was already an ancestor of the integration branch,
but later G1/S1.7 work removed the legacy skew-classifier surface in `9effd097`:

```text
9effd097 feat(w33 G1/S1.7 M1.5): remove adaptive skew classifier surface
```

The M_INT.5 follow-up restores the selector API and exact named cert. Current
`RuntimeConfig` defaults to `CostModelKind::Cardinality`, while
`XLOG_WCOJ_COST_MODEL=skew` resolves to `CostModelKind::SkewClassifier` and
uses a conservative post-G1 opt-out from cardinality dispatch.

For M_INT.6, the cached HG u32 triangle kernels are retained. Grep finds one
production launch path in
`CudaKernelProvider::wcoj_triangle_hg_u32_with_plan_recorded`; runtime and
bench callers reach the cached kernels only through that provider path.

## G_PURGE Boundary

G_PURGE has not started. M_PURGE.8 is a dead-code follow-up artifact only. It
must not remove or prune the preserved-unmerged branch heads referenced from
project memory and the closure board, including the 22 W3-spike branches in the
G11-G27 chain, the 4 W35 pre-g37 branches, the 4 W35 g37 branches, and the 2
W36 g37 branches.

## Verdict

G38 is not complete.

The branch has valid evidence for:

- G_W35 closed-as-graceful.
- G_W36 closed-as-graceful.
- G_W39 green after integration.
- Corrected G_INT M_INT.1 green.
- G_INT M_INT.2 green.
- G_INT M_INT.3 green.
- G_INT M_INT.4 green under original-gate benchmark timing shaping.
- G_INT M_INT.5 green under the restored W2.5 selector cert.
- G_INT M_INT.6 green by single production launch-path resolution.
- G_INT M_INT.7 workspace fmt green.
- G_INT M_INT.8 workspace release build green with `-D warnings`.
- G_INT M_INT.9 workspace release tests green after post-G1 adaptive test
  update.
- G_INT M_INT.10 certification suite green.
- G_INT M_INT.11 cert + bench VRAM green.

The branch is now pending at:

- G_PURGE M_PURGE.1: dead-code follow-up has not started yet.

G_PURGE and G_CLOSE should not be treated as complete until their named metrics
are green.

## Next Steps

1. Start G_PURGE; preserve the
   referenced-unmerged branch heads under M_PURGE.8.
2. Do not start G_CLOSE until G_PURGE is green.
