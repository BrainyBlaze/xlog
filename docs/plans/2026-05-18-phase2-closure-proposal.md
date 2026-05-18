# Goal-039 Phase-2 Closure Proposal

**Branch:** `feat/w6-bundle-integration-g39`
**Base:** `feat/w3-bundle-integration @ c1689d70`
**Closure candidate HEAD:** `d933200f`
**Date:** 2026-05-18
**Board action requested:** after explicit user approval only, mark W5.3, W5.4,
W6.1, and W6.2 DONE, and add W6.3, W6.4, W6.5, and W6.6 as DONE.

No closure-board edit, merge to `main`, push, or tag is included in this
proposal commit.

## Scope

Goal-039 closes the remaining Phase-2 WCOJ/DTS-DLM board work before the final
W7.1 release handoff. The integrated branch adds the cross-mode determinism
harness, widened-frontier replay gate, WCOJ architecture and user guides,
chain-join routing, K=7/K=8 clique template support, sort-label propagation with
the authorized DTS support-source fix, bounded CUDA Graph replay for Stage 4,
DTS-DLM analog fixture coverage, M37-A surface preservation, and the final
G_INT2/G_PURGE2 gates.

## Commit Chain

| Sub-goal | Commit(s) | Status |
|---|---|---|
| G_W53 / W5.3 determinism harness | `9839c039`, merged by `46806ba9` | DONE |
| G_W54 / W5.4 widened-frontier replay | `08cbcb96`, merged by `614323f7` | DONE |
| G_W61_DOC / W6.1 architecture guide | `3cd04302`, merged by `00a888bc` | DONE |
| G_W62_DOC / W6.2 user guide | `471b7f06`, merged by `ddf6cd61` | DONE |
| G_W63_CHAIN / W6.3 chain promoter | `bfae71a7`, `41f1447f`, merged by `bacca2ad` | DONE |
| G_W64_K78 / W6.4 K7/K8 templates | `134f506e`, `7459f035`, merged by `1d17f48b` | DONE |
| G_W65_SORT / W6.5 sort labels | `159f5f2b` through `79f6f259`, merged by `9a3f3f92`; DTS fix `e0324fa` on DTS-DLM branch `fix/w65-support-body-len-guard` | DONE |
| G_W66_CUDAGRAPH / W6.6 CUDA Graphs | `3473736e`, `7aa61c5b`, `5830d217`, `678b3737`, `8b11922b`, `f1933833` | DONE |
| G_W39_DTSDLM | `3f38abee` | DONE |
| G_M37A_SURFACE | `3cd64f45` | DONE |
| G_INT2 | `4d737ac5` | DONE |
| G_PURGE2 | `d933200f` | DONE |
| G_CLOSE2 | this proposal | Pending user approval for board action |
| G_E2E | not part of this board-action proposal | Pending next gate |

## Metric Summary

| Sub-goal | Result |
|---|---|
| M_W53.1-6 | PASS. Evidence `docs/evidence/2026-05-14-g39-w53-determinism-harness/report.md`: single cross-mode test file exists; WCOJ, binary fallback, and recursive outputs are bit-exact; fixed-seed evaluate loop `100/100`; dynamic rule injection `100/100`; Stage-5 rollback simulator `100/100`; no nondeterminism RCA required. |
| M_W54.1-4 | PASS. Evidence `docs/evidence/2026-05-14-g39-w54-widened-frontier-replay/run.log`: canonical DTS-DLM `data/serving/v2` artifacts discovered; replay harness committed; clean replay `1 passed; 0 failed`; evidence log has PASS marker; CUDA cert category hook passes `c15_determinism 9/9`. |
| M_W61D.1-5 | PASS. `docs/wcoj-architecture-guide.md` exists with 545 lines of architecture coverage. It covers RIR (`MultiWayJoin`, `ChainJoin`), promoter, dispatch, cost model, recursive SCC integration, HG block-slice, helper-split, stream-mux, K=7/K=8 templates, CUDA Graphs, and correct paper section citations. `ROADMAP.md` links the guide. |
| M_W62D.1-5 | PASS. `docs/wcoj-user-guide.md` exists with 425 lines. It includes a decision flowchart, eligibility/fallback paths, env/config knobs, `CompilerConfig` builders, cost-model trade-offs, threshold tuning, and recipes for skewed graphs, deep join trees, multi-rule strata, and recursive workloads. `ROADMAP.md` links the guide. |
| M_W63.1-8 | PASS with one documented caveat. Evidence `docs/evidence/2026-05-14-g39-w63-chain-prod/report.md`: m37c-prime trace subset speedup `933.200479x`; synthetic 977K speedup `15.659175x`; row equality green; zero misroutes; helper-split composition green; P4 delta-outermost cert green; largest cell under `512 MiB` budget. M_W63.4 is route/correctness preservation, not a separate triangle/cycle/clique perf A/B. |
| M_W64.1-7 | PASS. Evidence `docs/evidence/2026-05-18-g39-w64-k78-prod/README.md`: four provider entries present; Tier-1 source audit `12/12`; promoter K7/K8/K9 certs `15/15`; runtime dispatch certs `6/6`; K7/K8 row equality green; K8 ptxas register gate at `<=64`; certs run under a `64 MiB` device budget. |
| M_W65.1-5 | PASS. Evidence `docs/evidence/2026-05-18-g39-w65-authorized-dts-fix.md`: authorized DTS source-generation fix committed at `e0324fa`; 50-doc m37c-prime replay `RC=0`, `ROWS=50`, `SORT_WARNINGS=0`, `wall=1370.5s`; xlog sort-label tests `5/5`; pyxlog/xlog-integration check green; RCA preserved. |
| M_W66.1-6 | PASS. Evidence `docs/evidence/2026-05-18-g39-w66-cudagraph-integration-followup.md`: bounded m37c-scale evaluate cert speedup `27.739826x`; launch reduction `75%` structural and `96.395075%` synchronized-wall timing; determinism `100/100`; DLPack host-transfer counters `0`; peak snapshots `1.258 GiB` and `1.237 GiB`; graph recapture bounded at 4 captures, 1000 launches, 996 cache hits, 0 fallbacks. |
| M_W39D.1-8 | PASS_WITH_NOTE. Evidence `docs/evidence/2026-05-18-g39-w39-dts-dlm-analog/README.md`: DTS-DLM analog fixture registered; bundle paths `7/7`; fixture speedup `26.602774x`; four-fixture geomean `27.964641x`; row equality for all fixtures; CVs `<=0.05`; max allocation peak `2,286,157,868` bytes; generic harness hardening replaced the old 3-fixture assumption. |
| M_M37A.1-10 | PASS. Evidence `docs/evidence/2026-05-14-g39-m37a-surface-preservation/report.md`: DTS alpha tests `21 passed`; M18-D AUROC and coverage reproduced; neural zero-host-read cert green; XGCF caching exceeds the `50x` floor; MNIST-Add reference train accuracy `0.99835`; parser/network/training/induction Group-B smoke coverage `11/11`. |
| M_INT2.1-13 | PASS. Evidence `docs/evidence/2026-05-18-g39-int2/README.md`: W3.4 successor `4.209x`; W4.1 `8/8`; W5.1 trio `3/3`; W5.2 route corpus green; W2.5 default flip green; fmt/build/test green; CUDA release cert `207/207`; peak VRAM `203,423,744` bytes; DLPack zero-copy and witness-chain certs green. |
| M_PURGE2.1-9 | PASS. Evidence `docs/evidence/2026-05-18-g39-purge2/README.md`: source marker/churn scans clean; `cargo +nightly udeps --workspace --all-targets` green; strict release build with dead-code/import/variable denies green; co-author and future-version source scans clean; paper-citation coverage green. |

## KPI Status

| KPI | Status |
|---|---|
| KPI-1 m37c-prime wall-time speedup | Pending G_E2E full replay. Phase-2 bounded and analog certs are green, but this proposal does not claim the final full-pilot KPI. |
| KPI-2 determinism | Satisfied for Phase-2 board closure by W5.3 `100/100`, W5.4 replay cert, W66 bounded DTS `100/100`, and G_INT2 workspace regression. G_E2E still owns the full `100` replay KPI. |
| KPI-3 DTS-DLM API regression | Satisfied for board closure by M37-A surface preservation and the scoped W65 DTS fix evidence. G_E2E still owns final DTS call-site and functional replay checks. |
| KPI-4 closure-board state | Satisfied after user-approved board commit: W5.3/W5.4/W6.1/W6.2 flip DONE; W6.3-W6.6 are added DONE; only W7.1 remains open for final release/tag. |
| KPI-5 peak VRAM | Satisfied for Phase-2 board closure by G_INT2, W39D, W64, W66, and W65 bounded/full replay evidence. G_E2E still owns final full m37c-prime VRAM snapshot. |
| KPI-6 witness-chain recoverability | G_INT2 witness-chain cert is green; G_E2E still owns full 50-doc WMIR-derived fact recoverability. |
| KPI-7 DLPack zero-copy | G_INT2 and W66 zero-copy certs are green; G_E2E still owns final Stage-4 and Stage-5 trace assertion. |
| KPI-8 M37-A surface | Satisfied. M_M37A.1-10 green with full Group-B symbol-preservation smoke. |

## Divergences And Carry-Forward Notes

- W63 is xlog-original engineering, not directly prescribed by the paper. Its
  justification is the G_PRE measurement that made chain-shaped Stage-4
  evaluate work priority HIGH.
- W63 M_W63.4 is route/correctness preservation, not a separate performance
  A/B on triangle, cycle, and clique shapes. The relevant matcher rejects
  `MultiWayJoin`, so those shapes remain on their existing paths.
- W65 supersedes the earlier xlog-only lock-conflict escalation. The user
  authorized the narrow DTS-DLM support-source mutation needed to clear the
  sort-warning gate; the pre-authorization RCA and escalation remain preserved.
- W66 certifies a bounded DTS evaluate surface and analog fixture; the full
  m37c-prime replay remains owned by G_E2E.
- G_CLOSE2 intentionally precedes G_E2E in the Goal-039 DAG. Board closure for
  W5.3/W5.4/W6.1-W6.6 does not tag or release. W7.1 remains open until G_E2E is
  green and the user explicitly authorizes the release tag.

## Proposed Existing-Row Updates

| ID | Proposed status | Evidence summary |
|---|---|---|
| W5.3 | DONE | `test_cross_mode_determinism.rs`; WCOJ/binary/recursive equality; fixed-seed, dynamic-injection, and Stage-5 rollback loops all `100/100`. |
| W5.4 | DONE | `test_widened_frontier_replay.rs`; DTS serving-v2 artifacts discovered; replay PASS; CUDA determinism category hook PASS. |
| W6.1 | DONE | `docs/wcoj-architecture-guide.md`; ROADMAP cross-link; RIR/promoter/dispatch/cost-model/recursive plus Phase-1/Phase-2 architecture coverage. |
| W6.2 | DONE | `docs/wcoj-user-guide.md`; ROADMAP cross-link; eligibility/fallback/config/tuning decision flow and recipes. |

## Proposed New Board Entries

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|---|---|---|---|---|---|
| W6.3 | Goal-039 G_W63_CHAIN | DONE | - | Chain-join WCOJ-shape promoter and runtime route for two-atom inner-chain rules. | Evidence `docs/evidence/2026-05-14-g39-w63-chain-prod/report.md`; M_W63.1-8 green with documented M_W63.4 route/correctness caveat; integrated by `bacca2ad`. |
| W6.4 | Goal-039 G_W64_K78 | DONE | - | K=7/K=8 clique provider/template/promoter/runtime coverage. | Evidence `docs/evidence/2026-05-18-g39-w64-k78-prod/README.md`; M_W64.1-7 green; integrated by `1d17f48b`. |
| W6.5 | Goal-039 G_W65_SORT | DONE | - | Sort-label propagation and DTS-DLM support-source fix that eliminates m37c-prime sort-map warnings. | Evidence `docs/evidence/2026-05-18-g39-w65-authorized-dts-fix.md`; 50-doc replay `SORT_WARNINGS=0`; xlog certs green; DTS fix `e0324fa`. |
| W6.6 | Goal-039 G_W66_CUDAGRAPH | DONE | - | Bounded CUDA Graph capture/replay for Stage-4 hot-loop CSM paths. | Evidence `docs/evidence/2026-05-18-g39-w66-cudagraph-integration-followup.md`; M_W66.1-6 green; integrated through `f1933833`. |

## Approval Request

Approve the Phase-2 board update:

1. Mark W5.3, W5.4, W6.1, and W6.2 DONE.
2. Add W6.3, W6.4, W6.5, and W6.6 as DONE.

After approval, the only authorized follow-up for G_CLOSE2 is the board-update
commit on this branch. No merge to `main`, push, tag, or W7.1 release action is
implied. G_E2E remains the next validation gate after G_CLOSE2.
