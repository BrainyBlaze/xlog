# Goal-039 Phase-2 Closure Proposal v3

**Branch:** `feat/w6-bundle-integration-g39`
**Base:** `feat/w3-bundle-integration @ c1689d70`
**Closure candidate before this proposal:** `9416d69c`
**Date:** 2026-05-18
**Supersedes:** `docs/plans/2026-05-18-phase2-closure-proposal-v2.md`
at proposal commit `9416d69c`

**Board action requested:** after explicit user approval only:

1. Mark W1.1 DONE.
2. Mark W5.3, W5.4, W6.1, and W6.2 DONE.
3. Add W6.3, W6.4, W6.5, and W6.6 as DONE.

No closure-board edit, merge to `main`, push, or tag is included in this
proposal commit.

## Scope

This v3 proposal carries forward the Phase-2 board action requested by v2 and
adds the residual W1.1 closure-board process row to the approval request. The
v2 proposal is preserved as the pre-W1.1-audit record. This document is the
current closure proposal for Goal-039.

The W1.1 addition is necessary because Goal-039 §3.16 says W7.1 tag handoff is
valid only when every non-W7.1 closure-board item is DONE. The current board
still has W1.1 as `IN-PROGRESS`, even though its acceptance artifacts now exist:
`docs/v065-closure-board.md` is committed, process rules are locked, and
`ROADMAP.md` points to the board at `ROADMAP.md:1304`.

## Commit Chain

| Sub-goal | Commit(s) | Status |
|---|---|---|
| W1.1 closure-board process row | `55702bb8`, amended by later board commits through `9e1ec480` | Pending user approval for DONE |
| G_W53 / W5.3 determinism harness | `9839c039`, merged by `46806ba9` | DONE |
| G_W54 / W5.4 widened-frontier replay | `08cbcb96`, merged by `614323f7` | DONE |
| G_W61_DOC / W6.1 architecture guide | `3cd04302`, merged by `00a888bc` | DONE |
| G_W62_DOC / W6.2 user guide | `471b7f06`, merged by `ddf6cd61` | DONE |
| G_W63_CHAIN / W6.3 chain promoter | `bfae71a7`, `41f1447f`, merged by `bacca2ad` | DONE |
| G_W64_K78 / W6.4 K7/K8 templates | `134f506e`, `7459f035`, merged by `1d17f48b` | DONE |
| G_W65_SORT / W6.5 sort labels | `159f5f2b` through `79f6f259`, merged by `9a3f3f92`; DTS-DLM fix `e0324fa` | DONE |
| G_W66_CUDAGRAPH / W6.6 CUDA Graphs | `3473736e`, `7aa61c5b`, `5830d217`, `678b3737`, `8b11922b`, `f1933833` | DONE |
| G_W39_DTSDLM | `3f38abee` | DONE |
| G_M37A_SURFACE | `3cd64f45` | DONE |
| G_INT2 | `4d737ac5` | DONE |
| G_PURGE2 | `d933200f` | DONE |
| G_CLOSE2 proposal v1 | `d96b411c` | Preserved pre-G_E2E proposal |
| G_E2E | `a8ce46a5` | DONE |
| G_CLOSE2 proposal v2 | `9416d69c` | Preserved pre-W1.1-audit proposal |
| G_CLOSE2 proposal v3 | this proposal | Pending user approval for board action |

## W1.1 Audit

Current board state before this proposal:

- `docs/v065-closure-board.md` status tally: W1.1 is `IN-PROGRESS`.
- `docs/v065-closure-board.md` W1.1 required deliverable: closure board exists,
  is committed, has process rules locked, and enumerates the board items.
- `docs/v065-closure-board.md` W1.1 acceptance gate: board committed on `main`,
  `ROADMAP.md` points to it, and user approves tally plus amendments.

Current artifact evidence:

| W1.1 criterion | Evidence | Status |
|---|---|---|
| Closure board exists and is committed | `docs/v065-closure-board.md`; original commit `55702bb8` | SATISFIED |
| Process rules locked | `docs/v065-closure-board.md` §Process Rules | SATISFIED |
| Board tracks the closure items after amendments | Current board includes prior DONE rows, W6.7, W5.3/W5.4/W6.1/W6.2, and W7.1; v3 requests W6.3-W6.6 additions | SATISFIED after approved v3 board commit |
| ROADMAP points to board | `ROADMAP.md:1304` links `docs/v065-closure-board.md` | SATISFIED |
| User approves tally plus amendments | This v3 proposal requests explicit approval | PENDING |

Because W1.1 is itself a board item and W7.1 is blocked by every other board
item, W1.1 must be part of the final board-update approval request.

## G_E2E Evidence

Evidence file:
`docs/evidence/2026-05-18-g39-e2e/README.md` at `a8ce46a5`.

Raw artifact paths:

- Candidate metrics:
  `/tmp/g39-e2e-m37c-prime/g39-e2e-50doc-armc-instrumented-20260518-r2/eval/g39_e2e_metrics.json`
- Strict baseline manifest:
  `/tmp/g39-e2e-baseline-f62188b7-m37c/g39-e2e-baseline-f62188b7-50doc-armc-r1/eval/manifest.json`
- Stage-4 frozen replay summary:
  `/tmp/g39-e2e-m37c-prime/g39-e2e-m37c-stage4-capture/replay_100_summary.json`

| Metric | Status | Raw result |
|---|---|---|
| M_E2E.1 wall-time speedup vs `f62188b7` | PASS | `1466.4236392974854 / 278.3887164592743 = 5.267539783754166x` |
| M_E2E.2 determinism | PASS | m37c-prime Stage-4 frozen bundle `100/100`, one digest; full 50-doc semantic digest `2/2` |
| M_E2E.3 DTS-DLM API regression | PASS | pyxlog call-site scan plus DTS regression slice green |
| M_E2E.4 DTS-DLM serving regression equivalent | PASS | DTS regression slice `179 passed` |
| M_E2E.5 wheel install | PASS | `maturin build --release --compatibility linux --auditwheel skip` exit 0; pip reinstall exit 0 |
| M_E2E.6 peak VRAM | PASS | `12,820,480,000 bytes` <= `38 GB` |
| M_E2E.7 witness recoverability | PASS | `89/89` derived facts recoverable; `0` bad derived premise edges |
| M_E2E.8 DLPack zero-copy | PASS | xlog provider host-transfer delta `0` DtoH/HtoD bytes and calls |
| M_E2E.9 compile median | PASS | `13.499273 ms` <= `50 ms` |
| M_E2E.10 cross-doc cleanup | PASS | final-after minus initial-after allocated bytes `-29,930,496` |
| M_E2E.11 behavioral drift | PASS | `f62188b7` and `d96b411c` captured-bundle digests match |
| M_E2E.12 explicit W4.1 cert | PASS | `8 passed; 0 failed`, including the 3 W4.1 positive certs |

## KPI Status

| KPI | Status |
|---|---|
| KPI-1 m37c-prime wall-time speedup | PASS. G_E2E records `5.267539783754166x` vs strict `f62188b7`. |
| KPI-2 determinism | PASS. G_E2E records `100/100` frozen Stage-4 replays and `2/2` full 50-doc semantic output digest match. |
| KPI-3 DTS-DLM API regression | PASS. DTS pyxlog call-site scan plus functional regression slice green. |
| KPI-4 closure-board state | Pending user board approval. This proposal requests W1.1 DONE, W5.3/W5.4/W6.1/W6.2 DONE, and W6.3-W6.6 added DONE; W7.1 remains user-gated. |
| KPI-5 peak VRAM | PASS. G_E2E peak is `11.94000244140625 GiB`, below `38 GB`. |
| KPI-6 witness-chain recoverability | PASS. `89/89` derived facts recoverable, `0` bad derived premise edges. |
| KPI-7 DLPack zero-copy | PASS. Provider host-transfer delta is zero for DtoH and HtoD bytes/calls. |
| KPI-8 M37-A surface | PASS. M_M37A.1-10 green; no xlog-induce changes. |

## Board Update Requested

Proposed existing-row updates:

| ID | Proposed status | Evidence summary |
|---|---|---|
| W1.1 | DONE | Closure board exists and is committed; process rules are locked; ROADMAP links the board; all later amendments are represented by current board rows plus this v3 proposal's requested additions. |
| W5.3 | DONE | `test_cross_mode_determinism.rs`; WCOJ, binary fallback, recursive equality; fixed-seed, dynamic-injection, and Stage-5 rollback loops all `100/100`. |
| W5.4 | DONE | `test_widened_frontier_replay.rs`; DTS serving-v2 artifacts discovered; replay PASS; CUDA determinism category hook PASS. |
| W6.1 | DONE | `docs/wcoj-architecture-guide.md`; ROADMAP cross-link; RIR, promoter, dispatch, cost-model, recursive, Phase-1, and Phase-2 architecture coverage. |
| W6.2 | DONE | `docs/wcoj-user-guide.md`; ROADMAP cross-link; eligibility, fallback, config, tuning decision flow, and recipes. |

Proposed new board entries:

| ID | Source | Status | Required deliverable | Acceptance gate |
|---|---|---|---|---|
| W6.3 | Goal-039 G_W63_CHAIN | DONE | Chain-join WCOJ-shape promoter and runtime route for two-atom inner-chain rules. | Evidence `docs/evidence/2026-05-14-g39-w63-chain-prod/report.md`; M_W63.1-8 green with documented M_W63.4 route/correctness caveat; integrated by `bacca2ad`. |
| W6.4 | Goal-039 G_W64_K78 | DONE | K=7/K=8 clique provider/template/promoter/runtime coverage. | Evidence `docs/evidence/2026-05-18-g39-w64-k78-prod/README.md`; M_W64.1-7 green; integrated by `1d17f48b`. |
| W6.5 | Goal-039 G_W65_SORT | DONE | Sort-label propagation and DTS-DLM support-source fix that eliminates m37c-prime sort-map warnings. | Evidence `docs/evidence/2026-05-18-g39-w65-authorized-dts-fix.md`; 50-doc replay `SORT_WARNINGS=0`; xlog certs green; DTS fix `e0324fa`. |
| W6.6 | Goal-039 G_W66_CUDAGRAPH | DONE | Bounded CUDA Graph capture/replay for Stage-4 hot-loop CSM paths. | Evidence `docs/evidence/2026-05-18-g39-w66-cudagraph-integration-followup.md`; M_W66.1-6 green; integrated through `f1933833`. |

After the requested board update is explicitly approved and committed, the
expected board state is:

| State | Expected count |
|---|---|
| DONE | 30 |
| IN-PROGRESS | 0 |
| BLOCKED | 0 |
| OPEN | 1 (`W7.1`) |
| Total | 31 |

## W7.1 Handoff State

After the requested board update is explicitly approved and committed, W7.1
becomes the only remaining open board row. W7.1 remains gated by the separate
release rule: the tag fires only after the board action lands and the user
explicitly authorizes the `v0.6.5` tag.

No tag command is executed or implied by this proposal. The final tag handoff
document should be written after the approved board-update commit, so it can
record the actual post-board-update HEAD SHA and final board snapshot.

## Approval Request

Approve the Phase-2 board update:

1. Mark W1.1 DONE.
2. Mark W5.3, W5.4, W6.1, and W6.2 DONE.
3. Add W6.3, W6.4, W6.5, and W6.6 as DONE.

After approval, the only authorized follow-up for G_CLOSE2 is the board-update
commit on this branch. No merge to `main`, push, tag, or W7.1 release action is
implied.
