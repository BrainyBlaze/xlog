# W67B / Goal-038-B Closure Proposal v2

**Branch:** `feat/w67b-step11-close38b`
**Base:** `feat/w67b-step10-purge38b @ 1e8a055f951549d5acd985bcae252d0822501c18`
**Closure candidate HEAD:** `1e8a055f951549d5acd985bcae252d0822501c18`
**Date:** 2026-05-17
**Board action requested:** after explicit user approval only, add the composite W6.7 entry as DONE.

## Supersession

This proposal supersedes closure proposal `ef3fbc7e`. That proposal is preserved as evidence of the original 9-sub-goal state and is not deleted.

The amended governing plan is recorded in `docs/plans/2026-05-14-supervisor-goal-038-B.md`. It captures the Authorization 5 11-sub-goal hierarchy, inserted `G_HIST_KC` and `G_HELP_KC` steps, superseded prior bench/integration/purge runs, preserved step-1 through step-5 heads, and the W6.7 board HOLD.

Authorization 5 added two paper §5 alignment sub-goals:

- `G_HIST_KC` step 6: runtime-histogram-driven block-slice for K-clique.
- `G_HELP_KC` step 7: helper-splitting K-clique invocation.

The prior bench/integration/purge runs are superseded and rerun on the amended 11-sub-goal chain:

- Superseded `G_BENCH38B`: `1c8415f1`.
- Superseded `G_INT38B`: `b2eebb10`.
- Superseded `G_PURGE38B`: `32dd43c7`.

No W6.7 board edit, merge to `main`, push, or tag is included in this proposal commit.

## Scope

Goal-038-B makes the production K5/K6 WCOJ path planner-driven and paper §5-aligned. The implementation wires executor-aware hypergraph eligibility, a cost-aware K-clique planner, RIR variable-order metadata, plan-consuming runtime/kernel dispatch, structured cost-gate routing, runtime K-clique histogram metadata refresh, K-clique helper splitting, and rerun bench/integration/purge evidence.

## Commit Chain

| Step | Sub-goal | Commit | Status |
|---:|---|---|---|
| 1 | G_HG_ELIG | `ef241c7f` | DONE, preserved |
| 2 | G_HG_PLAN | `9c77c7d4` | DONE, preserved |
| 3 | G_RIR_VO | `3ea3c657` | DONE, preserved |
| 4 | G_DISPATCH_PLAN | `5e69adc4` | DONE, preserved |
| 5 | G_COST_GATE | `77106ea0` | DONE, preserved |
| 6 | G_HIST_KC | `4de1d0ba` | DONE |
| 7 | G_HELP_KC | `df09626c` | DONE |
| 8 | G_BENCH38B | `106a7c58` | DONE, rerun after steps 6-7 |
| 9 | G_INT38B | `feb758b7` | DONE, rerun after steps 6-8 |
| 10 | G_PURGE38B | `1e8a055f` | DONE, rerun after steps 6-9 |
| 11 | G_CLOSE38B | this proposal | PENDING user approval for board action |

## Metric Summary

| Sub-goal | Result |
|---|---|
| M_HG_ELIG.1-4 | PASS. `ExecutorContext` present; 12-cell eligibility cert PASS; zero untyped eligibility call sites; `BINARY_FALLBACK_KEY_LIMIT = 4` retained and context-gated. |
| M_HG_PLAN.1-6 | PASS. K5/K6 full orders `2/2`; deterministic cert `100/100`; planner uses existing stats surfaces only; incomplete-stats `Option::None` cert `4/4`; K7/K8 template-call-only audit green; W5.2 route prediction `36/36`. |
| M_RIR_VO.1-4 | PASS. K-clique variable-order plan surface committed; serialization/roundtrip certs green; rewrite preserves plan fields; W2.1 cert surface green. |
| M_DISP.1-6 | PASS. Promoter emits planner-backed K5/K6 plans; runtime consumes edge and iteration orders; CUDA kernel accepts plan launch params; no hardcoded K-clique leader in kernel body; row-equality cert grid green; source audits green. |
| M_GATE.1-5 | PASS. Structured `PlannedHashRoute` cost path committed; paper §7.3 comment present; no raw post-recognition decline path; W5.2 routing cert `36/36`; hash/WCOJ split encoded in RIR plan. |
| M_HIST_KC.1-8 | PASS. Evidence at `docs/evidence/2026-05-17-w67b-hist-kc/README.md`: 4 provider entries present; K5/K6 metadata source audit `8/8`; K5/K6 deterministic metadata paths `100/100`; recursive K5 `dispatch_count=5`, `refresh_count=3`, `metadata_build_count=4`, `metadata_ratio=0.000818`; W5.2 routing remains `36/36`; selected W5.2 ratios include `5clique_N10=0.009450`, `pivot5_N10=0.008668`. |
| M_HELP_KC.1-8 | PASS. Evidence at `docs/evidence/2026-05-17-w67b-help-kc/README.md`: planner buried-skew cert `2/2`; helper pass wired into compile; synthetic K5 buried skew creates exactly one `__w37_helper_*`; helper/direct K5 row equality `rows=1`; uniform K5 helper count `0`; W5.2 routing remains `36/36`; post-split helper metadata path recorded `helper_relations=1`, `dispatch_count=1`, `metadata_build_count=1`. |
| M_BENCH38B.1-5 | PASS. Evidence at `docs/evidence/2026-05-17-w67b-bench38b/README.md`: `24/24` accepted path medians within `1.10x`, `12/12` GPU-WCOJ, `12/12` hash-chain, `126` current VRAM snapshots, max VRAM delta `234,881,024` bytes, max histogram metadata ratio `0.033099`, workspace build/test green. |
| M_INT38B.1-15 | PASS. Evidence at `docs/evidence/2026-05-17-w67b-int38b/README.md`: W3.4 successor `6.609x`; W4.1 `8/8`; W5.1 trio `3/3`; W5.2 amended per-path green; W2.5 default flip `5/5`; K5/K6 clique cert `8/8`; workspace fmt/build/test green; CUDA release cert `206/206`; peak VRAM `201,326,592` bytes; DLPack/no-DtoH `7/7`; provenance `6/6`; M37-A surface-presence cert green; production K5/K6 planner source audit green. |
| M_PURGE38B.1-11 | PASS. Evidence at `docs/evidence/2026-05-17-w67b-purge38b/README.md`: touched-file scope `52` paths; marker/churn/future scans `0` hits; `cargo +nightly udeps --workspace --all-targets` green; strict dead-code/import/variable release build green; paper-citation coverage green; K-clique hardcoded-leader audit `0` hits; promoter/runtime source audits green. |
| M_CLOSE38B.1 | PASS once this proposal and the Authorization-5 amended governing plan are committed. |
| M_CLOSE38B.2-4 | Pending explicit user approval and the follow-up W6.7 board-entry commit. |

## KPI Status

| KPI | Status |
|---|---|
| KPI-38B.1 cost-aware production K5/K6 decision | SATISFIED. K5/K6 promotion runs the hypergraph planner and emits either `WcojWithPlan` with `VariableOrder::kclique` or structured `PlannedHashRoute`. |
| KPI-38B.2 W5.2 routing correctness | SATISFIED. Planner-predicted winner matches W5.2 same-machine evidence `36/36`, and the rerun benchmark has `24/24` accepted path medians. |
| KPI-38B.3 no Phase-1 regression | SATISFIED by G_INT38B: W3.4 successor, W4.1, W5.1, W5.2 amended, W2.5, and W3.2 K5/K6 certs all green. |
| KPI-38B.4 paper §5 + §7.3 alignment | SATISFIED. K-clique planner chooses variable/edge orders; runtime histograms drive K-clique block-slice; recursive K5 refreshes histogram metadata during Merge; helper splitting exposes buried inner-variable skew; cost gate preserves hash routing when measured evidence says hash wins. |
| KPI-38B.5 DTS-DLM dILP-shape readiness | SATISFIED for current 38-B scope. Phase-2 production replay is not claimed here; the planner/runtime foundation is live for K5/K6 shapes emitted by downstream consumers. |
| KPI-38B.6 peak VRAM | SATISFIED. G_INT38B peak `201,326,592` bytes and W5.2 bench peak `234,881,024` bytes are below `40,802,189,312` bytes. |

## Deviations

- G_HG_ELIG used a 2-context x 6-K-value matrix instead of the originally described 3-context x 4-arity matrix; cell count is still `12`, includes K9/K10 negative coverage, and was accepted by supervisor in the preserved step-1 chain.
- G_INT38B.4 uses the supervisor-amended W5.2 protocol shape: 12 workload cells x 2 paths = `24` gated path rows, with median-of-3 samples.
- M37-A remains a surface-presence cert because the Phase-2 production replay is separate scope.

## Proposed W6.7 Board Entry

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W6.7 | Goal-038-B | DONE | - | K5/K6 hypergraph planner as production planner: executor-aware eligibility, cost-aware K-clique planner, RIR variable-order surface, plan-consuming runtime/kernel dispatch, structured cost gate, K-clique runtime histogram refresh, K-clique helper-splitting invocation, bench/integration/purge evidence. | Goal-038-B amended plan `docs/plans/2026-05-14-supervisor-goal-038-B.md`; commit chain `ef241c7f` -> `1e8a055f`; M_HG_ELIG.1-4, M_HG_PLAN.1-6, M_RIR_VO.1-4, M_DISP.1-6, M_GATE.1-5, M_HIST_KC.1-8, M_HELP_KC.1-8, M_BENCH38B.1-5, M_INT38B.1-15, M_PURGE38B.1-11 all green; user approval in thread. |

## Approval Request

Approve adding the W6.7 composite closure-board entry as DONE.

After that approval, the only authorized next action is the board-entry commit on this branch. No merge to `main`, push, or tag is implied.
