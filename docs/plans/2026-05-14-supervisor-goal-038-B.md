# Supervisor Goal 038-B - Authorization 5 Amended Plan

**Date amended:** 2026-05-17
**Authority:** Supervisor Authorization 5.
**Execution branch sequence:** `feat/w67b-step1-eligibility` through `feat/w67b-step11-close38b`.
**Current state:** W6.7 board flip is HOLD until closure proposal v2 is user-approved.
**Superseded proposal:** `ef3fbc7e` is preserved as evidence of the 9-sub-goal state and is not deleted.

## 0. Process Locks

Process locks 1-29 remain in force. Authorization 5 adds scope, not new locks.

The Authorization 5 hold is binding:

- Do not commit a W6.7 board entry until closure proposal v2 is user-approved.
- Do not merge to `main`.
- Do not push.
- Do not tag.
- Do not edit the W6.7 board entry before approval.

Out of scope for this goal and deferred to v0.7+:

- Stream-aligned multiplexing for K-clique.
- Helper-relation splitting beyond K-clique into arbitrary deep-join trees.
- Adaptive histogram resolution.

## 1. Amendment Rationale

Supervisor validation found two paper §5 alignment gaps after the original 9-sub-goal closure proposal:

1. `WcojRelationMetadata` runtime histograms were not built in the K-clique provider path. The K-clique HG kernel used `leader_count` from the compile-time plan rather than a runtime histogram refreshed during Merge phase. Non-recursive K-clique is functionally equivalent, but recursive K-clique inside semi-naive fixpoint requires per-iteration histogram refresh under paper §5 Algorithm 1 Phase 1.
2. `helper_split_specs` was always empty for K-clique. The promoter emitted `Vec::<HelperSplitSpec>::new()`, so K-clique rules with buried inner-variable skew could not expose that skew via helper splitting.

Goal-038-B therefore grows from 9 sub-goals to 11 sub-goals by inserting:

- Step 6, `G_HIST_KC`: runtime-histogram-driven block-slice for K-clique.
- Step 7, `G_HELP_KC`: helper-splitting K-clique invocation.

The original bench/integration/purge runs are superseded and must be rerun after steps 6 and 7:

- Superseded `G_BENCH38B`: `1c8415f1`.
- Superseded `G_INT38B`: `b2eebb10`.
- Superseded `G_PURGE38B`: `32dd43c7`.

Preserved step-1 through step-5 branches:

- `G_HG_ELIG`: `ef241c7f`.
- `G_HG_PLAN`: `9c77c7d4`.
- `G_RIR_VO`: `3ea3c657`.
- `G_DISPATCH_PLAN`: `5e69adc4`.
- `G_COST_GATE`: `77106ea0`.

## 2. Goal Hierarchy

| Step | G-node | Status | Production head |
|---:|---|---|---|
| 1 | `G_HG_ELIG` - hypergraph eligibility executor-aware | DONE, preserved | `ef241c7f` |
| 2 | `G_HG_PLAN` - cost-aware full variable-order planner for K-clique | DONE, preserved | `9c77c7d4` |
| 3 | `G_RIR_VO` - RIR variable-order surface for K5..K8 | DONE, preserved | `3ea3c657` |
| 4 | `G_DISPATCH_PLAN` - promoter, runtime, and CUDA kernel consume the plan | DONE, preserved | `5e69adc4` |
| 5 | `G_COST_GATE` - structured WCOJ/hash cost gate | DONE, preserved | `77106ea0` |
| 6 | `G_HIST_KC` - runtime-histogram-driven block-slice for K-clique | DONE | `4de1d0ba` |
| 7 | `G_HELP_KC` - helper-splitting K-clique invocation | DONE | `df09626c` |
| 8 | `G_BENCH38B` - rerun W5.2 and dILP-shape bench gates with new mechanisms active | DONE | `106a7c58` |
| 9 | `G_INT38B` - rerun integration gate with new mechanisms active | DONE | `feb758b7` |
| 10 | `G_PURGE38B` - rerun purge gate with new mechanisms active | DONE | `1e8a055f` |
| 11 | `G_CLOSE38B` - closure proposal v2 and approved board action | PROPOSAL COMMITTED; board action pending approval | `feat/w67b-step11-close38b` |

## 3. Per-Goal Decomposition

### 3.1 G_HG_ELIG - Step 1

Preserved from the pre-Authorization-5 plan. Acceptance remains M_HG_ELIG.1-4.

### 3.2 G_HG_PLAN - Step 2

Preserved from the pre-Authorization-5 plan. Acceptance remains M_HG_PLAN.1-6.

### 3.3 G_RIR_VO - Step 3

Preserved from the pre-Authorization-5 plan. Acceptance remains M_RIR_VO.1-4.

### 3.4 G_DISPATCH_PLAN - Step 4

Preserved from the pre-Authorization-5 plan. Acceptance remains M_DISP.1-6.

### 3.5 G_COST_GATE - Step 5

Preserved from the pre-Authorization-5 plan. Acceptance remains M_GATE.1-5.

### 3.6 G_HIST_KC - Step 6: runtime-histogram-driven block-slice for K-clique

**Goal.** Extend Phase-1 G1's `WcojRelationMetadata` histogram mechanism from triangle/4-cycle HG kernels to K=5..K=8 clique kernels, including recursive K-clique refresh during Merge phase.

**Strategies.**

1. Extend `crates/xlog-cuda/src/provider/wcoj.rs` with K-clique-edge metadata builders and four provider entries:
   - `wcoj_clique5_metadata_recorded_u32`
   - `wcoj_clique5_metadata_recorded_u64`
   - `wcoj_clique6_metadata_recorded_u32`
   - `wcoj_clique6_metadata_recorded_u64`
2. Reuse Phase-1 G1's `multiblock_scan_u32_inplace_on_stream` mechanism.
3. Extend `wcoj_clique_template_count_hg_grid_t<K, T>` and `wcoj_clique_template_materialize_hg_grid_t<K, T>` kernel template signatures to accept:
   - `const T* unique_keys`
   - `const uint32_t* fan_out`
   - `const uint32_t* prefix_sum`
   - `uint32_t total`
4. Use `prefix_sum` and `total` to drive block-slice at the leader edge instead of `leader_count` alone.
5. Update K-clique provider entries, including planned recorded entries, to build metadata before kernel launch.
6. Extend `crates/xlog-runtime/src/executor/recursive.rs` to refresh K-clique edge metadata during Merge phase.
7. Add a synthetic recursive K=5 fixture for M_HIST_KC.5: transitive-closure-style rule over a K=5 clique structure.

**Metrics.**

| Metric | Target |
|---|---|
| M_HIST_KC.1 provider surface | Four K-clique metadata provider entries present and callable. |
| M_HIST_KC.2 source audit | K-clique provider builds metadata before launch and uses Phase-1 scan mechanics. |
| M_HIST_KC.3 kernel ABI | K-clique count/materialize HG templates accept runtime histogram launch parameters. |
| M_HIST_KC.4 block-slice behavior | K5/K6 metadata paths are bit-exact against CPU/reference or prior direct paths. |
| M_HIST_KC.5 recursive refresh | Synthetic recursive K=5 fixture refreshes metadata during Merge phase and preserves row equality. |
| M_HIST_KC.6 W5.2/cost gate preservation | W5.2 routing and K-clique planner decisions remain preserved. |
| M_HIST_KC.7 paper citation | Source includes the paper §5 Algorithm 1 Phase 1 histogram-refresh citation. |
| M_HIST_KC.8 metadata overhead | Report raw metadata build count/time and show metadata ratio stays within the accepted guard. |

**Evidence.** `docs/evidence/2026-05-17-w67b-hist-kc/README.md`.

### 3.7 G_HELP_KC - Step 7: helper-splitting K-clique invocation

**Goal.** Replace the always-empty K-clique helper-split spec surface with planner-detected helper splitting for buried inner-variable skew.

**Strategies.**

1. Replace the always-empty `Vec::<HelperSplitSpec>::new()` K-clique promotion path.
2. Detect buried inner-variable skew using heat ratio threshold `>= 3x`, configurable through `XLOG_BURIED_SKEW_THRESHOLD`.
3. When detected, invoke Phase-1 G4 `helper_split_pass` with the K-clique helper spec.
4. Preserve no-helper behavior for uniform K-clique heat.
5. Compose with `G_HIST_KC`: post-split helper relations receive fresh metadata.

**Metrics.**

| Metric | Target |
|---|---|
| M_HELP_KC.1 planner detection | Buried-skew fixture emits helper spec; uniform fixture emits none. |
| M_HELP_KC.2 source audit | K-clique promoter no longer emits unconditional empty helper specs; compile invokes helper pass. |
| M_HELP_KC.3 helper allocation | Synthetic K5 buried-skew compile creates exactly one `__w37_helper_*` relation. |
| M_HELP_KC.4 ordering/composition | Helper rule emits before outer K-clique and outer rule consumes it. |
| M_HELP_KC.5 row equality | Helper-split K5 output equals direct K-clique output. |
| M_HELP_KC.6 regression preservation | Uniform K5 helper count remains zero; W5.2 routing remains preserved. |
| M_HELP_KC.7 paper citation | Promoter source cites paper §5 Figure 3 helper-relation splitting. |
| M_HELP_KC.8 histogram composition | Post-split helper path records metadata build evidence. |

**Evidence.** `docs/evidence/2026-05-17-w67b-help-kc/README.md`.

### 3.8 G_BENCH38B - Step 8

Renumbered from step 6 to step 8. The prior run `1c8415f1` is superseded.

**Goal.** Rerun W5.2 and DTS-DLM dILP-shape benchmark gates with `G_HIST_KC` and `G_HELP_KC` active.

**Metrics.** M_BENCH38B.1-5 remain the benchmark acceptance gates, with current amended W5.2 shape 12 workload cells x 2 paths = 24 gated path rows.

**Evidence.** `docs/evidence/2026-05-17-w67b-bench38b/README.md`.

### 3.9 G_INT38B - Step 9

Renumbered from step 7 to step 9. The prior run `b2eebb10` is superseded.

**Goal.** Rerun integration gates with `G_HIST_KC`, `G_HELP_KC`, and rerun `G_BENCH38B` active.

**Metrics.** M_INT38B.1-15:

1. W3.4 successor revalidation.
2. W4.1 cert regression.
3. W5.1 cert trio EXACT.
4. W5.2 amended per-path gate.
5. W2.5 default-flip.
6. W3.2 K=5/K=6 clique cert grid.
7. Workspace fmt.
8. Workspace build with `-D warnings`.
9. Workspace test.
10. CUDA cert suite.
11. Peak VRAM.
12. DLPack zero-copy preserved.
13. Witness-chain recoverable.
14. M37-A surface preserved.
15. Hypergraph planner is production K5/K6 path.

**Evidence.** `docs/evidence/2026-05-17-w67b-int38b/README.md`.

### 3.10 G_PURGE38B - Step 10

Renumbered from step 8 to step 10. The prior run `32dd43c7` is superseded.

**Goal.** Rerun purge gates with the new K-clique histogram and helper-split surface active.

**Metrics.** M_PURGE38B.1-11:

1. Bundle-task marker scan.
2. Legacy-process phrase scan.
3. Unused dependency scan.
4. Strict dead-code/import/variable build.
5. Author-trailer scan.
6. Future-version scan.
7. Paper-citation coverage.
8. Pre-existing dead-code followup presence.
9. K-clique hardcoded leader audit.
10. K5/K6 promoter unconditional-none audit.
11. Runtime plan-driven layout audit.

**Evidence.** `docs/evidence/2026-05-17-w67b-purge38b/README.md`.

### 3.11 G_CLOSE38B - Step 11

Renumbered from step 9 to step 11.

**Goal.** Emit closure proposal v2 superseding `ef3fbc7e`, then hold until explicit user approval for W6.7 board action.

**Metrics.**

| Metric | Target |
|---|---|
| M_CLOSE38B.1 closure proposal v2 | `docs/plans/2026-05-17-w67b-closure-proposal-v2.md` committed. |
| M_CLOSE38B.2 user approval | Explicit approval in thread to mark W6.7 DONE. |
| M_CLOSE38B.3 board update commit | Board entry commit only after M_CLOSE38B.2. |
| M_CLOSE38B.4 post-board state | W6.7 DONE; no merge, push, or tag implied. |

## 4. Dependency DAG

```text
feat/w3-bundle-integration @ c1689d70
  -> G_HG_ELIG @ ef241c7f
  -> G_HG_PLAN @ 9c77c7d4
  -> G_RIR_VO @ 3ea3c657
  -> G_DISPATCH_PLAN @ 5e69adc4
  -> G_COST_GATE @ 77106ea0
  -> G_HIST_KC @ 4de1d0ba
  -> G_HELP_KC @ df09626c
  -> G_BENCH38B rerun @ 106a7c58
  -> G_INT38B rerun @ feb758b7
  -> G_PURGE38B rerun @ 1e8a055f
  -> G_CLOSE38B closure proposal v2 on feat/w67b-step11-close38b
  -> HOLD for user approval before W6.7 board entry
```

## 5. Definition of Done

Goal-038-B is ready for W6.7 board approval only when all 11 sub-goals are closed or explicitly held at their approval gate:

- `G_HG_ELIG`: M_HG_ELIG.1-4 green.
- `G_HG_PLAN`: M_HG_PLAN.1-6 green.
- `G_RIR_VO`: M_RIR_VO.1-4 green.
- `G_DISPATCH_PLAN`: M_DISP.1-6 green.
- `G_COST_GATE`: M_GATE.1-5 green.
- `G_HIST_KC`: M_HIST_KC.1-8 green.
- `G_HELP_KC`: M_HELP_KC.1-8 green.
- `G_BENCH38B`: M_BENCH38B.1-5 green after steps 6 and 7.
- `G_INT38B`: M_INT38B.1-15 green after step 8.
- `G_PURGE38B`: M_PURGE38B.1-11 green after step 9.
- `G_CLOSE38B`: closure proposal v2 committed; W6.7 board edit pending user approval.

Board closure is not complete until the user explicitly approves the W6.7 board entry.

## 6. Execution Protocol

Execute sub-goals sequentially. Halt only on §7.2 stuck conditions or at the `G_CLOSE38B` user-approval gate. No intermediate authorization request is required between steps 6 through 10.

## 7. Stuck Conditions

Escalate if any acceptance metric fails in a way that requires changing the authorized scope, if W3.2 K5/K6 clique certs regress, if row equality fails, if paper §5 / §7.3 alignment fails, or if closure requires a board edit before user approval.
