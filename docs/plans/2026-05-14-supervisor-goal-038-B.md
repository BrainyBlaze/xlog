# Supervisor Goal 038-B — K5/K6 Hypergraph-Planner-as-Production-Planner Architecture

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog` (separate dispatch from goal-038 / goal-039).
**Predecessor:** `docs/plans/2026-05-14-supervisor-goal-038.md` (Phase 1 W3-axis closure; integration HEAD is goal-038-B base).
**Siblings (concurrent):** `docs/plans/2026-05-14-supervisor-goal-039.md` (Phase 2 DTS-DLM hot-loop completion).
**Dispatch precondition:** Phase 1 (goal-038) DONE — closure board W3 axis 9/9 DONE — integration HEAD durable.
**Methodology:** Basili–Caldiera–Rombach GQM + GQM+Strategies. References: https://en.wikipedia.org/wiki/GQM.
**Paper:** SRDatalog arXiv:[2604.20073](https://arxiv.org/abs/2604.20073).
**Closure board:** `docs/v065-closure-board.md`. Goal-038-B adds **new W6.7 composite entry** to the board covering all 5 architectural steps as a unit. W7.1 release tag fires only after W6.7 + Phase 2 W6.x + all prior items DONE.

---

## 0. Process locks (durable, inherited + B-specific)

Goal-037 process locks 1–10 + goal-038 locks 11–15 + goal-039 locks 11–22 inherited where applicable. Goal-038-B-specific extensions:

The inherited locks (canonical reference: goal-037 §0 + goal-038 §1 + goal-039 §0):

1. No simplification clauses
2. No back-compat shims
3. No `Ok(None)` decline for paper-aligned shapes
4. No bench-gate substitution
5. No dead-code preservation
6. No comment rot
7. No `Co-Authored-By` trailers
8. No `v0.6.6` references (goal-038-B closes within v0.6.5 — no deferred-to-future language)
9. Bench-spike-first
10. GQM+Strategies dispatch shape

Goal-038-B-specific extensions:

23. **Hypergraph planner is the production planner — no parallel routing path.** When 38-B closes, K5/K6 production dispatch MUST go through `xlog_logic::hypergraph` planner. No retained "canonical fallback" path. The current hardcoded canonical edge (0,1) leader at `crates/xlog-cuda/kernels/wcoj.cu:1076` is REPLACED by a plan-consuming kernel — not kept as a fallback per process lock 2.
24. **W2.1 cost model is extended, not replaced.** `WcojVariableOrderingModel` (`crates/xlog-logic/src/wcoj_var_ordering.rs:49`) extends to K=5..K=8. The triangle + 4-cycle leader paths remain bit-identical pre-B (triangle preserves W2.1's 3-leader permutation, 4-cycle preserves W2.1's 4-leader permutation). The B extension adds K=5/K=6 (and K=7/K=8 once Phase-2 G_W64 ships) full-variable-order plans on top.
25. **`hypergraph::eligibility::BINARY_FALLBACK_KEY_LIMIT` is executor-context-aware, not removed.** The constant stays; it gains a context query that asks "is this shape WCOJ-eligible by executor capability?" The 4-key limit remains binding for hash-fallback paths; WCOJ-eligible K5/K6/K7/K8 lift the limit via executor context.
26. **Paper §5 + §7.3 alignment is the architectural target, not perf optimization.** The goal is to make the production K5/K6 path *capable* of paper-faithful planning. Whether it actually beats hash on diagonal / pivot-heavy cells is determined by the planner's cost model; some cells correctly route to hash. M_W67-COST-GATE explicitly accepts that.
27. **W5.2 36-cell corpus + DTS-DLM dILP-shape synthetic fixture are the canonical benches.** No new fixture invented to make 38-B pass; the existing W5.2 evidence + Phase-2 DTS-DLM-analog fixture (G_W39_DTSDLM in goal-039) cover the validation surface.

28. **Stats infrastructure extension scope (R5 per supervisor amendment 2026-05-14, `docs/evidence/2026-05-14-g38-mint4-supervisor-amendment.md`):** Goal-038-B G_HG_PLAN consults existing W2.1 + W2.3 + W3.3 (Phase-1 G1 `WcojRelationMetadata`) stats infrastructure. The planner MAY extend existing stats surfaces with new dimensions required by HoneyComb-style pessimistic cardinality estimation: NDV (Number of Distinct Values) per column as extension to `StatsSnapshot` or `RelationCardinalities`; prefix-degree per join key as extension to `WcojRelationMetadata`; per-key heat (skew indicator) as extension to `StatsSnapshot` (or to `WcojRelationMetadata` per-candidate-root metadata per R1). The planner MUST NOT introduce a parallel stats subsystem competing with W2.1's stats pipeline (e.g., a sibling `HoneyCombStatsAccumulator` that bypasses `StatsSnapshot`). Lock 24 (W2.1 extends, doesn't replace) governs. Cert: planner imports only `xlog_stats::*` + `xlog_runtime::executor::wcoj_metadata::WcojRelationMetadata` types; no parallel stats type introduced. New dimensions land on existing types' field surface.

29. **Cost-gate carve-out from process lock 3 (R4 per supervisor amendment 2026-05-14):** Process lock 3 forbids `Ok(None)` decline for paper-aligned shapes. Goal-038-B G_COST_GATE routes K-clique shapes (paper-aligned per §3 + §5) to WCOJ-with-plan or HASH-by-cost-decision. The HASH-by-cost-decision path is NOT `Ok(None)`. It is an authorized cost-planned HASH route per paper §7.3 conditional-win-on-skew-at-root acknowledgment. Promoter emits structured `MultiwayPlan { route: PlannedHashRoute { reason: PlannedHashReason, planner_evidence: CostPredictionRecord }, ... }` enum variant. Semantic distinction: `Ok(None)` = promoter cannot handle this shape (forbidden); `PlannedHashRoute` = promoter recognized shape, ran cost model, CHOSE hash (permitted). RIR `MultiwayPlan` enum gains `PlannedHashRoute { reason, planner_evidence }` variant. Cert: source-audit shows zero new `Ok(None)` branches for K=5..K=8 promotion; all hash routing goes through `PlannedHashRoute` variant.

---

## 1. Strategic context (GQM+Strategies)

### 1.1 Business goal

> **BG38B.** Make the production K5/K6 dispatch path capable of paper-faithful planning (paper §5 Algorithm 1 Phase 1 variable-order + helper-splitting at inner variables when applicable) such that DTS-DLM dILP-induced K5/K6 rules (M37-F class) route via cost-aware hypergraph planner rather than unconditional canonical-leader promotion, AND such that W5.2's 36-cell corpus splits correctly between WCOJ-eligible cells (where the planner predicts WCOJ wins) and HASH-eligible cells (where the planner predicts hash wins) on the same machine where it runs. W7.1 release tag fires only when W6.7 composite entry is DONE.

### 1.2 Assumptions

| ID | Assumption | Source |
|---|---|---|
| A1 | The existing `xlog_logic::hypergraph` planner (`mod.rs:63`) is architecturally the right place for production K-clique planning, but is currently oracle-only and not wired to executor / cost model / kernels. | User analysis 2026-05-14 + code-pointers verified |
| A2 | Production K5/K6 path bypasses any planner today: `promote_multiway` emits `var_order: None` for K5/K6 (`promote.rs:1327`); runtime always layout-sorts all edges and calls canonical provider (`wcoj_dispatch.rs:1932`); CUDA kernel hardcodes canonical edge (0,1) as leader (`wcoj.cu:1076`). | User analysis + code-pointers verified |
| A3 | W2.1's `WcojVariableOrderingModel` (`wcoj_var_ordering.rs:49`) supports only triangle (3 leaders) + 4-cycle (4 leaders); K5/K6 has no permutation table. | W2.1 closure board entry + code-pointer verified |
| A4 | `hypergraph::eligibility::BINARY_FALLBACK_KEY_LIMIT = 4` (`eligibility.rs:23`) classifies K5 (5 join-key variables) as over the limit unless made executor-aware. | User analysis + code-pointer verified |
| A5 | Paper §5 Algorithm 1 line 1 leaves variable order free; paper §7.3 ablation 1.1×–35.8× is conditional on skew being visible at the root variable. xlog's hardcoded canonical leader cannot satisfy "skew visible at root variable" for K5/K6 shapes where skew is at non-leader variables. | Paper reading + project memory `reference_srdatalog_paper.md` |
| A6 | DTS-DLM M37-F (dILP rule discovery) can emit arbitrary-arity rules including K5/K6. M37-F is queued post-M37-A but within v0.6.5+ DTS-DLM consumer roadmap per `dts-dlm/docs/research/2026-05-08-pre-m37/04-FINAL-REPORT.md`. | DTS-DLM consumer roadmap |
| A7 | W5.2 36-cell corpus is the canonical regression surface for K5/K6 routing decisions. Same-machine baseline from `/home/dev/projects/xlog/.worktrees/w52-skewed-multiway-bench` is the reference. | W5.2 closure board entry + Codex's same-machine evidence `/tmp/g38-w52-branch-w52-bench.log` |

### 1.3 Strategy

> Implement the 5-step architecture in sequence — eligibility (G_HG_ELIG) → planner (G_HG_PLAN) → RIR variable-order surface (G_RIR_VO) → promoter/runtime/kernel wiring (G_DISPATCH_PLAN) → cost gate (G_COST_GATE) — followed by integration + bench + closure. Each step's deliverable is the next step's substrate; out-of-order execution would leak hardcoded-canonical assumptions into the new architecture. W2.1 + W3.2 surfaces extend; they do not get replaced.

### 1.4 KPI

> **KPI-38B.1:** Production K5/K6 dispatch decision is cost-aware. `promote_multiway` for K5/K6 shapes emits a non-`None` `var_order` plan when WCOJ is predicted to win, OR declines promotion when HASH is predicted to win. No unconditional WCOJ promotion path remains.
> **KPI-38B.2:** W5.2 36-cell corpus routes correctly: cells where WCOJ wins on same-machine baseline → WCOJ-routed; cells where HASH wins → HASH-routed. Decision boundaries are evidence-driven, not heuristic.
> **KPI-38B.3:** No regression on Phase-1 (W3.4 / W4.1 / W5.1) and Phase-2 (W6.1–W6.6 if Phase 2 has closed) closure metrics on integration HEAD.
> **KPI-38B.4:** Paper §5 + §7.3 alignment: at least one K5/K6 cell where W5.2 historical evidence showed WCOJ winning reproduces that result via the new planner-driven path (validates that cost-aware planning hasn't regressed real WCOJ wins).
> **KPI-38B.5:** DTS-DLM dILP-shape synthetic fixture (sub-extension of G_W39_DTSDLM in goal-039) routes K5/K6 rules correctly when emitted via dILP-class shape distributions.
> **KPI-38B.6:** Peak VRAM stays ≤ 38 GB on production-scale W5.2 corpus + DTS-DLM-analog fixture (inherited from goal-038 KPI-P1.7).

---

## 2. Goal hierarchy (38-B GQM tree)

```
BG38B — K5/K6 hypergraph-planner-as-production-planner (organizational)
 │
 ├── G_HG_ELIG    — Step 1: hypergraph eligibility executor-aware (lift BINARY_FALLBACK_KEY_LIMIT context-conditionally)
 │      │
 │      ▼
 ├── G_HG_PLAN    — Step 2: cost-aware full variable-order planner for K-clique (extends W2.1)
 │      │
 │      ▼
 ├── G_RIR_VO     — Step 3: RIR VariableOrder surface for K5..K8 (full plan + edge permutation + column swaps)
 │      │
 │      ▼
 ├── G_DISPATCH_PLAN — Step 4: promoter + runtime + CUDA kernel consume the plan (replace hardcoded canonical)
 │      │
 │      ▼
 ├── G_COST_GATE  — Step 5: cost gate — promoter declines OR runtime routes-to-hash based on planner verdict
 │      │
 │      ▼
 ├── G_HIST_KC    — Step 6 (NEW per Authorization 5, 2026-05-17): runtime-histogram-driven block-slice for K-clique
 │      │
 │      ▼
 ├── G_HELP_KC    — Step 7 (NEW per Authorization 5, 2026-05-17): helper-splitting K-clique invocation
 │      │
 │      ▼
 ├── G_BENCH38B   — Step 8 (renumbered): validate against W5.2 36-cell corpus + DTS-DLM dILP-shape fixture
 │      │
 │      ▼
 ├── G_INT38B     — Step 9 (renumbered): integration gate W3.4/W4.1/W5.1/W5.2/W2.5 regression-free post-B + new sub-goal mechanisms
 │      │
 │      ▼
 ├── G_PURGE38B   — Step 10 (renumbered): cross-cutting refactor + dead-code/comment purge
 │      │
 │      ▼
 └── G_CLOSE38B   — Step 11 (renumbered): closure proposal (supersedes `ef3fbc7e` 9-sub-goal proposal) + user approval + W6.7 board entry → DONE
```

11 G-nodes (was 9; Authorization 5 added G_HIST_KC + G_HELP_KC). Strictly sequential (each step's surface is the next step's substrate). Dependency DAG at §4.

---

## 3. Per-goal GQM decomposition

> Each G-node follows Basili template: **Analyze** *object* **for the purpose of** *purpose* **with respect to** *quality* **from the viewpoint of** *viewpoint* **in the context of** *context*.

---

### 3.1 G_HG_ELIG — Step 1: hypergraph eligibility executor-aware

**Goal.** Analyze `xlog_logic::hypergraph::eligibility` (`crates/xlog-logic/src/hypergraph/eligibility.rs:23`) for the purpose of making `BINARY_FALLBACK_KEY_LIMIT` executor-context-aware with respect to admitting K5 (5 join-key variables) and K6 (6 join-key variables) as WCOJ-eligible shapes from the viewpoint of paper §3.5 imperative (3) + project lock 25 in the context of the K5/K6 production routing gap.

**Questions.**
- **Q_HG_ELIG.1** Does the eligibility layer expose an executor-context query that distinguishes "hash-fallback path" (4-key limit binding) from "WCOJ-eligible path" (limit lifted for K=5..K=8)?
- **Q_HG_ELIG.2** Does the new context query preserve the 4-key limit for hash-fallback callers without surprise?
- **Q_HG_ELIG.3** Are K=7 and K=8 eligible alongside K=5 and K=6, anticipating Phase-2 G_W64 K=7/K=8 templates landing?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_HG_ELIG.1** | New `ExecutorContext` enum or trait introduced in `hypergraph::eligibility` with at least `HashFallback` and `WcojEligible` variants | 1/1 |
| **M_HG_ELIG.2** | `is_eligible(shape, ctx)` function: returns false for K≥5 under `HashFallback` ctx; returns true for K=5/K=6/K=7/K=8 under `WcojEligible` ctx; returns false for K≥9 under any context | 12/12 cert (3 contexts × 4 arities) |
| **M_HG_ELIG.3** | Pre-existing eligibility call sites: every existing caller is updated to pass an explicit context (no implicit default that changes behavior) | source-audit cert: 0 untyped call sites |
| **M_HG_ELIG.4** | `BINARY_FALLBACK_KEY_LIMIT = 4` constant retained, gated by context | grep shows constant retained; new gate references it explicitly |

**Strategies.**
- **S_HG_ELIG.1** Cut `feat/w67b-step1-eligibility` from Phase-1 integration HEAD.
- **S_HG_ELIG.2** Add `pub enum ExecutorContext { HashFallback, WcojEligible }` in `crates/xlog-logic/src/hypergraph/eligibility.rs`. Update `is_eligible()` signature to take context.
- **S_HG_ELIG.3** Update every existing call site to pass explicit context. Test-side calls use `HashFallback`; future WCOJ promoter calls use `WcojEligible`.
- **S_HG_ELIG.4** Add cert `crates/xlog-logic/tests/test_hg_eligibility_executor_context.rs` covering the 12-cell matrix.

**Acceptance.** All M_HG_ELIG.* green.

---

### 3.2 G_HG_PLAN — Step 2: cost-aware full variable-order planner for K-clique

**Goal.** Analyze `xlog_logic::hypergraph::var_order` (`crates/xlog-logic/src/hypergraph/var_order.rs:1`) for the purpose of adding cost-aware full variable-order planning for K=5..K=8 cliques with respect to cardinality + selectivity + heat + prefix-degree statistics from the viewpoint of paper §5 Algorithm 1 line 1 in the context of replacing the first-appearance trivial order.

**Questions.**
- **Q_HG_PLAN.1** Does the planner produce a full variable order `[v0, v1, ..., v_{k-1}]` for K-clique shapes that minimizes expected join cost given current statistics?
- **Q_HG_PLAN.2** Does the planner consult W2.1's existing stats surfaces (cardinality, selectivity, heat) without requiring new stats infrastructure?
- **Q_HG_PLAN.3** Does the planner produce a deterministic plan for a given (shape, stats) input — reproducible across runs with fixed seed?
- **Q_HG_PLAN.4** Does the planner produce a "no-plan" output (delegating to cost gate) when stats are missing or ambiguous?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_HG_PLAN.1** | Planner produces full variable order for K=5, K=6 shapes given complete stats | 2/2 (K=5, K=6) shape coverage |
| **M_HG_PLAN.2** | Planner is deterministic: same (shape, stats, seed) → same plan | 100/100 reproducibility cert |
| **M_HG_PLAN.3** (refined R5) | Planner consults existing W2.1 + W2.3 + W3.3 stats infrastructure. Stats interface = HoneyComb pessimistic cardinality estimator (Khamis 2024 refinement, [arXiv:2502.06715](https://arxiv.org/abs/2502.06715)): cardinality + selectivity + NDV + prefix-degree + per-key heat. New stats DIMENSIONS PERMITTED as extensions to existing types (`StatsSnapshot`, `RelationCardinalities`, `WcojRelationMetadata`); parallel stats subsystems FORBIDDEN per lock 28. | grep cert: planner imports only existing stats types; new fields land on existing types; no new accumulator |
| **M_HG_PLAN.4** | Planner produces "no-plan" output when stats incomplete; no panics, no defaults | cert: 4 incomplete-stats cases each return `None` |
| **M_HG_PLAN.5** | Planner extension to K=7/K=8 is template-call-only (no per-K hand-written algorithm) | Tier-1 source-audit cert: K=7/K=8 paths are template instantiations |
| **M_HG_PLAN.6** | Planner cost-prediction precision: on W5.2 36-cell same-machine baseline, planner's predicted-winner-path matches measured-winner-path on ≥ 90% of cells | ≥ 33/36 cells planner-prediction-correct |

**Strategies.**
- **S_HG_PLAN.1** Cut `feat/w67b-step2-planner` from G_HG_ELIG production HEAD.
- **S_HG_PLAN.2** Extend `crates/xlog-logic/src/hypergraph/var_order.rs` with `pub fn plan_kclique_var_order<S: StatsSource>(shape: &KCliqueShape, stats: &S) -> Option<FullVariableOrder>`.
- **S_HG_PLAN.3** Cost model: per-variable cost = `expected_intersection_cost(stats.cardinality(rel), stats.selectivity(rel, pair_keys), stats.heat(rel, leader_key))`. Choose leader by minimum cost; recurse for siblings.
- **S_HG_PLAN.4** Add cert `crates/xlog-logic/tests/test_hg_kclique_planner.rs` covering M_HG_PLAN.1–6.
- **S_HG_PLAN.5** For M_HG_PLAN.6 prediction precision: load same-machine W5.2 baseline data; for each of 36 cells, compute planner's predicted-winner; compare against measured-winner.
- **S_HG_PLAN.6** (R1 per supervisor amendment 2026-05-14) Extend Phase-1 G1's `WcojRelationMetadata` struct (`crates/xlog-cuda/src/provider/wcoj_metadata.rs`) to carry per-candidate-root metadata. Existing fields from goal-037 G1 S1.2 preserved: `unique_keys: CudaBuffer`, `fan_out: CudaBuffer`, `prefix_sum: CudaBuffer`, `total: u64`. Add new field: `per_candidate_root: BTreeMap<VertexId, RootMetadata>` where `RootMetadata { column_permutation: Vec<u8>, sorted_layout_signature: LayoutSignature, heat_distribution: HeatDist }`. Persistence: built on first dispatch per (relation, variable-order-context); reused across iterations; invalidated on relation merge. This is the data substrate the planner consults at promotion time to predict each candidate root's expected fanout cost.
- **S_HG_PLAN.7** Algorithm-level reference: HoneyComb pessimistic cardinality estimator per [arXiv:2502.06715](https://arxiv.org/abs/2502.06715). Algorithm port (not code copy): Cai 2019 base + Khamis 2024 refinement; partitions ALL variables via HyperCube-derived share allocation. xlog adapts the share-allocation idea to SIMT-friendly per-block-slice rather than thread-share (compatible with Phase-1 G1's HG block-slice mechanics).

**Acceptance.** All M_HG_PLAN.* green.

---

### 3.3 G_RIR_VO — Step 3: RIR VariableOrder surface for K5..K8

**Goal.** Analyze RIR's `VariableOrder` type for the purpose of extending it beyond triangle + 4-cycle (which currently use single `leader_idx`) to support K=5..K=8 cliques requiring full variable order + edge permutation + optional column swaps with respect to consumable downstream (promoter, runtime, kernel) from the viewpoint of process lock 24 (W2.1 extends, doesn't replace) in the context of plan transmission across the dispatch chain.

**Questions.**
- **Q_RIR_VO.1** Does the new `VariableOrder` variant for K=5..K=8 encode the full per-variable position + edge-to-slot permutation + column swap (when source column order differs from plan's expected order)?
- **Q_RIR_VO.2** Are triangle + 4-cycle `VariableOrder` variants preserved bit-identical (W2.1 closure regression-free)?
- **Q_RIR_VO.3** Does the new variant serialize / deserialize stably for plan caching (anticipating future plan memoization)?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_RIR_VO.1** (refined R2 per supervisor amendment 2026-05-14) | New RIR variant `KCliqueVariableOrder { k: u8, variable_positions: [u8; K_MAX], edge_permutation: [u8; EDGE_MAX], column_swaps: Vec<ColumnSwap>, sorted_layout_requirements: SortedLayoutSpec, helper_split_specs: Vec<HelperSplitSpec>, stream_group: StreamGroupId }`. Fields `sorted_layout_requirements`, `helper_split_specs`, `stream_group` are R2 additions: the plan carries enough information for goal-039 G5 stream-mux to consume directly (no separate stream-assignment pass) AND for Phase-1 G4 helper-split outputs to attach to the plan. Plan IR structure follows Free Join (SIGMOD 2023, [arXiv:2301.10841](https://arxiv.org/pdf/2301.10841)) bag-list + per-bag variable order + edge permutation + column swaps formalism — algorithm reference, NOT data structure import (xlog uses `CudaColumn` SoA, not COLT). | type present with 7 fields |
| **M_RIR_VO.2** | Triangle + 4-cycle variants byte-identical pre-B: W2.1 cert `cargo test -p xlog-runtime test_w21_variable_ordering` 11/11 PASS unchanged | 11/11 |
| **M_RIR_VO.3** | K=5/K=6 variants round-trip via planner-to-promoter-to-runtime path | round-trip cert |
| **M_RIR_VO.4** | RIR equality semantics: two K-clique variable orders with same plan compare equal; with different plans compare unequal | equality cert 4/4 |

**Strategies.**
- **S_RIR_VO.1** Cut `feat/w67b-step3-rir` from G_HG_PLAN production HEAD.
- **S_RIR_VO.2** Add new variant to RIR `VariableOrder` enum or introduce sibling `KCliqueVariableOrder` struct (depending on RIR design). Preserve triangle + 4-cycle paths verbatim.
- **S_RIR_VO.3** Add cert `crates/xlog-ir/tests/test_kclique_variable_order.rs` covering M_RIR_VO.1–4.

**Acceptance.** All M_RIR_VO.* green.

---

### 3.4 G_DISPATCH_PLAN — Step 4: promoter + runtime + CUDA kernel consume the plan

**Goal (refined R3 per supervisor amendment 2026-05-14).** Analyze the K5/K6 dispatch chain (promoter `promote.rs:1174` per Codex pointers + runtime `wcoj_dispatch.rs:1932` + CUDA kernel `wcoj.cu:1069` per Codex pointers) for the purpose of **REPLACING THREAD-PER-ROW CLIQUE KERNELS WITH ONE GENERIC HG BLOCK-SLICE KERNEL FAMILY OVER C[]** (the prefix-sum-flattened root space, extending Phase-1 G1's HG kernel from triangle/4-cycle to K=5..K=8) with respect to consuming G_RIR_VO's `KCliqueVariableOrder` plan from the viewpoint of process lock 23 in the context of removing the hardcoded canonical fallback. **The K-clique kernel BODY mirrors the triangle/4-cycle HG kernel body verbatim, just parameterized on K and plan-derived launch params.** No new per-K hand-written algorithm — W3.2's template mechanism is reused. Tier-1 source-audit cert: clique kernel body is one template call per K (consistent with goal-039 G_W64 K=7/K=8 template approach).

**Questions.**
- **Q_DISP.1** Does `promote_multiway` for K5/K6 shapes consult the planner and emit non-`None` `var_order` containing a full `KCliqueVariableOrder` plan?
- **Q_DISP.2** Does the runtime clique dispatch consume the plan's edge-permutation + column-swap to layout-sort only the necessary edges in the plan-specified order (instead of unconditional all-10-edge layout-sort)?
- **Q_DISP.3** Does the CUDA clique kernel accept the plan's leader-edge index + iteration order as launch parameters and use them in place of hardcoded canonical (0,1)?
- **Q_DISP.4** Does removing the hardcoded canonical path leave any orphaned code (process lock 5)?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_DISP.1** | `promote_multiway` for K=5/K=6 shapes emits non-`None` `var_order` from planner | source-audit cert: promoter K5/K6 branch calls planner; no `var_order: None` path remains for K5/K6 (decline path goes through cost gate instead, per G_COST_GATE) |
| **M_DISP.2** | Runtime clique dispatch consumes `KCliqueVariableOrder`: layout-sort applied only per plan's required edges/orders | bench cert: layout-sort kernel-launch count per K5 dispatch < 10 (was unconditionally 10) |
| **M_DISP.3** | CUDA clique kernel accepts leader-edge index + iteration order as launch parameters | kernel signature audit: `wcoj_clique_recorded_inner` accepts `KCliqueVariableOrder`-derived launch params |
| **M_DISP.4** | Hardcoded canonical edge (0,1) leader path removed: grep `wcoj.cu` for `canonical` / `(0, 1)` literals returns 0 in K-clique kernel body | grep cert |
| **M_DISP.5** | Row equality preserved on W5.2 36-cell corpus + W3.2 clique cert grid: every row of every K5/K6 output is bit-identical pre-B | row-equality cert 36 cells + 6 K5/K6 width-class certs from W3.2 |
| **M_DISP.6** | Triangle + 4-cycle dispatch bit-identical pre-B (W3.3/W3.7/W3.8 surface preserved) | source-audit: triangle + 4-cycle code paths untouched; corresponding certs 100% PASS |

**Strategies.**
- **S_DISP.1** Cut `feat/w67b-step4-dispatch` from G_RIR_VO production HEAD.
- **S_DISP.2** Update `crates/xlog-logic/src/promote.rs:1327` K5/K6 promotion to call `hypergraph::plan_kclique_var_order` and attach result to RIR `MultiWayJoin.var_order`.
- **S_DISP.3** Update `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1932` clique dispatch to consume the plan: derive per-edge layout-sort order from plan; reorder before kernel launch.
- **S_DISP.4** Update `crates/xlog-cuda/kernels/wcoj.cu:1076` K-clique kernel to accept leader-edge index + iteration order; use them in template body. Delete hardcoded canonical path.
- **S_DISP.5** Add 36 row-equality certs + 6 width-class certs.
- **S_DISP.6** Triangle + 4-cycle bit-identical preservation cert (W2.1 + W3.7 + W3.8 surface unchanged).

**Acceptance.** All M_DISP.* green.

---

### 3.5 G_COST_GATE — Step 5: cost gate (promoter declines OR runtime routes-to-hash based on planner verdict)

**Goal.** Analyze the K5/K6 promoter decision surface for the purpose of adding a cost-aware gate that declines WCOJ promotion (falls back to hash-join via standard promoter `Ok(None)`) when the planner predicts hash wins with respect to W5.2's documented HASH-winning cells (5-clique diagonal, pivot-heavy K5 trending toward parity) from the viewpoint of paper §7.3 ablation conditional-win-on-skew-at-root in the context of completing the architecture's routing layer.

**Questions.**
- **Q_GATE.1** Does the cost gate query the planner's cost-prediction output to decide WCOJ-vs-HASH routing per K5/K6 shape?
- **Q_GATE.2** (refined R4): Does the cost gate emit a structured `PlannedHashRoute` variant (NOT `Ok(None)`) for K-clique shapes where the planner predicts hash wins, per lock 29 carve-out from process lock 3? The cost-planned HASH route is a positive emission with audit traceability via `planner_evidence: CostPredictionRecord`, paper-aligned per §7.3 conditional-win-on-skew-at-root.
- **Q_GATE.3** Does the cost gate threshold use a documented decision function (no magic constants)?
- **Q_GATE.4** Does the cost gate enable WCOJ for W5.2 hub-skew clique cells where W5.2 evidence shows WCOJ wins?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_GATE.1** | Cost gate decision function documented at `crates/xlog-logic/src/promote.rs` adjacent to K5/K6 branch; threshold parameters named (not magic numbers) | source-audit cert: no unnamed numeric thresholds in K5/K6 promote branch |
| **M_GATE.2** (refined R4 per supervisor amendment 2026-05-14) | Cost gate emits structured `PlannedHashRoute { reason: PlannedHashReason, planner_evidence: CostPredictionRecord }` enum variant of `MultiwayPlan` — **NOT `Ok(None)`**. Process lock 3 remains binding for paper-aligned shapes; lock 29 carve-out permits structured cost-planned HASH routing as positive emission with audit traceability via `planner_evidence`. Source-audit cert: zero new `Ok(None)` branches for K=5..K=8 promotion path. Pattern-match exhaustiveness cert: `match` on `MultiwayPlan` enum covers `WcojWithPlan(plan)` and `PlannedHashRoute(reason, evidence)` variants. Comment in code references paper §7.3 conditional-win-on-skew-at-root + lock 29. | source-audit cert + grep + match-exhaustiveness check |
| **M_GATE.3** | W5.2 36-cell corpus routes correctly: cells where W5.2 same-machine baseline showed WCOJ winning → planner gates IN; cells where HASH won → planner gates OUT | 36/36 routing-decision cells correct |
| **M_GATE.4** | DTS-DLM dILP-shape synthetic fixture routes correctly (uses G_W39_DTSDLM if Phase 2 has shipped, otherwise local synthetic) | dILP-shape routing cert |
| **M_GATE.5** | Hub-skew clique cells from W3.2 (where W3.2 closure documented WCOJ wins) continue to route WCOJ post-B | W3.2 hub-skew preserved cert |

**Strategies.**
- **S_GATE.1** Cut `feat/w67b-step5-costgate` from G_DISPATCH_PLAN production HEAD.
- **S_GATE.2** (refined R4 per supervisor amendment 2026-05-14) Add cost-gate decision in `crates/xlog-logic/src/promote.rs` K5/K6 branch: invoke `hypergraph::plan_kclique_var_order(shape, stats)`; emit structured `MultiwayPlan` enum variant per outcome:
  - Planner returns `Some(plan)` AND `plan.predicted_winner == WcojPath` → emit `MultiwayPlan::WcojWithPlan(plan)`; runtime routes WCOJ.
  - Planner returns `Some(plan)` AND `plan.predicted_winner == HashPath` → emit `MultiwayPlan::PlannedHashRoute { reason: PlannedHashReason::PlannerPredictsHashWins, planner_evidence: plan.cost_prediction_record() }`; runtime routes hash-join via the same code path the standard hash promoter uses.
  - Planner returns `None` (incomplete stats) → emit `MultiwayPlan::PlannedHashRoute { reason: PlannedHashReason::IncompleteStatsSafeDefault, planner_evidence: CostPredictionRecord::empty() }`; runtime routes hash-join.
  
  **No `Ok(None)` branches for K=5..K=8 shapes.** Process lock 3 unaffected; lock 29 governs the structured-emission pattern. Comment block at the promoter branch cites paper §7.3 + lock 29 verbatim.
- **S_GATE.3** Document decision-function parameters: `WCOJ_COST_GATE_PARAMS` named constants in `crates/xlog-logic/src/wcoj_var_ordering.rs` with explanatory comments per process lock 6 (only WHY comments).
- **S_GATE.4** Add 36-cell routing cert + dILP-shape routing cert.

**Acceptance.** All M_GATE.* green.

---

### 3.6 G_HIST_KC — Runtime-histogram-driven block-slice for K-clique (added Authorization 5, 2026-05-17)

**Goal.** Analyze the K-clique HG kernel launch surface for the purpose of extending Phase-1 G1's `WcojRelationMetadata` runtime-histogram mechanism (originally for triangle + 4-cycle) to K=5..K=8 cliques with respect to paper §5 Algorithm 1 Phase 1 alignment (histogram maintained during Merge phase; consumed at kernel launch) from the viewpoint of full paper-§5 substrate completeness for K-clique in the context of W6.7 closure conditional on this and G_HELP_KC.

**Anchor.** Supervisor decision artifact Authorization 5 (`docs/evidence/2026-05-14-g38-mint4-supervisor-amendment.md`). Paper §5 Algorithm 1 Phase 1: *"Histograms maintained alongside data; computed incrementally during Merge; consumed at kernel launch-time to assign balanced thread-block work-unit slices."*

**Predecessor state (Authorization 5 finding).** K-clique HG kernels post-G_COST_GATE (step 5) accept `leader_count` as launch param populated from compile-time plan (HoneyComb cost model via `plan_kclique_var_order`). `WcojRelationMetadata` is NOT built in K-clique provider path. For non-recursive K-clique this is functionally equivalent; for recursive K-clique within semi-naïve fixpoint, paper §5 mandates per-iteration histogram refresh.

**Questions.**
- **Q_HIST_KC.1** Does `wcoj_build_metadata_recorded` provider entry extend to K-clique edge relations?
- **Q_HIST_KC.2** Do K-clique HG kernels accept `WcojRelationMetadata` launch params for the leader edge (per `KCliqueVariableOrder.leader_edge_idx`)?
- **Q_HIST_KC.3** Does runtime dispatch build metadata before kernel launch in non-recursive context?
- **Q_HIST_KC.4** Does histogram refresh during Merge phase work in recursive context (semi-naïve fixpoint with K-clique recursive body)?
- **Q_HIST_KC.5** Is determinism preserved under metadata refresh?
- **Q_HIST_KC.6** Does per-iteration histogram refresh cost stay bounded?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_HIST_KC.1** | `WcojRelationMetadata` builder extends to K-clique edge relations; provider entries `wcoj_clique{5,6}_metadata_recorded_{u32,u64}` | 4 entries present |
| **M_HIST_KC.2** | K-clique HG kernels accept runtime histogram launch params (signature includes `WcojRelationMetadata`-derived `{unique_keys, fan_out, prefix_sum, total}` per leader edge) | kernel signature audit cert PASS |
| **M_HIST_KC.3** | K-clique dispatch builds metadata before kernel launch in non-recursive context | source audit + provider trace cert PASS |
| **M_HIST_KC.4** | Determinism: bit-exact across 100 runs with `XLOG_DETERMINISTIC=1` + seed-pin on K5/K6 fixtures | 100/100 PASS |
| **M_HIST_KC.5** | Histogram refresh in recursive context — semi-naïve fixpoint with K-clique recursive body (synthetic dILP-induced K=5 transitive-closure-style fixture) produces bit-exact output across iterations | recursive cert PASS; paper P1 + P4 alignment preserved |
| **M_HIST_KC.6** | No regression on W5.2 36-cell routing prediction | 36/36 routing prediction preserved |
| **M_HIST_KC.7** | Per-iteration histogram refresh cost on `wcoj_w52_skewed_multiway` bench | ≤ 5% of iteration wall-time |
| **M_HIST_KC.8** | Paper §5 Algorithm 1 Phase 1 source-citation comment present in K-clique kernel + provider | `// Paper §5 Algorithm 1 Phase 1: Histograms maintained alongside data; refreshed during Merge per Authorization 5 (2026-05-17)` |

**Strategies.**
- **S_HIST_KC.1** Cut `feat/w67b-step6-hist-kc` from `feat/w67b-step5-costgate @ 77106ea0`. Worktree: `.worktrees/w67b-step6-hist-kc`.
- **S_HIST_KC.2** Extend `crates/xlog-cuda/src/provider/wcoj.rs` with K-clique-edge metadata builders. Reuse Phase-1 G1's `multiblock_scan_u32_inplace_on_stream` mechanism; new entry: `wcoj_clique{5,6}_metadata_recorded_{u32,u64}`.
- **S_HIST_KC.3** Extend `wcoj_clique_template_count_hg_grid_t<K, T>` and `wcoj_clique_template_materialize_hg_grid_t<K, T>` to accept additional launch params: `const T* unique_keys, const uint32_t* fan_out, const uint32_t* prefix_sum, uint32_t total`. Use prefix_sum + total to drive block-slice at the leader edge instead of `leader_count` alone.
- **S_HIST_KC.4** Update K-clique provider entries to build metadata before kernel launch via the new builders.
- **S_HIST_KC.5** Recursive integration: extend `crates/xlog-runtime/src/executor/recursive.rs` to refresh K-clique edge metadata during Merge phase (mirrors Phase-1 G1 mechanism for triangle/4-cycle).
- **S_HIST_KC.6** Synthetic recursive K=5 fixture for M_HIST_KC.5: transitive-closure-style rule over K=5 clique structure.
- **S_HIST_KC.7** Per-iteration cost measurement: extend `wcoj_phase_report` feature to expose histogram-refresh time per Merge call.

**Acceptance.** All M_HIST_KC.* green.

---

### 3.7 G_HELP_KC — Helper-splitting K-clique invocation (added Authorization 5, 2026-05-17)

**Goal.** Analyze the K-clique promoter helper-split emission for the purpose of replacing always-empty `Vec::<HelperSplitSpec>::new()` (`promote.rs:1466`) with planner-driven non-empty emission when buried inner-variable skew is detected with respect to paper §5 Figure 3 helper-relation-splitting alignment from the viewpoint of full paper-§5 substrate completeness for K-clique in the context of W6.7 closure conditional on this and G_HIST_KC.

**Anchor.** Supervisor decision artifact Authorization 5. Paper §5 Figure 3 (CallGraphEdge example): *"By factoring only these specific clauses into an independent HelpNT relation, the previously buried skew keys (sn, dsc, h) are exposed as top-level columns in the newly generated rule."*

**Predecessor state (Authorization 5 finding).** `HelperSplitSpec` type imported (`promote.rs:81`) per R2; K-clique promoter at `promote.rs:1466` emits empty `Vec`. Phase-1 G4's helper-split pass operates at AST→RIR boundary on full rules. K-clique rules with buried-skew at non-leader variables cannot expose that skew via helper-splitting in 38-B as currently shipped.

**Questions.**
- **Q_HELP_KC.1** Can the planner (from G_HG_PLAN) detect buried inner-variable skew via the new per-key heat infrastructure (per R7 + lock 28)?
- **Q_HELP_KC.2** Does K-clique promoter invoke helper_split_pass when planner emits a non-empty `HelperSplitSpec`?
- **Q_HELP_KC.3** Do helper relations correctly compose with K-clique plans (helper-split-then-K-clique-on-helper-rule)?
- **Q_HELP_KC.4** Is row equality preserved across split vs non-split paths?
- **Q_HELP_KC.5** Does G_HELP_KC compose with G_HIST_KC's runtime histogram (post-split relations get fresh histograms)?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_HELP_KC.1** | Planner buried-inner-variable-skew detection cert | 2/2 (positive: heat ratio ≥ 3× at non-leader variable; negative: uniform heat) |
| **M_HELP_KC.2** | K-clique promoter invokes helper_split_pass when planner emits non-empty `HelperSplitSpec` (source audit) | source-audit cert PASS; line 1466 emission is conditional, not always-empty |
| **M_HELP_KC.3** | `helper_split_specs` populated non-empty when buried skew present | synthetic K=5 buried-skew fixture cert PASS |
| **M_HELP_KC.4** | Helper relations emit additional plans that compose with K-clique plan | integration cert PASS |
| **M_HELP_KC.5** | Row equality on K-clique-with-helper-split vs direct K-clique on equivalent input | bit-exact cert on synthetic fixture |
| **M_HELP_KC.6** | No regression on K-clique-without-buried-skew (helper-split NOT invoked → empty vec → same as pre-G_HELP_KC behavior) | W5.2 36/36 routing preserved |
| **M_HELP_KC.7** | Paper §5 Figure 3 helper-relation-splitting source-citation comment present in promoter | `// Paper §5 Figure 3: Helper-relation splitting elevates buried inner-variable skew per Authorization 5 (2026-05-17)` |
| **M_HELP_KC.8** | Integration with G_HIST_KC — post-split helper relations get fresh `WcojRelationMetadata` builds | integration cert PASS |

**Strategies.**
- **S_HELP_KC.1** Cut `feat/w67b-step7-help-kc` from `feat/w67b-step6-hist-kc` HEAD (G_HIST_KC must close first).
- **S_HELP_KC.2** Extend `plan_kclique_var_order` in `crates/xlog-logic/src/hypergraph/var_order.rs` to detect buried-skew condition: compute heat ratio = `max(per_variable_heat) / heat[leader_variable]`; if ≥ `BURIED_SKEW_THRESHOLD` (default 3.0, configurable via `XLOG_BURIED_SKEW_THRESHOLD`), emit `HelperSplitSpec` describing which sub-clique to extract.
- **S_HELP_KC.3** Update K-clique promoter at `promote.rs:1466` to invoke `helper_split_pass(spec)` when planner emits non-empty spec; otherwise preserve empty-vec emission for the non-buried-skew case (avoids regression per M_HELP_KC.6).
- **S_HELP_KC.4** Helper relations emit via existing Phase-1 G4 `helper_split_pass` infrastructure; extend to accept K-clique-class specs.
- **S_HELP_KC.5** Synthetic K=5 buried-skew fixture: K=5 clique with one variable (v3) bound to a high-fanout relation while other variables are uniform.
- **S_HELP_KC.6** G_HIST_KC composition: post-split helper relations are NEW relations; runtime dispatch builds metadata for them via the new builders from G_HIST_KC.

**Acceptance.** All M_HELP_KC.* green.

---

### 3.8 G_BENCH38B — Validate against W5.2 36-cell corpus + DTS-DLM dILP-shape fixture

**Goal.** Analyze the 5-step architecture's end-to-end behavior for the purpose of validating cost-aware routing decisions on production fixtures with respect to W5.2 corpus + DTS-DLM dILP-shape from the viewpoint of KPI-38B.2 + KPI-38B.4 + KPI-38B.5 in the context of pre-integration acceptance.

**Questions.**
- **Q_BENCH.1** On W5.2 36-cell corpus: do cells route per same-machine baseline winner?
- **Q_BENCH.2** On DTS-DLM dILP-shape fixture: do K5/K6 rules route per cost-aware planner verdict?
- **Q_BENCH.3** On the hub-skew clique fixtures from W3.2: does WCOJ still win where W3.2 closure documented it?
- **Q_BENCH.4** Are per-path wall-times honest (no gate-substitution per process lock 4)?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_BENCH.1** | W5.2 36-cell corpus same-machine routing: 36/36 cells route to planner-predicted-winner path; per-path wall-time within ±10% of same-machine W5.2 branch baseline | 36/36 routing + 36/36 per-path within ±10% |
| **M_BENCH.2** | DTS-DLM dILP-shape fixture: at least 3 distinct dILP-class K5/K6 shape patterns route correctly per planner | 3/3 dILP-shape patterns route correctly |
| **M_BENCH.3** | Hub-skew clique fixtures (W3.2 closure-time): WCOJ still wins; row equality preserved | preserved per W3.2 closure baseline |
| **M_BENCH.4** | Honest per-path wall-time measurements: no `w52_literal_gate_reported_duration`-class shaping helpers in bench source | source-audit cert: `start.elapsed()` directly |
| **M_BENCH.5** | Peak VRAM ≤ 38 GB on benches | per-cell cudaMemGetInfo snapshot |

**Strategies.**
- **S_BENCH.1** Run W5.2 36-cell bench on G_COST_GATE production HEAD; capture per-path wall-time + routing decision per cell.
- **S_BENCH.2** Run DTS-DLM dILP-shape fixture (either via Phase-2 G_W39_DTSDLM if available, or local synthetic); capture routing decisions.
- **S_BENCH.3** Run W3.2 clique cert grid; assert hub-skew clique fixtures still WCOJ-route + row-equality preserved.
- **S_BENCH.4** Source-audit benches for absence of shaping helpers per process lock 4.

**Acceptance.** All M_BENCH.* green.

---

### 3.9 G_INT38B — Integration gate: W3.4 / W4.1 / W5.1 / W5.2 / W2.5 regression-free post-B (renumbered Authorization 5)

**Goal.** Analyze the integrated 38-B branch for the purpose of verifying composition-time correctness with respect to all prior closure metrics (W3.4, W4.1, W5.1, W5.2 amended per goal-038 M_INT.4, W2.5) regression-free from the viewpoint of Phase-1 + Phase-2 closure preservation in the context of pre-closure-proposal validation.

**Questions, Metrics, Strategies** parallel goal-038 §5.4 G_INT, applied to `feat/w67b-integration` HEAD with all 5 step branches merged in.

**Metrics.**

| Metric | Target |
|---|---|
| **M_INT38B.1** W3.4 successor revalidation | ratio ≥ 1.51× (per goal-038 amended M_INT.1) |
| **M_INT38B.2** W4.1 cert regression | 3/3 PASS (or 8/8 including non-W4.1 dispatch certs as in goal-038 result) |
| **M_INT38B.3** W5.1 cert trio EXACT | 3/3 EXACT counter + row-set match |
| **M_INT38B.4** W5.2 amended per-path | 36/36 GPU paths ≤ 1.10× same-machine baseline AND 36/36 hash paths ≤ 1.10× baseline AND 36/36 row equality (per goal-038 amended M_INT.4) |
| **M_INT38B.5** W2.5 default-flip | PASS |
| **M_INT38B.6** W3.2 K=5/K=6 clique cert grid | 6/6 PASS (preserved) |
| **M_INT38B.7** Workspace fmt | EXIT 0 |
| **M_INT38B.8** Workspace build `-D warnings` | EXIT 0 |
| **M_INT38B.9** Workspace test | 0 fail |
| **M_INT38B.10** CUDA cert suite | 1/1 (206 internal) |
| **M_INT38B.11** Peak VRAM | ≤ 38 GB |
| **M_INT38B.12** DLPack zero-copy preserved (KPI-7 inherited) | 0 host transfers on hot path |
| **M_INT38B.13** Witness-chain recoverable (KPI-6 inherited) | 100% recoverable on test fixtures |
| **M_INT38B.14** M37-A surface preserved (KPI-8 inherited from goal-039 lock 11) | M_M37A.1–10 reproducible if Phase 2 has shipped, else surface presence cert |
| **M_INT38B.15** Hypergraph planner is production K5/K6 path | grep cert: `promote.rs:1327` K5/K6 branch calls planner; no canonical fallback |

**Strategies.**
- **S_INT38B.1** Create branch `feat/w67b-integration` from `feat/w3-bundle-integration` (Phase-1 HEAD) OR `feat/w6-bundle-integration` (Phase-2 HEAD) — supervisor picks based on Phase-2 status at dispatch time.
- **S_INT38B.2** Merge step branches in order: G_HG_ELIG → G_HG_PLAN → G_RIR_VO → G_DISPATCH_PLAN → G_COST_GATE → G_BENCH38B.
- **S_INT38B.3** Run M_INT38B.1–15 sequentially; stop on first failure; fix on integration branch.

**Acceptance.** All M_INT38B.* green.

---

### 3.10 G_PURGE38B — Cross-cutting refactor + dead-code/comment purge (renumbered Authorization 5)

**Goal.** Analyze the 38-B-integrated codebase for the purpose of removing all dead code/comments/env vars introduced by 38-B from the viewpoint of process locks 5 + 6 + Karpathy 3 in the context of pre-closure cleanup.

Inherits goal-038 §5.5 G_PURGE methodology; applied on `feat/w67b-integration` HEAD post G_INT38B.

**Metrics.** M_PURGE38B.1–8 parallel goal-038 M_PURGE.1–8, scoped to 38-B touched files.

**Additional 38-B-specific metrics:**

| Metric | Target |
|---|---|
| **M_PURGE38B.9** | Hardcoded canonical edge (0,1) leader REMOVED from K-clique kernel | grep `wcoj.cu` for hardcoded canonical literal returns 0 hits in K-clique body |
| **M_PURGE38B.10** | Pre-B unconditional `var_order: None` path for K5/K6 REMOVED from promoter | grep `promote.rs` shows planner call, no unconditional None for K5/K6 |
| **M_PURGE38B.11** | Layout-sort-all-edges-unconditionally path REMOVED from runtime clique dispatch | grep `wcoj_dispatch.rs` shows plan-driven layout |

---

### 3.11 G_CLOSE38B — Closure proposal + user approval + W6.7 board entry → DONE (renumbered Authorization 5; supersedes proposal `ef3fbc7e` from 9-sub-goal state)

**Goal.** Analyze the integrated, purged 38-B bundle for the purpose of obtaining user approval to ADD W6.7 closure-board entry as DONE from the viewpoint of process rule 1 in the context of 38-B closure.

**Questions.**
- **Q_CLOSE38B.1** Does the closure proposal enumerate per-step status with evidence?
- **Q_CLOSE38B.2** Does the user explicitly approve W6.7 entry addition + DONE marking?

**Metrics.**

| Metric | Target |
|---|---|
| **M_CLOSE38B.1** Closure proposal at `docs/plans/2026-05-XX-w67b-closure-proposal.md` | committed |
| **M_CLOSE38B.2** User approval in thread | explicit "mark W6.7 DONE" |
| **M_CLOSE38B.3** Board update commit: ADD W6.7 entry as DONE | committed |
| **M_CLOSE38B.4** Closure-board state post-38B | W6.7 DONE; remaining OPEN items: W7.1 release tag + any incomplete Phase-2 items |

---

## 4. Dependency DAG (38-B execution order)

```
Phase-1 integration HEAD (feat/w3-bundle-integration, post-goal-038 DONE)
    │
    ▼
G_HG_ELIG   (Step 1: eligibility executor-aware)             [DONE @ ef241c7f]
    │
    ▼
G_HG_PLAN   (Step 2: cost-aware planner)                     [DONE @ 9c77c7d4]
    │
    ▼
G_RIR_VO    (Step 3: RIR VariableOrder surface)              [DONE @ 3ea3c657]
    │
    ▼
G_DISPATCH_PLAN (Step 4: promoter + runtime + kernel)        [DONE @ 5e69adc4]
    │
    ▼
G_COST_GATE (Step 5: cost gate)                              [DONE @ 77106ea0]
    │
    ▼
G_HIST_KC   (Step 6 NEW Authorization 5: runtime histogram)  [PENDING]
    │
    ▼
G_HELP_KC   (Step 7 NEW Authorization 5: helper splitting)   [PENDING]
    │
    ▼
G_BENCH38B  (Step 8 renumbered: validate routing + wall-time + new mechanisms)
    │
    ▼
feat/w67b-integration (re-cut post Authorization 5 sub-goals)
    │
    ▼
G_INT38B    (Step 9 renumbered: regression-free integration with new mechanisms)
    │
    ▼
G_PURGE38B  (Step 10 renumbered: cleanup)
    │
    ▼
G_CLOSE38B  (Step 11 renumbered: closure proposal v2 supersedes ef3fbc7e; user approval + W6.7 board entry → DONE)
```

**Strictly sequential.** Each step's deliverable is the next step's substrate. No parallelization opportunities until G_BENCH38B (which can technically run as the steps complete, but its gate requires all 5 steps in place).

**Concurrency with Phase 2 (goal-039):** 38-B is independent of Phase 2 from the dispatch perspective. Phase 2 (DTS-DLM hot-loop completion) and 38-B (K5/K6 planner architecture) touch different code surfaces:
- Phase 2: `xlog-runtime` (G_PRE instrumentation, reverted), `xlog-logic::optimizer` (G_W63 chain promoter), `xlog-cuda` (G_W64 K=7/K=8 templates, G_W66 CUDA Graphs), `xlog-runtime` schema (G_W65 sort labels)
- 38-B: `xlog-logic::hypergraph` (eligibility + planner), RIR `VariableOrder`, `xlog-logic::promote` (K5/K6 branch), `xlog-runtime::executor::wcoj_dispatch` (K-clique branch), `xlog-cuda::kernels::wcoj.cu` (K-clique kernel)

Overlap: `xlog-logic::promote` is touched by both Phase 2 G_W63 (chain promoter) and 38-B G_DISPATCH_PLAN (K5/K6 promoter). Merge conflict resolution at G_INT38B time.

**Sequencing options for supervisor:**
1. **38-B first, Phase 2 second.** 38-B closes K5/K6 architecture; Phase 2 builds on 38-B HEAD with K5/K6 planner already in place.
2. **Phase 2 first, 38-B second.** Phase 2 closes DTS-DLM hot-loop; 38-B builds on Phase-2 HEAD with chain promoter + K=7/K=8 templates already in place.
3. **Parallel.** Both dispatch concurrently from Phase-1 HEAD; merge at v0.6.5 final integration. Higher merge-conflict risk in `promote.rs`.

Supervisor picks at dispatch time. Recommended: option 2 (Phase 2 first, since DTS-DLM v3 is the primary KPI driver; 38-B architecturally enables future DTS-DLM M37-F dILP rules but isn't M37-A blocking).

---

## 5. Definition of Done (38-B)

38-B is DONE when ALL hold simultaneously:

1. **Per-sub-goal metrics green (11 sub-goals post-Authorization 5):**
   - G_HG_ELIG: M_HG_ELIG.1–4 green ✅ DONE @ ef241c7f
   - G_HG_PLAN: M_HG_PLAN.1–6 green ✅ DONE @ 9c77c7d4
   - G_RIR_VO: M_RIR_VO.1–4 green ✅ DONE @ 3ea3c657
   - G_DISPATCH_PLAN: M_DISP.1–6 green ✅ DONE @ 5e69adc4
   - G_COST_GATE: M_GATE.1–5 green ✅ DONE @ 77106ea0
   - **G_HIST_KC: M_HIST_KC.1–8 green** (NEW Authorization 5; pending)
   - **G_HELP_KC: M_HELP_KC.1–8 green** (NEW Authorization 5; pending)
   - G_BENCH38B: M_BENCH.1–5 green (re-validation with G_HIST_KC + G_HELP_KC mechanisms active; prior run @ 1c8415f1 superseded)
   - G_INT38B: M_INT38B.1–15 green (re-validation; prior run @ b2eebb10 superseded)
   - G_PURGE38B: M_PURGE38B.1–11 green (re-validation; prior run @ 32dd43c7 superseded)
   - G_CLOSE38B: M_CLOSE38B.1–4 green (proposal v2 supersedes `ef3fbc7e`)
2. **KPI satisfaction:** KPI-38B.1 through KPI-38B.6 all hold.
3. **Closure board:** W6.7 ADDED as DONE.
4. **User explicit DONE approval** in thread per process rule 1.
5. **W7.1 release tag** remains user-gated; 38-B DONE does NOT auto-tag.
6. **Phase 2 unblocked or compatible:** if Phase 2 hasn't started, 38-B's HEAD becomes Phase-2's base; if Phase 2 has started, 38-B integrates cleanly into Phase-2's branch.

---

## 6. Out-of-bounds (38-B constraints)

Goal-037 §13 items 1–10 + goal-038 §7 items 11–15 + goal-039 §6 items 16–22 in force. 38-B-specific additions:

23. **No new closure board entries beyond W6.7.** The 5 architectural steps roll into one composite W6.7 entry. Splitting would fragment review.
24. **No new env vars beyond:** `XLOG_WCOJ_CLIQUE_PLANNER_DEBUG` (optional planner-decision trace; default OFF; remove if unused per lock 5). NO `_LEGACY`, NO `_FALLBACK`, NO `_DISABLE_*`.
25. **No DTS-DLM repo mutations.** All DTS-DLM-side work is Phase 2 scope (goal-039).
26. **No xlog-induce changes.** Stage 5 ILP path stays frozen.
27. **No xlog-prob / xlog-neural changes.** M37-A surface (Group B per goal-039 lock 11) preserved verbatim — 38-B is xlog-logic + xlog-runtime + xlog-cuda only.
28. **No regression on W2.1 triangle + 4-cycle leader paths.** Process lock 24 binds; M_RIR_VO.2 + M_DISP.6 enforce.
29. **No retention of hardcoded canonical fallback after G_COST_GATE closes.** Process lock 23 binds; M_DISP.4 + M_PURGE38B.9 enforce.

---

## 7. Iteration protocol

### 7.1 Per-sub-goal loop

For each G-node ∈ {G_HG_ELIG, G_HG_PLAN, G_RIR_VO, G_DISPATCH_PLAN, G_COST_GATE, G_BENCH38B, G_INT38B, G_PURGE38B, G_CLOSE38B}:
1. Read G-node section.
2. Cut `feat/w67b-step<N>-<descriptor>` from prior step's production HEAD.
3. Implement; apply G_PURGE38B-equivalent cleanup to touched files in same commit chain.
4. Integration: merge into `feat/w67b-integration`; run G_INT38B gates per merge; fix regressions on integration branch.

### 7.2 Bundle stop conditions

COMPLETE when §5 DoD 1–6 hold.

STUCK (escalate) when:
- Any step's planner-prediction-precision (M_HG_PLAN.6) is < 70% (significantly below 90% target; indicates cost model insufficient).
- M_INT38B.6 W3.2 K=5/K=6 clique cert grid regresses (means dispatch-plan refactor broke W3.2 closure).
- M_DISP.5 row-equality cert fails on any of 42 cells (means kernel refactor broke correctness).
- KPI-38B.4 paper-§5+§7.3 alignment fails: zero K5/K6 cells reproduce W5.2 WCOJ wins (means planner over-routes to hash).

### 7.3 Self-evaluation checklist

```
[ ] Step N completes before step N+1 starts (strict ordering)
[ ] Pre-B triangle + 4-cycle paths bit-identical post-step (W2.1 surface preserved)
[ ] Pre-B W3.2 K5/K6 clique cert grid 6/6 PASS preserved (W3.2 surface preserved)
[ ] No new env vars beyond doc-allowed
[ ] No cfg(test) gates on production code
[ ] Hardcoded canonical fallback REMOVED at G_DISPATCH_PLAN completion
[ ] Cost gate decline path documented with paper-citation comment per M_GATE.2
[ ] Process lock 3 `Ok(None)` decline carve-out exercised only via cost-gate path (not via raw fallback)
[ ] W3.4 successor revalidation, W4.1 certs, W5.1 cert trio, W5.2 amended per-path, W2.5 default-flip all preserved
[ ] DLPack zero-copy preserved (KPI-7 inherited)
[ ] Witness-chain recoverable (KPI-6 inherited)
[ ] M37-A Group B surface preserved (KPI-8 inherited from goal-039)
[ ] Peak VRAM ≤ 38 GB (KPI-5 inherited)
[ ] No co-authored-by trailers
[ ] No v0.6.6 references
[ ] G_PURGE38B applied to touched files
[ ] W6.7 composite closure-board entry added (no sub-entries W6.8..W6.11)
```

---

## 8. Dispatch instructions

Dispatch into `codex-xlog` (or a sibling Codex session if Phase 2 is concurrent):
```
/goal @docs/plans/2026-05-14-supervisor-goal-038-B.md
```

Tab → Enter to confirm. Never `C-c` on idle codex. `codex resume <UUID>` on death.

**Dispatch precondition (CHECK BEFORE):**
- Phase 1 (goal-038) DONE: closure board W3 axis 9/9 DONE
- Phase-1 `feat/w3-bundle-integration` HEAD SHA recorded
- User has explicitly authorized 38-B dispatch (sequencing decision: 38-B-first, Phase-2-first, or parallel per §4)
- If parallel with Phase 2: coordinate merge-conflict ownership of `xlog-logic::promote` (38-B owns K5/K6 branch; Phase 2 owns chain-join promoter)

**First dispatch action:** G_HG_ELIG step 1. Implementer cuts `feat/w67b-step1-eligibility` from Phase-1 HEAD, adds `ExecutorContext` enum to `hypergraph::eligibility`, updates callers, adds 12-cell eligibility cert.

---

## 9. Code-pointer reference (verified 2026-05-14)

| Pointer | File:Line | What's there today |
|---|---|---|
| Hypergraph planner contract | `crates/xlog-logic/src/hypergraph/mod.rs:63` | Oracle/planning contract; opt-in; pure Rust; no CUDA, no executor, no cost model beyond AppearanceOrder |
| Hypergraph variable order | `crates/xlog-logic/src/hypergraph/var_order.rs:1` | First-appearance trivial order |
| Hypergraph eligibility | `crates/xlog-logic/src/hypergraph/eligibility.rs:23` | `BINARY_FALLBACK_KEY_LIMIT = 4` |
| Promoter multiway | `crates/xlog-logic/src/promote.rs:145` | Recognizes triangle, 4-cycle, K5/K6 shapes |
| Promoter K5/K6 var_order emit | `crates/xlog-logic/src/promote.rs:1327` | Always `var_order: None` for K5/K6 |
| W2.1 cost model | `crates/xlog-logic/src/wcoj_var_ordering.rs:49` | Triangle (3 leaders) + 4-cycle (4 leaders) only |
| Runtime clique dispatch | `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1932` | Always layout-sorts all edges; calls canonical provider |
| CUDA clique kernel | `crates/xlog-cuda/kernels/wcoj.cu:1076` | Hardcodes canonical edge (0,1) as leader; iterates edge(0,L) at each level |

---

## 10. References

- **GQM paradigm:** https://en.wikipedia.org/wiki/GQM; Basili–Caldiera–Rombach (1994); Basili et al. (2007).
- **Paper:** arXiv:[2604.20073](https://arxiv.org/abs/2604.20073) — Sun, Qi, Gilray, Kumar, Micinski. SRDatalog. Sections relevant to 38-B: §3.5 imperatives, §5 Algorithm 1 variable-order, §7.3 ablation 1.1×–35.8× conditional-win-on-skew-at-root.
- **Sibling goals:**
  - Phase 1: `docs/plans/2026-05-14-supervisor-goal-038.md`
  - Phase 2: `docs/plans/2026-05-14-supervisor-goal-039.md`
  - W3 paper-alignment bundle: `docs/plans/2026-05-13-supervisor-goal-037.md`
- **Closure board:** `docs/v065-closure-board.md`. 38-B adds W6.7 composite entry covering all 5 steps.
- **W5.2 evidence:** `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md` (line 346: HASH 12/12 on 5-clique diagonal + pivot-heavy K5).
- **W3.2 evidence:** `docs/evidence/2026-05-06-w32-general-arity-wcoj-template/` (K=5/K=6 template instantiation; 6 width-class certs preserved).
- **W2.1 evidence:** `docs/evidence/2026-05-04-w21-variable-ordering-cost-model/` (triangle + 4-cycle leader permutation tables).
- **DTS-DLM M37-F (eventual consumer):** `dts-dlm/docs/research/2026-05-08-pre-m37/04-FINAL-REPORT.md` lines 230+ (M37-F dILP-driven rule discovery; queued post-M37-A).
- **Algorithm-level prior art (composed architecture per supervisor amendment 2026-05-14):**
  - **HoneyComb** (PACMMOD 2025) — [arXiv:2502.06715](https://arxiv.org/abs/2502.06715) — pessimistic cardinality estimator (Cai 2019 + Khamis 2024 refinement); partitions ALL variables via HyperCube-derived share allocation; skew-aware. Adopted as G_HG_PLAN cost model algorithm reference (not code import).
  - **Free Join** (SIGMOD 2023) — [arXiv:2301.10841](https://arxiv.org/pdf/2301.10841) — unifies WCOJ + binary; bag-list + per-bag variable order + edge permutation + column swaps plan IR. Adopted as G_RIR_VO plan IR structure reference (not COLT data structure import — xlog uses `CudaColumn` SoA).
  - **EmptyHeaded** (SIGMOD 2017) — [PDF](https://stanford-ppl.github.io/website/papers/emptyheaded.pdf) / [GitHub](https://github.com/HazyResearch/EmptyHeaded) — GHD-based query plan representation; bag-decomposition framework.
  - **LevelHeaded** (ICDE 2018) — EmptyHeaded successor; SQL → query hypergraph → GHD.
- **External candidates explicitly REJECTED as wrong-domain per supervisor decision:** gHyPart, BiPart, mt-KaHyPar, HyperG, G-kway, iHyperG, Zoltan PHG, KaHyPar, hMETIS, PaToH (computation partitioning vs query planning). GDlog, VFLog (binary-join-based, not WCOJ planning).
- **Reference for kernel mechanics (not import):** Lai et al. GPU multi-way joins ([HKUST research portal](https://researchportal.hkust.edu.hk/en/publications/accelerating-multi-way-joins-on-the-gpu)); cuMatch GPU WCOJ subgraph matching ([colab.ws](https://colab.ws/articles/10.1145%2F3725398)).
- **Deferred to v0.7+:** PANDA submodular-width algorithm ([arXiv:2402.02001v3](https://arxiv.org/html/2402.02001v3)); FHD branch-and-bound ([VLDB 2024](https://www.vldb.org/pvldb/vol17/p4655-he.pdf)).
- **Supervisor decision artifact:** `docs/evidence/2026-05-14-g38-mint4-supervisor-amendment.md`.
- **Karpathy guidelines:** https://x.com/karpathy/status/2015883857489522876.

---

**End of goal-038-B document.** Implementer agent begins with G_HG_ELIG step 1 from Phase-1 integration HEAD. Supervisor awaits G_HG_ELIG completion + 12-cell eligibility cert before authorizing G_HG_PLAN step 2 dispatch.
