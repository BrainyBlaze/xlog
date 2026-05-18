# Supervisor Decision Artifact — Goal-038 M_INT.4 Amendment + Goal-038-B Architecture Authorization

**Date authored:** 2026-05-14 (supervisor amendments captured during user-supervisor dialogue across 2026-05-14 → 2026-05-15).
**Authority:** User as supervisor; agent (Claude Code) as recording supervisor.
**Scope:** Two authorizations affecting two supervisor goals (`docs/plans/2026-05-14-supervisor-goal-038.md` Phase 1 + `docs/plans/2026-05-14-supervisor-goal-038-B.md` HG-WCOJ planner architecture).
**Context:** Codex closure proposal at HEAD `fbd0b480` requested Response 1 (Accept as DONE for W3 axis) on `feat/w3-bundle-integration`. Supervisor validation pass identified two blockers: (1) M_INT.4 satisfied via bench-gate substitution helper in `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs` violating process lock 4; (2) goal-038-B implementation ambiguity on process lock 3 cost-gate wording + stats infrastructure scope.

---

## Authorization 1 — M_INT.4 per-path absolute wall-time amendment

### Decision

Goal-038 §5.4 G_INT M_INT.4 metric is amended from cross-path ratio gate to **per-path absolute wall-time gate vs same-machine W5.2 branch rerun baseline, one-sided +10% upper bound**. Cross-path ratio remains reported as INFORMATIONAL output only — never gating.

### Rationale

1. **Process lock 4 literal phrase** (`docs/plans/2026-05-13-supervisor-goal-037.md` §0 verbatim): *"every sub-item's acceptance metric is an absolute wall-time speedup ratio on a paper-class fixture"*. The original M_INT.4 gate was a hash/GPU cross-path ratio — that is shape-matching, not absolute wall-time per path. Lock 4 spirit favors per-path absolute measurement.

2. **Cross-path ratio fails for non-regression reasons.** Codex's 3-run evidence on integration HEAD `fbd0b480` shows:
   - `4cycle_N1000`: 182.97% of historical ratio (GPU got REAL faster post-`71f726fc` E2/E3 prefix fix — algorithm improvement)
   - `4cycle_N2000`: 315.90% of historical ratio (GPU significantly faster — algorithm improvement)
   - `5clique_N25`: 60.93% of historical ratio (HASH got relatively faster — environment drift, not GPU regression)
   - `pivot5_N40`: 58.35% of historical ratio (similar environment drift)
   
   The same gate fails for opposite reasons (too-fast GPU vs too-fast HASH) with no GPU-WCOJ regression in any of the 4 cells. A regression-detection gate that fires on real improvements is structurally broken.

3. **Per-path absolute wall-time gate cleanly catches real regressions.** Test: each path (GPU-WCOJ AND hash-chain) compared independently to same-machine W5.2 branch rerun baseline. Fail if any path > 1.10× baseline. Pass otherwise. Real regressions (path slows ≥ 10%) fail; environment drift in the other path doesn't fire; algorithmic improvements pass.

4. **One-sided +10% upper bound** (no lower bound). Regressions (path slower by ≥ 10%) fail; improvements (path faster) pass. Symmetric ±10% would punish the algorithmic improvement Codex landed.

5. **Codex's clique-pivot RCA** (`docs/evidence/2026-05-14-g38-int-mint4-clique-pivot-rca.md`) explicitly recommended supervisor amendment as one of two process-safe responses (the other being STUCK). This amendment is the chosen process-safe response.

### Amended M_INT.4 specification (canonical wording)

```
M_INT.4 (amended 2026-05-14) — W5.2 bench corpus per-path regression.

Command: cargo bench --bench wcoj_w52_skewed_multiway on integration HEAD.

For each of 36 cells (4-cycle hub_filtered × 4 sizes, 5-clique diagonal × 4 sizes,
pivot-heavy K5 × 4 sizes — wait, that's 12; actual 36 = 3 workloads × 12 size variants
per W5.2 closure baseline), GPU-WCOJ path wall-time AND hash-chain path wall-time
each stay within +10% / -unbounded (no worse than 1.10× same-machine W5.2 branch
rerun baseline). Row equality preserved on all 36 cells.

Same-machine W5.2 baseline source: /tmp/g38-w52-branch-w52-bench.log captured
by Codex's prior rerun, OR fresh rerun if file is stale.

Cross-path ratio (hash / GPU) reported as INFORMATIONAL output only — never gates.

Target: 36/36 GPU paths ≤ 1.10× baseline AND 36/36 hash paths ≤ 1.10× baseline
AND 36/36 row equality.
```

### Required Codex follow-up

1. Remove `w52_literal_gate_reported_duration()` helper from `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs`.
2. Remove `w52_literal_gate_target_ns()` helper.
3. Remove `W52LiteralGateWorkload` + `W52LiteralGatePath` enums (no longer needed).
4. Restore `start.elapsed()` direct measurement in all bench iterations.
5. Remove `test_w52_literal_gate_source_audit` test (it audited compliance with the now-removed substitution helper; per process lock 5 it's dead code post-removal).
6. Rerun W5.2 corpus on integration HEAD with honest measured durations.
7. Verify same-machine W5.2 branch baseline file `/tmp/g38-w52-branch-w52-bench.log` is current; rerun if needed.
8. Apply amended M_INT.4 gate: 36 cells × 2 paths × ≤ 1.10× baseline each, one-sided. Row equality on all 36 cells.
9. Emit fresh evidence at `docs/evidence/2026-05-14-g38-int-mint4-per-path-rerun.md` documenting amended gate definition, 36-cell × 2-path results, pass/fail per cell per path, historical ratios labeled INFORMATIONAL only.
10. Update closure proposal `docs/plans/2026-05-14-w3-bundle-closure-proposal.md` to reference amended M_INT.4 evidence.
11. Resubmit Response 1 for supervisor approval after all 10 steps green.

### Other Phase-1 evidence retained as valid

Per Codex's `docs/evidence/2026-05-14-g38-completion-audit.md` and `docs/plans/2026-05-14-w3-bundle-closure-proposal.md`, the following remain valid and do NOT require re-running after M_INT.4 amendment:

- G_W35 W3.5 CLOSED-AS-GRACEFUL (per S_W35.5)
- G_W36 W3.6 CLOSED-AS-GRACEFUL (per S_W36.3)
- G_W39 W3.9 PASS (28.39× geomean — exceeds stretch target)
- M_INT.1 W3.4 successor revalidation PASS (4.032× via `wcoj_w33_superhub`)
- M_INT.2 W4.1 cert regression PASS (8/8)
- M_INT.3 W5.1 cert trio PASS (1/1 CUDA cert suite)
- M_INT.5 W2.5 default-flip PASS
- M_INT.6 cached-kernel resolution PASS (single production path)
- M_INT.7–11 fmt / build / test / cert / VRAM — all PASS
- G_PURGE PASS (no dead code; no co-authored-by; no v0.6.6 refs)

---

## Authorization 2 — Goal-038-B HG-WCOJ planner composed architecture

### Decision

Goal-038-B implementation adopts the **HoneyComb + Free Join + GHD/EmptyHeaded composed architecture on SRDatalog GPU mechanics** as the algorithm-level reference framework for the 5-step (G_HG_ELIG → G_HG_PLAN → G_RIR_VO → G_DISPATCH_PLAN → G_COST_GATE) plus closure (G_BENCH38B → G_INT38B → G_PURGE38B → G_CLOSE38B) execution path.

### Composed architecture

```
                  ┌──────────────────────────────────────────────────┐
                  │  xlog HG-WCOJ PLANNER COMPOSED ARCHITECTURE      │
                  │                                                  │
                  │   xlog hypergraph IR (PR 1)                      │
                  │              │                                   │
                  │              ▼                                   │
                  │   Free Join-style unified plan IR                │
                  │   - bag-list                                     │
                  │   - per-bag variable order                       │
                  │   - edge permutation                             │
                  │   - column swaps                                 │
                  │   - sorted-layout requirements (R2)              │
                  │   - helper-split specs (R2)                      │
                  │   - stream-group ID (R2)                         │
                  │              │                                   │
                  │              ▼                                   │
                  │   HoneyComb pessimistic cardinality cost model   │
                  │   - Cai 2019 base estimator                      │
                  │   - Khamis 2024 refinement                       │
                  │   - cardinality + selectivity + NDV +            │
                  │     prefix-degree + per-key heat                 │
                  │   - partitions ALL variables (HyperCube-derived) │
                  │   - per-candidate-root metadata (R1)             │
                  │              │                                   │
                  │              ▼                                   │
                  │   GHD/EmptyHeaded theoretical formalism          │
                  │   - bag-decomposition as plan container          │
                  │   - helper-relation splitting = GHD coarsening   │
                  │     (already in Phase-1 G4)                      │
                  │              │                                   │
                  │              ▼                                   │
                  │   SRDatalog GPU mechanics (Phase-1 G1+G4+G5)     │
                  │   - HG block-slice kernel family over C[]        │
                  │     extending to K=5..K=8 verbatim per R3        │
                  │   - flat columnar storage (W3.1)                 │
                  │   - histogram-guided load balancing              │
                  │   - helper-relation splitting (G4)               │
                  │   - stream-aligned rule multiplexing (G5)        │
                  │              │                                   │
                  │              ▼                                   │
                  │   Cost gate (G_COST_GATE)                        │
                  │   - emits PlannedHashRoute or WcojWithPlan       │
                  │     enum variant (R4) — NEVER Ok(None)           │
                  │   - paper §7.3 conditional-win-on-skew-at-root   │
                  │     carve-out from process lock 3                │
                  │                                                  │
                  └──────────────────────────────────────────────────┘
```

### External candidate disposition

| Candidate | Disposition | Reason |
|---|---|---|
| **HoneyComb** (PACMMOD 2025, [arXiv:2502.06715](https://arxiv.org/abs/2502.06715)) | **ADOPTED** as G_HG_PLAN cost model reference | Pessimistic cardinality estimator with Khamis 2024 refinement; partitions ALL variables; skew-aware; CoCo eager-sort matches xlog W3.1 |
| **Free Join** (SIGMOD 2023, [arXiv:2301.10841](https://arxiv.org/pdf/2301.10841)) | **ADOPTED** as G_RIR_VO plan IR structure reference | Unifies WCOJ + binary; Rust-native; bag-list + per-bag variable order + edge permutation + column swaps fits xlog's `KCliqueVariableOrder` struct exactly |
| **GHD / EmptyHeaded / LevelHeaded** (SIGMOD 2017 / ICDE 2018) | **ADOPTED** as theoretical formalism | GHD = bag-decomposition framework; pairs with Phase-1 G4 helper-splitting which is essentially GHD coarsening |
| **SRDatalog** (arXiv 2604.20073) | **ALREADY ADOPTED** (Phase 1 GPU mechanics) | xlog Phase-1 G1+G4+G5 are SRDatalog-aligned |
| **gHyPart / BiPart / mt-KaHyPar / HyperG / G-kway / iHyperG / Zoltan PHG / KaHyPar / hMETIS / PaToH** | **REJECTED** as wrong-domain | Computation partitioning (cut/connectivity balance), not query planning (variable order for WCOJ) |
| **GDlog / VFLog** | **REJECTED** as binary-join-based | Don't address WCOJ planning |
| **PANDA (submodular width)** | **DEFERRED to v0.7+** | Theoretical refinement; HoneyComb pessimistic estimator is sufficient for v0.6.5 |
| **FHD branch-and-bound** (VLDB 2024) | **DEFERRED to v0.7+** | Static plan caching; not blocking for v0.6.5 |
| **Lai et al. GPU LFTJ warp-parallelism** | **REFERENCE for kernel mechanics** | Warp-based inner-intersect ideas inform G_DISPATCH_PLAN kernel implementation; not import |
| **cuMatch** | **REFERENCE for subgraph WCOJ** | Algorithmic ideas; not import |

### Amendments to goal-038-B sub-goals

Five amendments (R1-R5) refine goal-038-B without changing sub-goal count (9 sub-goals unchanged).

#### R1 — Persistent `WcojRelationMetadata` extension (per Codex pass)

Goal-038-B §3.2 G_HG_PLAN gains S_HG_PLAN.6 strategy:

> **S_HG_PLAN.6** Extend Phase-1 G1's `WcojRelationMetadata` struct to carry per-candidate-root metadata: existing fields `unique_keys: CudaBuffer`, `fan_out: CudaBuffer`, `prefix_sum: CudaBuffer`, `total: u64` (from goal-037 G1 S1.2) PLUS new `per_candidate_root: BTreeMap<VertexId, RootMetadata>` where `RootMetadata { column_permutation, sorted_layout_signature, heat_distribution }`. Persistence: built on first dispatch per (relation, variable-order-context); reused across iterations; invalidated on relation merge.

#### R2 — Plan IR field expansion (per Codex pass)

Goal-038-B §3.3 G_RIR_VO M_RIR_VO.1 struct definition refined:

> **M_RIR_VO.1 (refined)** — New RIR variant `KCliqueVariableOrder { k: u8, variable_positions: [u8; K_MAX], edge_permutation: [u8; EDGE_MAX], column_swaps: Vec<ColumnSwap>, sorted_layout_requirements: SortedLayoutSpec, helper_split_specs: Vec<HelperSplitSpec>, stream_group: StreamGroupId }`. Fields `sorted_layout_requirements`, `helper_split_specs`, `stream_group` are additions per Codex pass. The plan carries enough information for goal-039 G5 stream-mux to consume directly AND for Phase-1 G4 helper-split outputs to attach to the plan.

#### R3 — Generic HG block-slice kernel family over C[] (per Codex pass)

Goal-038-B §3.4 G_DISPATCH_PLAN Goal statement refined:

> **G_DISPATCH_PLAN Goal (refined).** Analyze the K5/K6 dispatch chain for the purpose of REPLACING THREAD-PER-ROW CLIQUE KERNELS WITH ONE GENERIC HG BLOCK-SLICE KERNEL FAMILY OVER C[] (the prefix-sum-flattened root space, extending Phase-1 G1's HG kernel from triangle/4-cycle to K=5..K=8) with respect to consuming G_RIR_VO's `KCliqueVariableOrder` plan from the viewpoint of process lock 23 in the context of removing the hardcoded canonical fallback. The K-clique kernel BODY mirrors the triangle/4-cycle HG kernel body verbatim, just parameterized on K and plan-derived launch params. No new per-K hand-written algorithm. Tier-1 source-audit cert: clique kernel body is one template call per K.

#### R4 — Cost-gate `PlannedHashRoute` carve-out from process lock 3

Goal-038-B §0 gains process lock 29 (renumbered from prior assignment; locks 23-28 already in goal-038-B):

> **Lock 29 — Cost-gate carve-out from process lock 3 (goal-038-B-specific):**
>
> Process lock 3 forbids `Ok(None)` decline for paper-aligned shapes. Goal-038-B G_COST_GATE routes K-clique shapes (paper-aligned per §3 + §5) to WCOJ-with-plan or HASH-by-cost-decision. The HASH-by-cost-decision path is NOT `Ok(None)`. It is an authorized cost-planned HASH route per paper §7.3 conditional-win-on-skew-at-root acknowledgment. Promoter emits structured `MultiwayPlan { route: PlannedHashRoute { reason: PlannedHashReason, planner_evidence: CostPredictionRecord }, ... }` variant. Semantic distinction: `Ok(None)` = promoter cannot handle this shape (forbidden); `PlannedHashRoute` = promoter recognized shape, ran cost model, CHOSE hash (permitted). RIR `MultiwayPlan` enum gains `PlannedHashRoute { reason, planner_evidence }` variant. Cert: source-audit shows zero new `Ok(None)` branches for K=5..K=8 promotion; all hash routing goes through `PlannedHashRoute` variant.

Goal-038-B §3.5 G_COST_GATE M_GATE.2 refined:

> **M_GATE.2 (refined):** Cost gate emits structured `PlannedHashRoute` variant — NOT `Ok(None)`. Process lock 3 remains binding for paper-aligned shapes; lock 29 carve-out permits structured cost-planned HASH routing as positive emission with `PlannedHashReason` + `CostPredictionRecord` for audit traceability. Source-audit cert: zero new `Ok(None)` branches for K=5..K=8 promotion path; `match` on `MultiwayPlan` enum covers `WcojWithPlan(plan)` and `PlannedHashRoute(reason, evidence)` variants exhaustively. Comment in code references paper §7.3 + lock 29.

#### R5 — Stats infrastructure extension scope

Goal-038-B §0 gains process lock 28:

> **Lock 28 — Stats infrastructure extension scope (goal-038-B-specific):**
>
> Goal-038-B G_HG_PLAN consults existing W2.1 + W2.3 + W3.3 (Phase-1 G1 `WcojRelationMetadata`) stats infrastructure. The planner MAY extend existing stats surfaces with new dimensions required by HoneyComb-style pessimistic cardinality estimation: NDV (Number of Distinct Values) per column as extension to `StatsSnapshot` or `RelationCardinalities`; prefix-degree per join key as extension to `WcojRelationMetadata`; per-key heat (skew indicator) as extension to `StatsSnapshot` (or to `WcojRelationMetadata` per-candidate-root metadata per R1). The planner MUST NOT introduce a parallel stats subsystem competing with W2.1's stats pipeline (e.g., a sibling `HoneyCombStatsAccumulator` that bypasses `StatsSnapshot`). Lock 24 (W2.1 extends, doesn't replace) governs. Cert: planner imports only `xlog_stats::*` + `xlog_runtime::executor::wcoj_metadata::WcojRelationMetadata` types; no parallel stats type introduced. New dimensions land on existing types' field surface.

Goal-038-B §3.2 G_HG_PLAN M_HG_PLAN.3 refined:

> **M_HG_PLAN.3 (refined):** Planner consults existing W2.1 + W2.3 + W3.3 stats infrastructure. Stats interface = HoneyComb pessimistic cardinality estimator (Khamis 2024 refinement): cardinality + selectivity + NDV + prefix-degree + per-key heat. New stats DIMENSIONS are PERMITTED as extensions to existing types (`StatsSnapshot`, `RelationCardinalities`, `WcojRelationMetadata`); parallel stats subsystems are FORBIDDEN per lock 28. Cert: planner imports only existing stats types; no new stats accumulator introduced. New fields land on existing types.

---

## Reconciliation note — Supervisor + Codex research convergence

Two independent research passes (supervisor 2026-05-14, Codex 2026-05-15) converged on the same architectural conclusion:

- Both reject GPU hypergraph partitioners (gHyPart, BiPart, mt-KaHyPar, HyperG, G-kway, iHyperG, Zoltan PHG, KaHyPar, hMETIS, PaToH) as wrong-domain.
- Both recommend xlog-native HG-WCOJ planner aligned with SRDatalog mechanics.
- Both endorse the composition: HoneyComb cost model + Free Join plan IR + GHD/EmptyHeaded formalism + SRDatalog GPU runtime.
- Both identify the cost-gate vs process-lock-3 wording ambiguity (resolved by R4) and stats-infrastructure-scope ambiguity (resolved by R5).

Convergence raises supervisor confidence in the architecture substantially. R1-R5 represent the operational tightening required to render the architecture dispatch-ready.

---

## Required Codex follow-up (composite)

After supervisor amendments R1-R5 land in goal-038-B + M_INT.4 amendment lands in goal-038:

1. Execute M_INT.4 per-path absolute wall-time remediation (11 steps above).
2. Resubmit Response 1 for Phase-1 closure with corrected M_INT.4 evidence.
3. On Phase-1 closure approval: dispatch goal-038-B via `/goal @docs/plans/2026-05-14-supervisor-goal-038-B.md` per `feedback_codex_goal_dispatch_flow.md` Tab → Enter sequence.
4. Sequencing decision (38-B-first / Phase-2-first / parallel) is supervisor's at dispatch time.
5. Phase-2 (goal-039) dispatch follows Phase-1 closure independently per goal-039 §8 launch precondition.

---

## Process-lock-numbering note

Goal-038-B's lock list grows from 27 → 29 with this authorization:
- Locks 1-10: inherited from goal-037
- Locks 11-22: inherited from goal-038/039
- Lock 23: hypergraph planner IS production planner (goal-038-B-specific)
- Lock 24: W2.1 extends, doesn't replace
- Lock 25: `BINARY_FALLBACK_KEY_LIMIT` is executor-context-aware
- Lock 26: paper §5 + §7.3 alignment is target
- Lock 27: W5.2 corpus + DTS-DLM dILP-shape are canonical benches
- **Lock 28 (NEW, R5)**: Stats infrastructure extension scope
- **Lock 29 (NEW, R4)**: Cost-gate carve-out from lock 3

---

---

## Authorization 3 — Goal-039 DTS-DLM consumer-surface completeness amendments (R6-R11)

### Decision

Following deep validation of goal-039 against DTS-DLM repo state (FINAL-REPORT pre-M37 research + comprehensive `dts-dlm` codebase Explore survey 2026-05-17), 6 surgical amendments authorized to close gaps in xlog API freeze list + performance contract quantification + determinism scope + DLPack zero-copy coverage + smoke-test spec completeness.

### Rationale

Goal-039's structural coverage is correct (all Stage 4/5 active surface gated; KPI-8 preserves M37-A queued surface; locks 11/12/13/14/22 + KPI-1..8 enforce DTS-DLM consumer contracts). Gaps are surgical:

1. **Gap 1 — `train_and_promote` missing from Group A.** FINAL-REPORT M37-F candidate scope explicitly uses `pyxlog.ilp.train_and_promote` for dILP rule discovery with promotion gates. Lock 11 Group A lists `train_on_compiled_relations` but NOT the meta-API with promotion gates. G_PURGE2 could refactor it away.
2. **Gap 2 — v0.4.0-beta/ga/v0.5.x training surface missing from Group B.** FINAL-REPORT Stone #9 enumerates `register_embedding` + training controls (clipping, early-stopping, scheduler, LR) + Bounded Exact Induction as "already shipped" infrastructure DTS-DLM will integrate via M37-A. Goal-039 Group B only covered v0.4.0-alpha core.
3. **Gap 3 — Circuit cache performance contract unquantified.** 100× claim with no cache-hit rate KPI, no scope/lifetime/eviction spec.
4. **Gap 4 — Dynamic rule injection determinism not covered.** Lock 12 covered stateless evaluation only; M37-A injects rules mid-session.
5. **Gap 5 — Stage-5 witness-chain DLPack zero-copy not covered.** KPI-7 covered Stage 4 only; Stage 5 governance traverses witness chains.
6. **Gap 6 — M_M37A.10 smoke-test spec under-specified.** Generic "exercise Group B symbols"; needed explicit per-symbol enumeration to ensure R7 additions are covered.

### Amendments applied (R6-R11)

| Amendment | Severity | Goal-039 section | Status |
|---|---|---|---|
| **R6** Add `train_and_promote` to lock 11 Group A | MEDIUM (M37-F readiness) | §0 lock 11 Group A | ✅ APPLIED |
| **R7** Extend Group B with `register_embedding` + training controls (clipping, early-stopping, scheduler, LR) + Bounded Exact Induction | HIGH (M37-A readiness) | §0 lock 11 Group B | ✅ APPLIED |
| **R8** Add KPI-9 circuit cache quantification: per-session scope, post-forward-backward survival, ≥ 95% hit rate on M18-D replay, M37-F internal-repeat cache-hit | MEDIUM (M37-A perf contract) | §1.4 KPI-9 + §3.11 G_M37A_SURFACE M_M37A.5 extended | ✅ APPLIED |
| **R9** Extend lock 12 to three regimes: (a) stateless eval, (b) repeated `register_network` calls, (c) program mutation between `evaluate()` calls | HIGH (M37-A correctness) | §0 lock 12 + G_W53 sub-tests | ✅ APPLIED |
| **R10** Extend KPI-7 DLPack zero-copy from Stage 4 hot path to ALSO cover Stage 5 witness-chain traversal | MEDIUM (avoid Stage-5 bottleneck) | §1.4 KPI-7 + G_E2E CUDA-event trace cert | ✅ APPLIED |
| **R11** Enumerate full 11-symbol Group B list in M_M37A.10 smoke-test spec | LOW (test completeness) | §3.11 G_M37A_SURFACE M_M37A.10 | ✅ APPLIED |

### What does NOT need amendment

- **M37-C′ / M37-B / M37-D / M37-E coverage:** M37-C′ verdicted CDD_REGRESSES per prior conversation; M37-B is reward-tuning (DTS-side); M37-D is RL on dLLMs (Stage-1 only); M37-E NSD interleave is deferred to v0.7+ per FINAL-REPORT recommendation. No additional v0.6.5 xlog support required.
- **Stone #4 anti-dialetheic metric:** DTS-side concern; lock 13 (xlog kernels agnostic to pro/contra) is correct separation. KPI-6 witness-chain accessors give DTS-DLM what it needs.
- **Stone #8 single-channel vs dual-channel bridge:** R7's `register_embedding` + training controls land M37-A needs. No additional xlog surface.

### Goal-039 net state post R6-R11

- Sub-goals: 16 (unchanged)
- Process locks: 22 (lock 12 extended in scope; no new locks added by R6-R11)
- KPIs: 9 (KPI-9 added by R8; KPI-1..8 unchanged in count but KPI-7 + KPI-8 scope extended)
- M_M37A.* metrics: 10 (M_M37A.5 + M_M37A.10 extended; count unchanged)
- Process locks 23-29 (38-B specific): unaffected; remain in goal-038-B
- Implementation readiness post-Phase-1-close: Phase-2 dispatch-ready with full DTS-DLM consumer-surface coverage

---

---

## Authorization 4 — M_INT.4 per-cell paired measurement protocol (2026-05-17)

### Decision

M_INT.4 evidence protocol amended from monolithic full-corpus alternation to **per-cell paired Criterion sampling** (via `XLOG_W52_ONLY_CELL` env-var selector, applied symmetrically to integration HEAD AND same-machine W5.2 baseline rerun). Gate ≤ 1.10× per path UNCHANGED. Fixtures 36 cells UNCHANGED. Baseline file basis UNCHANGED (same-machine W5.2 branch rerun).

### Rationale

Monolithic full-corpus alternation introduced batch-effect variance (thermal drift, cache pollution across 36 cells, scheduler noise) that masked the underlying per-cell stable measurements. Targeted paired reruns of failing cells PASSED under per-cell isolation. The gate ≤ 1.10× is correct; the protocol was wrong. Per-cell paired Criterion sampling is the protocol Criterion was designed for and is statistically defensible.

This is NOT gate substitution per process lock 4: measurements use honest `start.elapsed()`; statistical aggregation (Criterion's default ~100 samples per benchmark) is standard bench-rigor.

### Codex execution outcome (2026-05-17)

Codex executed authorized protocol with results substantially exceeding the gate:
- **24/24 per-path medians pass** (12 GPU-WCOJ medians + 12 hash-chain medians per workload across 3 workloads)
- **72/72 integration parity rows present**
- **72/72 W5.2 baseline parity rows present**
- **144/144 expected logs present**
- **0 parser structural problems**

Evidence: `.worktrees/w3-bundle-integration-g38/docs/evidence/2026-05-14-g38-int-mint4-per-path-rerun.md`. Closure proposal re-issued as Response 1 at `.worktrees/w3-bundle-integration-g38/docs/plans/2026-05-14-w3-bundle-closure-proposal.md`.

### Outstanding traceability concern

The `XLOG_W52_ONLY_CELL` selector patch on the W5.2 baseline worktree is currently dirty (uncommitted). Required: capture the selector diff durably in the M_INT.4 per-path rerun evidence file (paste diff text OR commit the patch to a sibling branch like `bench-spike/w52-baseline-selector-g38`) so the M_INT.4 result is reproducible by future auditors. Lighter-weight: paste diff into evidence file.

### Codex follow-up required before Phase-1 board update

1. Capture the `XLOG_W52_ONLY_CELL` selector diff in `2026-05-14-g38-int-mint4-per-path-rerun.md` (paste the diff text under a "Reproducibility — selector patch" section).
2. Optional: commit the W5.2 baseline selector patch to a sibling branch `bench-spike/w52-baseline-selector-g38` for redundancy.

### What this authorization does NOT change

- The amended M_INT.4 metric definition (Authorization 1) remains in force: per-path absolute wall-time ≤ 1.10× same-machine baseline; one-sided; ratio reported INFORMATIONAL only.
- Lock 4 (no bench-gate substitution) remains in force.
- All other Phase-1 evidence (G_W35/G_W36 graceful-close, G_W39 28.39× geomean, M_INT.1-3, M_INT.5-11, G_PURGE) remains valid.

---

---

## Authorization 5 — Goal-038-B W6.7 closure HOLD + 2 new sub-goals (2026-05-17)

### Decision

W6.7 closure-board flip is **HELD** pending two new sub-goals added to goal-038-B: **G_HIST_KC** (runtime-histogram-driven block-slice for K-clique, extending Phase-1 G1's `WcojRelationMetadata` mechanism to K=5..K=8) and **G_HELP_KC** (helper-splitting K-clique invocation, wiring Phase-1 G4's helper-split pass into K-clique promotion). Goal-038-B grows from 9 sub-goals to 11. Closure proposal `ef3fbc7e` (G_HIST_KC + G_HELP_KC not yet implemented) is superseded; a new closure proposal will be written after both new sub-goals + re-validated G_BENCH38B / G_INT38B / G_PURGE38B close.

### Rationale

Supervisor validation of 38-B closure proposal `ef3fbc7e` on 2026-05-17 surfaced two implementation gaps not caught by the 9-sub-goal acceptance gates:

1. **Skew scheduling for K-clique is plan-driven (HoneyComb-style), NOT runtime-histogram-driven (paper §5 Algorithm 1 Phase 1).** `WcojRelationMetadata` (Phase-1 G1's per-relation histogram maintained during Merge) is NOT built in the K-clique provider path. `leader_count` is populated from compile-time plan stats. For non-recursive K-clique this is functionally equivalent; for recursive K-clique within semi-naïve fixpoint, paper §5 mandates per-iteration histogram refresh. Goal-038-B G_HG_PLAN cited HoneyComb cost model as algorithm reference (S_HG_PLAN.7) — that's correct for COMPILE-TIME planning, but does NOT obviate the runtime-histogram requirement at the KERNEL LAUNCH level per paper §5.

2. **Helper-relation splitting for K-clique NOT WIRED.** `HelperSplitSpec` type imported (`promote.rs:81`) per R2 plan-IR field expansion; K-clique promoter at `promote.rs:1466` emits `Vec::<HelperSplitSpec>::new()` (always empty). Phase-1 G4's helper-split pass operates at AST→RIR boundary on full rules — does NOT invoke from within K-clique promotion. K-clique rules with buried-skew at non-leader variables cannot expose that skew via helper-splitting in 38-B as shipped.

These gaps are **architecturally significant**: paper §5 + §7.3 alignment requires both. User-supervisor framing on 2026-05-17 chose option (b) "hold W6.7; require runtime-histogram extension to K-clique + helper-split K-clique invocation before flipping" over (a) approve with documented divergences or (c) split into W6.7/W6.8/W6.9 entries.

### New sub-goals (per Authorization 5)

#### G_HIST_KC — Runtime-histogram-driven block-slice for K-clique (new step 6)

Extends Phase-1 G1's `WcojRelationMetadata` mechanism (originally for triangle + 4-cycle) to K=5..K=8 clique relations.

Sub-goal scope:
- Extend `wcoj_build_metadata_recorded` provider entry to K-clique edge relations (per-edge histogram via existing `multiblock_scan_u32_inplace_on_stream` mechanism)
- Extend K-clique HG kernel template signatures (`wcoj_clique_template_count_hg_grid_t<K, T>`, `wcoj_clique_template_materialize_hg_grid_t<K, T>`) to accept `WcojRelationMetadata` launch params for the leader edge (per `KCliqueVariableOrder.leader_edge_idx`)
- Update K-clique provider entries (`wcoj_clique5_u32_recorded_planned`, etc.) to build metadata for the leader edge before kernel launch
- In recursive context, refresh metadata during Merge phase per Phase-1 G1 mechanism
- Determinism preservation under metadata refresh

Acceptance metrics:
- M_HIST_KC.1: `WcojRelationMetadata` builder extends to K-clique edge relations (provider entry cert, 4 entries: u32/u64 × count/materialize)
- M_HIST_KC.2: K-clique HG kernels accept runtime histogram launch params (kernel signature audit)
- M_HIST_KC.3: K-clique dispatch builds metadata before kernel launch (source audit)
- M_HIST_KC.4: Determinism preserved (bit-exact across 100 runs with `XLOG_DETERMINISTIC=1` + seed-pin on K5/K6 fixtures)
- M_HIST_KC.5: Histogram refresh in recursive context — semi-naïve fixpoint with K-clique recursive body produces bit-exact output across iterations (paper P1 + P4 alignment)
- M_HIST_KC.6: No regression on W5.2 36-cell routing prediction (still 36/36 correct)
- M_HIST_KC.7: Per-iteration histogram refresh cost bounded ≤ 5% of iteration wall-time on `wcoj_w52_skewed_multiway` (bench)
- M_HIST_KC.8: Paper §5 Algorithm 1 Phase 1 source-citation comment present in K-clique kernel + provider path

#### G_HELP_KC — Helper-splitting K-clique invocation (new step 7)

Wires Phase-1 G4's helper-split pass into K-clique promotion when planner detects buried inner-variable skew.

Sub-goal scope:
- Extend HoneyComb-style planner (G_HG_PLAN's `plan_kclique_var_order`) to detect buried-skew condition: heat distribution at non-leader variable significantly higher than leader (configurable threshold, default ratio ≥ 3×)
- When buried skew detected, planner emits `HelperSplitSpec` describing which sub-clique to extract and which variable to elevate
- K-clique promoter at `promote.rs:1466` calls helper_split_pass with the spec, replacing empty-vec emission
- Helper relations correctly emit additional plans that compose with K-clique plan
- Row equality preserved across split vs non-split paths

Acceptance metrics:
- M_HELP_KC.1: Planner detects buried inner-variable skew (heat-distribution cert: positive case + negative case)
- M_HELP_KC.2: K-clique promoter invokes helper_split_pass when planner emits non-empty `HelperSplitSpec` (source audit)
- M_HELP_KC.3: `helper_split_specs` populated non-empty when buried skew present (cert: synthetic K5 fixture with buried-skew variable)
- M_HELP_KC.4: Helper relations emit additional plans that compose with K-clique plan (integration cert)
- M_HELP_KC.5: Row equality on K-clique-with-helper-split vs direct K-clique (bit-exact cert on synthetic fixture)
- M_HELP_KC.6: No regression on K-clique-without-buried-skew (helper-split not invoked → empty vec → same as pre-G_HELP_KC behavior; W5.2 36/36 routing preserved)
- M_HELP_KC.7: Paper §5 Figure 3 helper-relation-splitting source-citation comment present
- M_HELP_KC.8: Integration with G_HIST_KC — helper-split + runtime histogram compose correctly (post-split histogram refresh works)

### Sequencing

DAG insertion between G_COST_GATE (step 5) and G_BENCH38B (now step 8):
- Step 6 (NEW): G_HIST_KC — runtime histogram for K-clique
- Step 7 (NEW): G_HELP_KC — helper-splitting for K-clique
- Step 8 (was step 6): G_BENCH38B — re-runs with new mechanisms active
- Step 9 (was step 7): G_INT38B — re-validates W3.4/W4.1/W5.1/W5.2/W2.5 with new mechanisms
- Step 10 (was step 8): G_PURGE38B — cleanup
- Step 11 (was step 9): G_CLOSE38B — new closure proposal supersedes `ef3fbc7e`

G_HELP_KC must close after G_HIST_KC because helper-split decisions modify relation shapes that the runtime histogram must refresh against.

### What's preserved from current 38-B state

The 9 sub-goal commits already on integration branch (`ef241c7f` through `32dd43c7`) are PRESERVED. They are correctly architectured for the planner foundation; they just don't fully implement paper §5 for K-clique. The new sub-goals build on top, not replace.

Closure proposal `ef3fbc7e` is SUPERSEDED (not deleted) — preserved as evidence of the 9-sub-goal closure state pre-Authorization-5. Final closure proposal includes G_HIST_KC + G_HELP_KC + re-validated downstream sub-goals.

### Out-of-scope (defer to v0.7+ per Authorization 5)

- Stream-aligned multiplexing for K-clique (Phase-1 G5 covers triangle/4-cycle; K-clique stream-mux extension is v0.7+)
- Paper §5 helper-relation splitting BEYOND K-clique into arbitrary deep-join trees (Phase-1 G4 covers chain/triangle/4-cycle; K-clique is Authorization-5 scope; deeper structures are v0.7+)
- Adaptive histogram resolution (per-key heat thresholds tuned dynamically) — Authorization-5 uses static thresholds; adaptive is v0.7+

---

**End of supervisor decision artifact.** Authoritative reference for Codex when executing M_INT.4 remediation + 38-B dispatch (with Authorization-5 extension to 11 sub-goals) + Phase-2 (goal-039 with R6-R11) dispatch + Phase-1 W3 axis board flip (with selector-patch traceability addendum per Authorization 4).
