# Factorized Program Finalization (Tier 1) — Evidence

Date: 2026-06-14. Branch `feat/factorized-finalize` from main `e23b70a9`. Co-designed with
`@dts-dlm-main` (research) per `@human`'s directive; approved Tier-1 scope.

The factorized-hypergraph research program: D1 aggregate-fused WCOJ, D2 GPU Free Join, D3
factorized recursive deltas (dense + sparse) — all merged to origin/main with measured gate
evidence. D4 factorized provenance — parked, verified negative (cost is CDCL verification,
not the d-DNNF frontier). This Tier-1 slice resolves the remaining hygiene/robustness gaps.

## 1. pyxlog D3 observability (done)

`factorized_delta_dispatch_count` existed on `Executor` but was absent from the
consumer-facing `WcojDispatchStats` (xlog-gpu) and pyxlog `wcoj_dispatch_stats` (which
exposed free_join + groupby_fusion + error_decline only). Added the field, populated it from
the executor, exposed it in the pyxlog dict and `_native.pyi`. Completes D1/D2/D3 dispatch
observability for Python consumers.

## 2. Fail-open cost-model loss-veto on D1/D2 (done)

The factorized routes fired on **shape + env kill-switch only** — never consulting the W2.5
cost model (which gates the base triangle/4-cycle). So they could fire in their loss regions
(D2's measured 1.7–2.0× cost-of-generality on small joins; the small-triangle region the base
triangle already declines). Per report §6.3 ("extend the W2.5 cost model rather than add a
new oracle"), added `WcojCostModel::factorized_loss_veto`:

- Vetoes (declines factorized → binary fallback) ONLY when the model has cardinality stats
  for **every** slot relation AND the **largest** is below the WCOJ-worthwhile threshold
  (4096) — a provably-small join where no intermediate can blow up and binary wins.
- **FAIL-OPEN**: stats absent for any slot, or any slot large → no veto. This NEVER vetoes a
  large-input case (where factorized wins on the avoided large intermediate) and NEVER
  vetoes when stats are unavailable (recursive deltas on early iterations). Every measured
  D1/D2 gate win is preserved by construction.
- Wired into `try_dispatch_free_join` (D2) and the fused-triangle path of
  `try_dispatch_wcoj_groupby_root_agg` (D1). The rarer fused 4-cycle/K-clique sub-paths
  inherit their base shapes' gating posture (not separately vetoed).

D3 needs no veto — it is already cost-aware (dense work-floor + sparse distinct-aware
sizing). Documented as such.

Validation: 5 unit tests (fires-all-small / declines-on-large / fail-open-missing /
general-arity / skew-never); no-regression — D1 fusion 29/29 (4 suites) + integration,
D2 e2e 6/6, D3 e2e 9/9 all unchanged (fail-open: tiny statless e2e fixtures fire as before).

## 3. Flatten-boundary guarantee (verified clean — documented)

All three factorized routes materialize their output to normal `CudaBuffer` rows **before**
the store/DLPack/Arrow edge — no factorized intermediate escapes (report §6.4). D1's fused
aggregate kernel emits `(key, agg)` rows; D2's Free Join emit kernel materializes the
frontier to dense rows; D3's `fj_delta_novel_*` emits the deduped novel set as rows. Each is
installed via `union_gpu` + `store_put` in `recursive.rs`. No flatten step is needed because
the kernels produce row buffers directly; the boundary is structurally safe.

## 4. P0.3 calibration data (handed to @xlog-claude)

`crates/xlog-prob/tests/d4_verify_calibration_caps.rs`: GpuCnf (var_cap, clause_cap) at the
dense-correlated D4-verify boundary — n=5 168/312, n=6 351/668 (47s, completes), n=7 654/1262
(launch-fail), n=8 1119/2178. Read post-encode pre-compile so n=7 does not crash. Finding:
the verify explosion is **treewidth-driven, not size-driven** (onset at ~654 vars, where
legitimate medium programs live), so a pure size bound is too coarse and the CDCL
branch-budget backstop must be primary — fed `@xlog-claude`'s P0.3 conflict-budget design.

## 5. D2 skew/order decider (the Tier-2 gate) — Tier 2 INDICATED

`test_free_join_e2e.rs::d2_skew_order_decider`: an adversarial blow-up chain (prefix → N²,
result = 1 row) where FJ's prefix constraint forces materializing the large intermediate
while the binary fallback reorders to exploit the selective tail. Result: **FJ peak 746 KB
vs binary 243 KB = 3.07×**. The fail-open veto does not catch it (large input). Per the
pre-registered rule, this promotes Tier 2 (order-aware FJ planning) to indicated.

Honest framing: the fixture is adversarial — it proves the order-loss EXISTS and is
structural (FJ cannot reorder a chain; binary can), NOT that real consumer bodies commonly
hit it. Decision (Tier 2 full planner / Tier 1.5 cardinality-greedy reorder / document as
known limitation) is `@human`'s; pending.

## Status

Tier 1 complete and validated. The decider surfaced a real FJ order-loss → the program is NOT
fully closed until the Tier-2 decision lands. No overclaim: D1–D3 ship as measured; D4 is a
verified negative; the FJ bad-order case is a documented, decider-proven gap.
