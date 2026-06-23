# Plasticity & Saliency Rule Induction (demo)

Learns a symbolic STDP/LTP plasticity rule by inducing the correct candidate
among distractors with xlog's multi-rule neural-bodied joint mixture.

**Planted ground truth:** an edge *strengthens* iff it has a pre-before-post
coincidence AND its saliency >= 0.5.

**Candidates competing for `strengthens(Edge)`:**
- `cand_prepost_rel`    — relational-only pre-post (over-fires on weak coincidences)
- `cand_prepost_neural` — pre-post AND a learned saliency gate (**the true rule**)
- `cand_postpre_neural` — post-pre AND a learned gate (wrong-timing distractor)

The demo trains all three, selects the winner by **held-out coverage** (guard-free),
and admits it with a faithful held-out read. The induced winner is
`cand_prepost_neural`; it generalizes to new strong coincidences and stays
vigilant against weak/wrong-timing ones. A second outcome, `weakens` (LTD via
post-before-pre), is available via `run_demo(..., outcome="weakens")` and shows
the same framework recovers the opposite plasticity direction with no new mechanism.

## Run (requires CUDA)

    python examples/plasticity_saliency/run_demo.py

Verified on an RTX 3090 (CUDA 12.9): the recovery, held-out generalization/vigilance,
zero-host, and rule-inventory tests in `python/tests/test_plasticity_demo.py` all pass.

## Scope (honest)

This demo runs entirely on the current engine. The neural saliency is a
torch-side straight-through gate `g_theta(phi) >= tau` over a fixed per-edge
feature `phi` (no backbone gradient); the existential event->edge aggregation is
projected to head-bound ground relations (`edge_pre_post`) in preprocessing,
because existential-join trainable bodies are not yet supported on the engine.
xlog provides the relational gating, multi-rule noisy-OR mixture, candidate
selection, and the rule/proof inventory. Lifting saliency into an in-circuit
neural predicate over a real event domain is the separate "Stage B" engine track.
