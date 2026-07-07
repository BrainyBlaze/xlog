#!/usr/bin/env python3
"""Source-B per-step selection policy — tractable graded-pro subset to ground.

The graded-pro pool is large (~13k facts across 16 traces). Exact WMC is
exponential in the induced treewidth, so we cannot ground the whole pool in one
program. This selects a bounded, RELEVANT subset to ground as p::atom per step.

Relevance principle = locality around the clone-fork points:
  1. ANCHOR on the contested facts (belnap_state==3) — these are where clones
     fork; their consequence-marginal is what the step actually needs.
  2. EXPAND by entity-locality: graded-pro facts sharing an argument (arg0/arg1)
     with an anchor are in the same logical world-region (they feed the same
     derivations). 1-hop entity overlap is a tractable proxy for rule-derivation
     reachability (the principled upgrade when the rule graph is available).
  3. RANK the remaining graded-pro facts by uncertainty (|pro - 0.5| ascending —
     most-uncertain first, the pre-collapse information argmax would erase).
  4. CAP total at K (tractability budget). Anchors are always kept.

Honest caveat: K is a var-count budget; true WMC tractability depends on the
induced sub-program's treewidth. Entity-locality keeps the region connected and
bounded, but the selected set's treewidth must be verified at compile time.

Run with no args for the self-test; pass a fact_table_dump path to select.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from p1_contested_fact_records import extract_contested_fact_records
from p1_source_b_grounding import extract_graded_pro_grounding

DEFAULT_K = 32


def _entities(rec):
    return {rec["arg0"], rec["arg1"]}


def select_grounding_subset(facts, *, anchor_fact_id=None, k=DEFAULT_K):
    """Select a tractable per-step grounding set around ONE anchor.

    The contour processes ONE clone-fork (one contested fact) at a time — the
    engine is single-world per clone (clone_id = 2*fact_id), so we ground that
    single anchor plus its entity-local graded-pro neighborhood, NOT all
    contested facts (grounding N contested simultaneously = 2**N worlds,
    intractable and architecturally wrong).

    anchor_fact_id: the fork-point (a contested fact) or query atom for this
    step. If None, falls back to the single most-contested fact (highest
    min(pro,contra)) as the anchor, or — on a no-contestation trace — to the K
    most-uncertain graded-pro facts (pure probabilistic-marginal mode).
    """
    contested = extract_contested_fact_records(facts)
    graded = extract_graded_pro_grounding(facts)
    by_id = {f["fact_id"]: f for f in facts}

    if anchor_fact_id is None and contested:
        anchor_fact_id = max(
            contested, key=lambda c: min(c["pro"], c["contra"])
        )["fact_id"]

    if anchor_fact_id is None:
        # no contestation: pure probabilistic-marginal mode, top-K most-uncertain
        cands = sorted(graded, key=lambda g: abs(g["p"] - 0.5))[:k]
        return {"k": k, "mode": "no_anchor_graded_marginal", "anchor_fact_id": None,
                "n_graded_pool": len(graded), "n_selected_total": len(cands),
                "selected_graded_entity_neighbors": 0, "graded": cands}

    anchor = by_id[anchor_fact_id]
    anchor_entities = {int(anchor["arg0"]), int(anchor["arg1"])}
    cands = [g for g in graded if g["fact_id"] != anchor_fact_id]

    def rank_key(g):
        neighbor = 0 if (_entities(g) & anchor_entities) else 1
        return (neighbor, abs(g["p"] - 0.5))
    cands.sort(key=rank_key)

    selected_graded = cands[: max(0, k - 1)]  # 1 slot for the anchor
    n_neighbors = sum(1 for g in selected_graded if _entities(g) & anchor_entities)
    return {
        "k": k, "mode": "per_fork_anchor", "anchor_fact_id": anchor_fact_id,
        "n_contested_in_trace": len(contested), "n_graded_pool": len(graded),
        "n_graded_selected": len(selected_graded),
        "n_selected_total": 1 + len(selected_graded),
        "selected_graded_entity_neighbors": n_neighbors,
        "graded": selected_graded,
    }


def consume_source_b_sink(sink_record, *, k=DEFAULT_K):
    """Drive per-fork selection from a wired per_step_source_b_record.

    sink_record shape (dts-dlm runtime.py:6077, default-off):
      {step_index, facts:[{fact_id,pred_id,arg0,arg1,pro,contra}],
       fork_anchors:[{fact_id,clone_id,resolution}]}

    Returns one bounded grounding set per fork anchor, carrying the producer's
    clone_id/resolution through (so the engine marginal-compute + routing index
    on the same clone_id). If there are no fork anchors (thin step), returns a
    single no-anchor graded-marginal selection over the step's facts.
    """
    facts = sink_record["facts"]
    forks = sink_record.get("fork_anchors", [])
    if not forks:
        sel = select_grounding_subset(facts, anchor_fact_id=None, k=k)
        return [{"step_index": sink_record.get("step_index"), "clone_id": None,
                 "resolution": None, **sel}]
    out = []
    for fk in forks:
        sel = select_grounding_subset(facts, anchor_fact_id=fk["fact_id"], k=k)
        out.append({"step_index": sink_record.get("step_index"),
                    "clone_id": fk.get("clone_id"), "resolution": fk.get("resolution"),
                    **sel})
    return out


def _self_test():
    # 2 contested anchors over entities {1,2} and {3,4}; graded facts: some share
    # entities (neighbors), some don't; one near-committed must be excluded.
    facts = [
        {"fact_id": 0, "pred_id": 1, "arg0": 1, "arg1": 2, "pro": 0.6, "contra": 0.30},  # anchor
        {"fact_id": 1, "pred_id": 1, "arg0": 3, "arg1": 4, "pro": 0.6, "contra": 0.30},  # anchor
        {"fact_id": 2, "pred_id": 2, "arg0": 1, "arg1": 9, "pro": 0.55, "contra": 0.0},  # neighbor (shares 1)
        {"fact_id": 3, "pred_id": 2, "arg0": 7, "arg1": 8, "pro": 0.50, "contra": 0.0},  # non-neighbor, most uncertain
        {"fact_id": 4, "pred_id": 2, "arg0": 20, "arg1": 21, "pro": 0.99, "contra": 0.0},  # near-committed -> excluded
    ]
    # per-fork: anchor on contested fact 0 (entities {1,2}); fact 2 shares entity 1
    out = select_grounding_subset(facts, anchor_fact_id=0, k=3)
    assert out["mode"] == "per_fork_anchor" and out["anchor_fact_id"] == 0, out
    sel_ids = [g["fact_id"] for g in out["graded"]]
    assert sel_ids[0] == 2, sel_ids            # neighbor (shares entity 1) ranks first
    assert 4 not in sel_ids, sel_ids           # near-committed excluded (not graded)
    assert out["n_selected_total"] <= 3        # anchor + (k-1) graded
    assert out["selected_graded_entity_neighbors"] >= 1
    # no-anchor fallback on a contestation-free fact set -> graded-marginal mode
    out2 = select_grounding_subset(
        [{"fact_id": 9, "pred_id": 2, "arg0": 1, "arg1": 2, "pro": 0.5, "contra": 0.0}], k=4)
    assert out2["mode"] == "no_anchor_graded_marginal", out2
    print("per-fork selection:", {k: v for k, v in out.items() if k != "graded"})
    print("graded ids (neighbor-first):", sel_ids)
    # sink-consumption: drive selection from a wired per_step_source_b_record
    sink = {
        "step_index": 3,
        "facts": facts,
        "fork_anchors": [{"fact_id": 0, "clone_id": 0, "resolution": "pro"},
                         {"fact_id": 0, "clone_id": 1, "resolution": "contra"}],
    }
    consumed = consume_source_b_sink(sink, k=3)
    assert len(consumed) == 2, consumed                       # one selection per fork
    assert {c["resolution"] for c in consumed} == {"pro", "contra"}, consumed
    assert {c["clone_id"] for c in consumed} == {0, 1}, consumed
    assert all(c["anchor_fact_id"] == 0 for c in consumed), consumed
    assert all(c["step_index"] == 3 for c in consumed), consumed
    print("sink-consumption: 2 fork selections, clone_ids", [c["clone_id"] for c in consumed],
          "resolutions", [c["resolution"] for c in consumed])
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K
        d = json.load(open(sys.argv[1]))
        out = select_grounding_subset(d["facts"], k=k)
        slim = {kk: vv for kk, vv in out.items() if kk not in ("anchors", "graded")}
        print(json.dumps(slim, indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
