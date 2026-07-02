#!/usr/bin/env python3
"""Source-B grounding policy — graded pre-collapse neural confidence -> p::atom.

Companion to p1_contested_fact_records.py. Where that extractor selects the
HARD-contested facts (belnap_state==3, contra>=0.25) that drive clone-DIVERGENCE
(Source-A / epistemic axis), this selects facts whose pre-argmax neural support
`pro` is GRADED (genuinely uncertain, not near 0 or 1) and grounds each as a
probabilistic `p::atom` with p = pro.

Axis distinction (do NOT conflate):
  - Source-A (hard-contested, contra>=0.25): pro AND contra both active -> a
    genuine CONFLICT that forks clone-worlds -> clone-divergence / metastability.
    Sparse: fires on ~2 of 16 traces.
  - Source-B (graded-pro, this module): pro alone is uncertain (e.g. 0.54) with
    no opposing contra -> NOT a conflict, does NOT fork clones. It yields graded
    PROBABILISTIC MARGINALS (the vision's "probabilistic features") that argmax
    would collapse to 1. Broad: graded-pro is abundant on ALL 16 traces.

So Source-B is the probabilistic-feature axis with broad coverage, complementary
to Source-A's epistemic-divergence axis -- not a clone-divergence extension.

Grounding band: GRADED_LO < pro < GRADED_HI excludes near-deterministic facts
(pro ~0 or ~1) where there is no pre-collapse uncertainty to preserve.

Run with no args for the self-test; pass a fact_table_dump path to extract.
"""
import json
import sys

GRADED_LO = 0.05
GRADED_HI = 0.95
DEFAULT_TARGET_MODE = "logits_phi"  # graded marginals bias next-step logits


def extract_graded_pro_grounding(facts, *, lo=GRADED_LO, hi=GRADED_HI,
                                 source="graded_pro", target_mode=DEFAULT_TARGET_MODE):
    """Return p::atom grounding records for graded (non-degenerate) pre-collapse pro."""
    records = []
    for f in facts:
        pro = float(f.get("pro", 0.0))
        if not (lo < pro < hi):
            continue
        records.append({
            "fact_id": int(f["fact_id"]),
            "pred_id": int(f.get("pred_id", -1)),
            "arg0": int(f.get("arg0", -1)),
            "arg1": int(f.get("arg1", -1)),
            "p": round(pro, 6),               # p::atom probability = pre-collapse neural confidence
            "contra": round(float(f.get("contra", 0.0)), 6),
            "marginal": None,                 # filled by engine (Source-A/node#3 marginal-compute)
            "source": source,
            "target_mode": target_mode,
        })
    return records


def summarize(records, n_facts):
    if not records:
        return {"n_graded": 0, "coverage_frac": 0.0}
    ps = [r["p"] for r in records]
    return {
        "n_graded": len(records),
        "coverage_frac": round(len(records) / max(n_facts, 1), 4),
        "p_min": round(min(ps), 4), "p_median": round(sorted(ps)[len(ps) // 2], 4),
        "p_max": round(max(ps), 4),
    }


def _self_test():
    facts = [
        {"fact_id": 0, "pred_id": 1, "arg0": 1, "arg1": 2, "pro": 0.54, "contra": 0.0},   # graded -> in
        {"fact_id": 1, "pred_id": 1, "arg0": 3, "arg1": 4, "pro": 0.99, "contra": 0.0},   # near-committed -> out
        {"fact_id": 2, "pred_id": 1, "arg0": 5, "arg1": 6, "pro": 0.02, "contra": 0.0},   # near-zero -> out
        {"fact_id": 3, "pred_id": 1, "arg0": 7, "arg1": 8, "pro": 0.31, "contra": 0.30},  # graded (also hard-contested) -> in
    ]
    recs = extract_graded_pro_grounding(facts)
    ids = sorted(r["fact_id"] for r in recs)
    assert ids == [0, 3], ids                       # only the two graded-pro facts
    assert all(0.05 < r["p"] < 0.95 for r in recs)
    assert recs[0]["marginal"] is None
    print("graded-pro grounded:", ids, "->", summarize(recs, len(facts)))
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        d = json.load(open(sys.argv[1]))
        recs = extract_graded_pro_grounding(d["facts"])
        print(json.dumps({"summary": summarize(recs, len(d["facts"])),
                          "records": recs[:50]}, indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
