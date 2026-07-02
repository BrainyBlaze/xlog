#!/usr/bin/env python3
"""P1 deliverable — grounded contested-fact record extractor for the in-loop contour.

Reference implementation of the interface @dts-dlm-main requested: given the
post-admit/propagate WMIR fact table for a step, return the set of GROUNDED
contested fact records to route into clone-divergence + Stage-1 feedback.

Eligibility mirrors the production Belnap semantics
(dts_dlm/propagate/belnap.py classify_belnap_state):

    belnap_state == 3 (Both)  iff  pro > 0  AND  contra >= CONTRA_ACTIVE_THRESHOLD

CONTRA_ACTIVE_THRESHOLD = 0.25 is the live activation gate. A fact whose contra
sits in the 0.1-0.25 twilight is NOT clone-seed-eligible (it reads pro-dominant),
which is why admission contestation is sparse at the working threshold (222/16
traces, concentrated in NB2/NB3) even though graded contestation is broader.

Record shape (what @dts-dlm-main routes; marginal/clone_id filled downstream):
    {fact_id, pred_id, arg0, arg1, pro, contra, belnap_state,
     marginal: None, clone_id: None, source, target_mode}

This is the Milestone-1 slice: ADMISSION-grounded contested facts only. It does
NOT depend on contra-propagation through derivation (the full-loop P1 task) — the
admission facts already carry contra>=0.25 where genuine conflict was admitted.

Run with no args for the self-test; pass a fact_table_dump path to extract.
"""
import json
import sys

CONTRA_ACTIVE_THRESHOLD = 0.25  # mirrors dts_dlm/propagate/belnap.py
DEFAULT_TARGET_MODE = "guidance_weights"  # reuses _emit_guidance_weights_from_fact_ids


def classify_belnap_state(pro: float, contra: float,
                          threshold: float = CONTRA_ACTIVE_THRESHOLD) -> int:
    """0=neither, 1=true, 2=false, 3=both — matches belnap.py classify_belnap_state."""
    contra_active = contra >= threshold
    pro_pos = pro > 0.0
    if pro_pos and contra_active:
        return 3
    if pro_pos:
        return 1
    if contra_active:
        return 2
    return 0


def extract_contested_fact_records(facts, *, threshold=CONTRA_ACTIVE_THRESHOLD,
                                   source="admission",
                                   target_mode=DEFAULT_TARGET_MODE):
    """Return clone-seed-eligible contested-fact records (belnap_state == 3)."""
    records = []
    for f in facts:
        pro = float(f.get("pro", 0.0))
        contra = float(f.get("contra", 0.0))
        if classify_belnap_state(pro, contra, threshold) != 3:
            continue
        records.append({
            "fact_id": int(f["fact_id"]),
            "pred_id": int(f.get("pred_id", -1)),
            "arg0": int(f.get("arg0", -1)),
            "arg1": int(f.get("arg1", -1)),
            "pro": round(pro, 6),
            "contra": round(contra, 6),
            "belnap_state": 3,
            "marginal": None,    # filled by Source-A engine (evaluate_query_probabilities)
            "clone_id": None,    # assigned by WorldCloner clone-seed enumerator
            "resolution": None,  # clone branch: "pro" | "contra" — clone identity, NOT a routing channel
            "source": source,
            "target_mode": target_mode,  # Stage-1 feedback channel (guidance_weights|stage5_summary|logits_phi) — routing, orthogonal to resolution
        })
    return records


def summarize(records):
    return {
        "n_contested": len(records),
        "contra_min": round(min((r["contra"] for r in records), default=0.0), 4),
        "contra_max": round(max((r["contra"] for r in records), default=0.0), 4),
        "all_state_both": all(r["belnap_state"] == 3 for r in records),
    }


def _self_test():
    facts = [
        {"fact_id": 0, "pred_id": 10, "arg0": 1, "arg1": 2, "pro": 0.6, "contra": 0.0},   # true
        {"fact_id": 1, "pred_id": 10, "arg0": 3, "arg1": 4, "pro": 0.6, "contra": 0.30},  # BOTH
        {"fact_id": 2, "pred_id": 11, "arg0": 5, "arg1": 6, "pro": 0.5, "contra": 0.18},  # twilight -> true (below 0.25)
        {"fact_id": 3, "pred_id": 11, "arg0": 7, "arg1": 8, "pro": 0.0, "contra": 0.40},  # false
        {"fact_id": 4, "pred_id": 12, "arg0": 9, "arg1": 0, "pro": 0.5, "contra": 0.50},  # BOTH (max)
    ]
    recs = extract_contested_fact_records(facts)
    ids = sorted(r["fact_id"] for r in recs)
    assert ids == [1, 4], ids  # only the two genuine state==3 facts
    assert all(r["belnap_state"] == 3 for r in recs)
    assert recs[0]["marginal"] is None and recs[0]["clone_id"] is None
    # threshold sharpness: 0.18 contra must NOT qualify, 0.25 must
    assert classify_belnap_state(0.6, 0.18) == 1
    assert classify_belnap_state(0.6, 0.25) == 3
    print("contested records extracted:", ids, "->", summarize(recs))
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        d = json.load(open(sys.argv[1]))
        recs = extract_contested_fact_records(d["facts"])
        print(json.dumps({"summary": summarize(recs), "records": recs[:50]}, indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
