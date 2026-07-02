#!/usr/bin/env python3
"""Q-B sub-gate-0 — Separability LIFT-over-input on the natural-contestation dual-vector export.

The decisive measure (per main's bar + the Four-Axioms paper's IE-competitive finding):
not "WMIR features beat chance" (an artifact can fake that) but whether the WMIR
coupling-substrate discriminates genuine contestation ABOVE the input-embedding
baseline. Both are 1024-dim, same candidates/labels:
- WMIR Separability = bounded probe on train_features/heldout_features (the substrate).
- INPUT Separability = bounded probe on content_features_by_pair (input/content-token baseline).
- LIFT = AUC_WMIR - AUC_input.  lift>0 (meaningful) -> substrate adds discriminative
  info the input lacks (sound contestation constructible). lift~0 -> input-competitive
  (the paper's structural finding; substrate adds nothing over input).

Run: qb_separability_lift.py <axis3_probe_dual_vector_evidence.json>
"""
import json
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from qb_separability_probe import _standardize, _bounded_logreg, _auc, _balanced_accuracy

LIFT_BAR = 0.05  # mean AUC lift over input above this = substrate adds discriminative info


def _sep(trX, trY, teX, teY):
    if len(np.unique(trY)) < 2 or len(np.unique(teY)) < 2:
        return None
    Xtr, Xte = _standardize(trX, teX)
    p = _bounded_logreg(Xtr, trY.astype(float), Xte)
    return {"auc": _auc(teY, p), "bal_acc": _balanced_accuracy(teY, p)}


def run(ev):
    cells = list(ev.values()) if isinstance(ev, dict) else ev
    per = []
    seen_hashes = {}
    for i, c in enumerate(cells):
        nbp = c.get("neural_body_probe", c)
        trp = [tuple(p) for p in nbp["train_pairs"]]
        hep = [tuple(p) for p in nbp["heldout_pairs"]]
        trY = np.asarray(nbp["train_labels"], float)
        teY = np.asarray(nbp["heldout_labels"], float)
        wmir_tr = np.asarray(nbp["train_features"], float)
        wmir_te = np.asarray(nbp["heldout_features"], float)
        cf = {tuple(e["pair"]): e["value"] for e in nbp["content_features_by_pair"]}
        # build content matrices aligned to the SAME pairs/labels (row i <-> pair i <-> label i)
        if not all(p in cf for p in trp + hep):
            per.append({"probe": i, "skip": "incomplete content coverage"}); continue
        cont_tr = np.asarray([cf[p] for p in trp], float)
        cont_te = np.asarray([cf[p] for p in hep], float)
        h = hash(wmir_te.tobytes())
        dup = seen_hashes.get(h)
        seen_hashes.setdefault(h, i)
        wmir = _sep(wmir_tr, trY, wmir_te, teY)
        cont = _sep(cont_tr, trY, cont_te, teY)
        rec = {"probe": i, "n_tr": len(trY), "n_te": len(teY),
               "te_pos_frac": round(float(np.mean(teY)), 3),
               "dup_of": dup}
        if wmir and cont:
            rec.update({"wmir_auc": round(wmir["auc"], 4), "input_auc": round(cont["auc"], 4),
                        "lift_auc": round(wmir["auc"] - cont["auc"], 4),
                        "wmir_bal": round(wmir["bal_acc"], 4), "input_bal": round(cont["bal_acc"], 4)})
        else:
            rec["unscorable_labels"] = True
        per.append(rec)
    # aggregate over UNIQUE scorable probes only
    uniq = [r for r in per if r.get("dup_of") is None and "lift_auc" in r]
    if uniq:
        mean_lift = float(np.mean([r["lift_auc"] for r in uniq]))
        mean_wmir = float(np.mean([r["wmir_auc"] for r in uniq]))
        mean_input = float(np.mean([r["input_auc"] for r in uniq]))
        verdict = "SUBSTRATE_ADDS_LIFT" if mean_lift >= LIFT_BAR else "INPUT_COMPETITIVE_NO_LIFT"
    else:
        mean_lift = mean_wmir = mean_input = None
        verdict = "UNSCORABLE"
    return {"n_probes": len(cells), "n_unique_scorable": len(uniq),
            "mean_wmir_auc": round(mean_wmir, 4) if uniq else None,
            "mean_input_auc": round(mean_input, 4) if uniq else None,
            "mean_lift_auc": round(mean_lift, 4) if uniq else None,
            "lift_bar": LIFT_BAR, "verdict": verdict, "per_probe": per}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        ev = json.load(f)
    print(json.dumps(run(ev), indent=2))
