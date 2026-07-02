#!/usr/bin/env python3
"""Q-B sub-gate-0 — bounded Separability probe (axiom-integrated, research instrument).

Implements the Separability axiom (within-task discriminator accuracy) on the
recovered Axis-III probe evidence (axis3_probe_evidence.json: per-cell
neural_body_probe with 1024-dim train_features/heldout_features + binary labels).

Measure: a BOUNDED (strong-L2) linear probe trained on a cell's train_features
-> train_labels, evaluated on HELD-OUT features -> labels. Heldout balanced-
accuracy / AUC vs chance (0.5) is the functional-discriminability signal:
- testing on HELD-OUT candidates (not trained-on) enforces anti-overfit (a
  memorizing probe scores chance on heldout), per the bounded-probe safeguard.
- balanced-accuracy (not raw) because labels are imbalanced (~94% positive);
  raw accuracy is gamed by predicting the majority.

This is the FROZEN baseline (pre-coupling, in-run fallback features). Phase-2
falsifier = post-coupling Separability LIFT over this baseline AND over an
input-embedding baseline (the latter NOT available here — content_features_by_pair
is empty — so this run reports vs-chance only; lift-over-input is deferred).

Verdict: mean heldout balanced-accuracy across cells. >> chance (bounded) ->
features carry discriminative structure (separable). ~chance -> degenerate (no
within-set discriminative structure) = the paper's within-task-collapse finding
+ our determinism-root, on the frozen substrate.

Run with no args for the synthetic self-test (validates the bounded probe).
"""
import json
import sys
import math

import numpy as np

L2_LAMBDA = 10.0      # strong regularization -> bounded capacity (1024-dim, ~92 samples)
LR = 0.5
N_ITERS = 300
SEPARABLE_BAR = 0.65  # mean heldout balanced-accuracy above this = separable (bounded)


def _standardize(X_tr, X_te):
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _bounded_logreg(X_tr, y_tr, X_te):
    """Numpy L2-regularized logistic regression (GD); returns held-out probabilities."""
    n, d = X_tr.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(N_ITERS):
        z = X_tr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y_tr
        gw = (X_tr.T @ g) / n + L2_LAMBDA * w / n
        gb = g.mean()
        w -= LR * gw
        b -= LR * gb
    return 1.0 / (1.0 + np.exp(-(X_te @ w + b)))


def _balanced_accuracy(y, p):
    yhat = (p >= 0.5).astype(int)
    y = y.astype(int)
    accs = []
    for cls in (0, 1):
        m = (y == cls)
        if m.sum() > 0:
            accs.append((yhat[m] == cls).mean())
    return float(np.mean(accs)) if accs else float("nan")


def _auc(y, p):
    y = y.astype(int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    allp = np.concatenate([pos, neg])
    order = allp.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allp) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separability_for_cell(tr_X, tr_y, te_X, te_y):
    if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
        return {"degenerate_labels": True, "bal_acc": float("nan"), "auc": float("nan"),
                "n_tr": len(tr_y), "n_te": len(te_y),
                "te_pos_frac": round(float(np.mean(te_y)), 3)}
    Xtr, Xte = _standardize(tr_X, te_X)
    p = _bounded_logreg(Xtr, tr_y.astype(float), Xte)
    return {"degenerate_labels": False,
            "bal_acc": round(_balanced_accuracy(te_y, p), 4),
            "auc": round(_auc(te_y, p), 4),
            "n_tr": len(tr_y), "n_te": len(te_y),
            "te_pos_frac": round(float(np.mean(te_y)), 3)}


def run(probe_evidence):
    cells = list(probe_evidence.values()) if isinstance(probe_evidence, dict) else probe_evidence
    per_cell = []
    for c in cells:
        nbp = c.get("neural_body_probe", c)
        tr_X = np.asarray(nbp["train_features"], dtype=float)
        tr_y = np.asarray(nbp["train_labels"], dtype=float)
        te_X = np.asarray(nbp["heldout_features"], dtype=float)
        te_y = np.asarray(nbp["heldout_labels"], dtype=float)
        per_cell.append(separability_for_cell(tr_X, tr_y, te_X, te_y))
    scored = [c for c in per_cell if not c["degenerate_labels"] and not math.isnan(c["bal_acc"])]
    mean_bal = float(np.mean([c["bal_acc"] for c in scored])) if scored else float("nan")
    mean_auc = float(np.mean([c["auc"] for c in scored])) if scored else float("nan")
    separable = bool(scored) and mean_bal >= SEPARABLE_BAR
    return {"n_cells": len(per_cell), "n_scored": len(scored),
            "mean_heldout_bal_acc": round(mean_bal, 4) if scored else None,
            "mean_heldout_auc": round(mean_auc, 4) if scored else None,
            "chance_bal_acc": 0.5,
            "separable_verdict": ("SEPARABLE" if separable else "DEGENERATE_NEAR_CHANCE")
                                 if scored else "UNSCORABLE_LABELS",
            "input_embedding_baseline": "UNAVAILABLE (content_features_by_pair empty) — vs-chance only; lift-over-input deferred",
            "frozen_baseline": True, "per_cell": per_cell}


def _self_test():
    rng = np.random.RandomState(0)
    # SEPARABLE cell: features carry the label (signal in dim 0)
    def sep_cell(n_tr, n_te):
        def make(n):
            y = rng.randint(0, 2, n).astype(float)
            X = rng.randn(n, 1024) * 0.5
            X[:, 0] += 3.0 * y  # label signal
            return X, y
        Xtr, ytr = make(n_tr); Xte, yte = make(n_te)
        return {"neural_body_probe": {"train_features": Xtr.tolist(), "train_labels": ytr.tolist(),
                "heldout_features": Xte.tolist(), "heldout_labels": yte.tolist()}}
    # DEGENERATE cell: features pure noise, label uncorrelated
    def deg_cell(n_tr, n_te):
        def make(n):
            y = rng.randint(0, 2, n).astype(float)
            X = rng.randn(n, 1024)
            return X, y
        Xtr, ytr = make(n_tr); Xte, yte = make(n_te)
        return {"neural_body_probe": {"train_features": Xtr.tolist(), "train_labels": ytr.tolist(),
                "heldout_features": Xte.tolist(), "heldout_labels": yte.tolist()}}
    sep = run({"a": sep_cell(92, 32), "b": sep_cell(56, 18)})
    deg = run({"a": deg_cell(92, 32), "b": deg_cell(56, 18)})
    print("SEPARABLE synthetic ->", sep["separable_verdict"], "bal_acc", sep["mean_heldout_bal_acc"])
    print("DEGENERATE synthetic ->", deg["separable_verdict"], "bal_acc", deg["mean_heldout_bal_acc"])
    assert sep["separable_verdict"] == "SEPARABLE", sep
    assert deg["separable_verdict"] == "DEGENERATE_NEAR_CHANCE", deg
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            ev = json.load(f)
        print(json.dumps(run(ev), indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
