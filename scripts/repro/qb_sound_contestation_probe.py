#!/usr/bin/env python3
"""Q-B sub-gate-0 probe — SOUND contestation vs τ-artifact (research instrument).

Consumes a diffusion-deliberation Axis-III readout (per the DTS Axis-III contract,
st_trc_phase2.py): per-context comparison_set of candidates, each carrying
`g_theta` (raw candidate logit), `tau_logit` (production readout temperature),
`within_set_norm` (offset-invariant within-set rank). Decides whether the
contestation (comparison_set >= 2) is SOUND (content-genuine) or a τ-ARTIFACT.

Correct test (NOT "persist at τ->0" — everything collapses to argmax at τ->0):
- SOUND: the top candidates are genuinely close at the NATURAL scale (τ=1, raw
  g_theta) — closeness lives in the raw evidence (STAGE-A qualifier: small margin
  / high entropy at τ=1).
- τ-ARTIFACT: closeness exists ONLY at the high production tau_logit (which
  flattens the distribution) but the candidates are well-separated at τ=1 — the
  high τ inflated the margin.

A sound contestation is invariant to the monotone readout transform (the
within_set_norm offset-invariance property); a τ-artifact is an artifact of the
specific high τ.

Outcome (trust-the-negative): genuine-fraction across contexts >= GENUINE_BAR ->
sound substrate constructible -> unblocks L3-robustness + Phase-2 coupling.
Else -> τ-artifact-only -> determinism wall (consistent M11/M12-null) -> honest stop.

Usage: feed a real readout JSON (list of contexts). Run with no args for the
synthetic self-test that validates the probe logic.
"""
import json
import math
import sys

# STAGE-A pre-registered qualifier thresholds (natural-scale genuineness).
MARGIN_MAX = 0.7        # top1-top2 prob gap at τ=1 must be <= this to be "close"
ENTROPY_EXP_MIN = 1.5   # OR exp(H) >= this (effective #candidates) at τ=1
GENUINE_BAR = 0.50      # fraction of contested sets that must be genuine
TAU_INFLATION_MIN = 1.5 # production tau_logit considered "high" above this


def _softmax(logits, tau):
    m = max(logits)
    exps = [math.exp((x - m) / tau) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def _is_close_at_natural_scale(logits):
    """Genuine closeness at τ=1 (raw evidence), per STAGE-A qualifier."""
    p = _softmax(logits, 1.0)
    p_sorted = sorted(p, reverse=True)
    margin = p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0)
    exp_h = math.exp(_entropy(p))
    return (margin <= MARGIN_MAX) or (exp_h >= ENTROPY_EXP_MIN), margin, exp_h


def classify_context(ctx):
    """Return ('sound'|'tau_artifact'|'degenerate'|'not_contested', detail) for one context.

    ADMISSION GATE (precedes sound-vs-tau): genuine contestation requires >= 2
    ADMITTED/firing candidates. A context whose fallback probe was declined
    (none axis1_admissible / none fired) has no genuine firing competition no
    matter how high the raw-logit entropy looks -> 'degenerate', not 'sound'.
    This closes the uninformative-flatness blind spot."""
    cands = ctx["candidates"]
    admitted = [c for c in cands if c.get("axis1_admissible") or c.get("fired")]
    if len(admitted) < 2:
        return "degenerate", {"n_admitted": len(admitted), "n_members": len(cands),
                              "reason": "declined/non-firing — no >=2 admitted candidates"}
    # sound-vs-tau judged ONLY over the admitted (genuinely-competing) subset
    cands = admitted
    logits = [c["g_theta"] for c in cands]
    # tau_logit is a per-MEMBER field (st_trc_phase2.py:466), set-level constant;
    # read it from the candidates, not the context.
    tau = cands[0].get("tau_logit", ctx.get("tau_logit", 1.0))
    # contested at production τ?
    p_prod = _softmax(logits, tau)
    contested_prod = sum(1 for q in p_prod if q >= 1.0 / len(p_prod)) >= 2 or (
        sorted(p_prod, reverse=True)[0] - sorted(p_prod, reverse=True)[1] <= MARGIN_MAX
        if len(p_prod) > 1 else False)
    if not contested_prod:
        return "not_contested", {"tau": tau}
    genuine, margin, exp_h = _is_close_at_natural_scale(logits)
    if genuine:
        return "sound", {"margin_tau1": round(margin, 4), "exp_H_tau1": round(exp_h, 3), "tau": tau}
    # contested at prod τ but separated at τ=1 -> the high τ inflated it
    return "tau_artifact", {"margin_tau1": round(margin, 4), "exp_H_tau1": round(exp_h, 3),
                            "tau": tau, "high_tau": tau >= TAU_INFLATION_MIN}


def run(readout):
    contexts = readout["contexts"] if isinstance(readout, dict) else readout
    verdicts = [classify_context(c) for c in contexts]
    n_degenerate = sum(1 for v in verdicts if v[0] == "degenerate")
    contested = [v for v in verdicts if v[0] in ("sound", "tau_artifact")]
    sound = [v for v in contested if v[0] == "sound"]
    genuine_frac = (len(sound) / len(contested)) if contested else 0.0
    substrate_sound = bool(contested) and genuine_frac >= GENUINE_BAR
    if substrate_sound:
        verdict = "SOUND"
    elif contested:
        verdict = "TAU_ARTIFACT_ONLY"
    elif n_degenerate:
        verdict = "DEGENERATE_NO_ADMITTED_CONTESTATION"
    else:
        verdict = "NO_CONTESTATION"
    return {
        "n_contexts": len(contexts),
        "n_degenerate": n_degenerate,
        "n_contested": len(contested),
        "n_sound": len(sound),
        "n_tau_artifact": sum(1 for v in contested if v[0] == "tau_artifact"),
        "genuine_fraction": round(genuine_frac, 4),
        "substrate_verdict": verdict,
        "unblocks_phase2": substrate_sound,
    }


def _self_test():
    """Synthetic validation: a sound set (raw-close) and a τ-artifact set
    (raw-separated but flattened by high τ) must classify correctly."""
    A = {"axis1_admissible": True}  # admitted candidates (genuine competition)
    readout = {"contexts": [
        # SOUND: admitted, raw logits genuinely close at τ=1
        {"context_id": "sound1", "candidates": [
            {"g_theta": 2.0, "tau_logit": 1.0, **A}, {"g_theta": 1.85, "tau_logit": 1.0, **A},
            {"g_theta": 0.1, "tau_logit": 1.0, **A}]},
        # τ-ARTIFACT: admitted, raw logits well-separated, production τ=4 flattens them
        {"context_id": "art1", "candidates": [
            {"g_theta": 6.0, "tau_logit": 4.0, **A}, {"g_theta": 1.0, "tau_logit": 4.0, **A},
            {"g_theta": 0.2, "tau_logit": 4.0, **A}]},
        # DEGENERATE: declined probe — no admitted candidates (the real-readout case)
        {"context_id": "deg1", "candidates": [
            {"g_theta": -1.7, "tau_logit": 2.2, "axis1_admissible": False, "fired": False},
            {"g_theta": -1.8, "tau_logit": 2.2, "axis1_admissible": False, "fired": False}]},
    ]}
    per_ctx = [classify_context(c) for c in readout["contexts"]]
    assert per_ctx[0][0] == "sound", per_ctx[0]
    assert per_ctx[1][0] == "tau_artifact", per_ctx[1]
    assert per_ctx[2][0] == "degenerate", per_ctx[2]
    summary = run(readout)
    assert summary["n_sound"] == 1 and summary["n_tau_artifact"] == 1 and summary["n_degenerate"] == 1, summary
    print("SELF-TEST PASS:", json.dumps(summary))
    for cid, v in zip(["sound1", "art1", "deg1"], per_ctx):
        print("  %-8s ->" % cid, v)
    return 0


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            readout = json.load(f)
        print(json.dumps(run(readout), indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
