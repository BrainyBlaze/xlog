#!/usr/bin/env python3
"""STAGE-A — sound-vs-temperature qualifier (Tier-0 diagnostic, research instrument).

Third STAGE-A measure (alongside distribution-sharpness and downstream
posterior-movement). Question this instrument answers, and ONLY this one:

    Is the uncertainty in a candidate source GENUINE structured content, or is it
    merely an artifact of the softmax temperature being warm?

This matters because sharpness (entropy / top-1 mass) is, by itself, NOT a sound
signal: temperature scaling divides every score by the same tau, so entropy can
be dialed anywhere between one-hot (tau->0) and uniform (tau->inf) without any
change to the underlying structure. A source can look "uncertain" purely because
tau is warm. We must separate that artifact from real competing mass.

The separation is exact. Dividing every logit by tau divides every gap between
sorted logits by tau, so the *location* of the largest gap in the sorted score
vector is invariant to tau. That location partitions the vector into a leading
"competing cluster" and a trailing tail:

    m* = number of scores before the largest gap = effective competing-mode count

    m* == 1 with a dominant leading gap -> a single mode dominates; any apparent
        softmax uncertainty at the operating tau is a temperature artifact (and at
        the limit, this is the determinism-root: collapse to one fact).
    m* >= 2                              -> genuine multimodal competition that no
        single temperature can resolve into determinism without also flattening
        everything; this is SOUND, non-input-substitutable uncertainty.
    no dominant gap (flat profile)       -> undifferentiated mass; not structure,
        just noise/near-uniform; UNSOUND for a different reason.

Source-agnostic by construction. A2 (stage1 neural softmax pre-argmax) supplies
logits; A1 (pro_seed / contra_seed VQ geometry pre-coalesce) supplies negative
codebook distances / similarity scores. Both are just a per-probe score vector;
the same tau-invariant statistic applies to each. This is why the export must
carry the PRE-collapse score vector per probe -- the argmax atom has already
discarded exactly the structure measured here.

The tau-sweep in the report is the VALIDATION, not the measure: it confirms m*
is stable across a tau band (it must be, by the gap-location argument) while
entropy is not -- the contrast is the deliverable ("sharpness moves with tau,
mode-structure does not; here is which source carries sound mode-structure").

Run with no args for the synthetic self-test (validates the qualifier on known
SOUND / DETERMINISTIC / UNIFORM sources).
"""
import json
import sys
import math

import numpy as np

# Instrument defaults. These are the qualifier's own defensible bars; the
# pre-registered STAGE-A scoring thresholds (@dts-dlm-glm, dc2f3e9) are
# authoritative and override these when the scoring line runs.
#
# A sound competing cluster is SMALL (genuine rivals, not a broad plateau) ...
SOUND_MODE_FLOOR = 2
SOUND_MODE_CEIL = 8
# ... and INTERNALLY TIGHT: the drop from the cluster to the tail must dominate
# the largest gap *inside* the cluster by this ratio. This is what rejects noise
# (whose largest gap is only ~1.5-2x its neighbours) and is exactly tau-invariant
# (all gaps scale by 1/tau, so the ratio is unchanged).
DOMINANT_GAP_RATIO = 3.0
# Fraction of probes that must be SOUND for the source verdict to be SOUND.
SOURCE_SOUND_FRACTION = 0.5
# tau band over which m* invariance is validated and entropy contrast is shown.
TAU_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0)


def _softmax(scores, tau):
    z = np.asarray(scores, dtype=float) / float(tau)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _entropy_nats(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _cluster(scores):
    """Return (m*, kind) from the sorted-descending score vector.

    Split the vector at its largest gap into a leading cluster s[:m*] and a tail.
    kind is one of:
      "SOUND"     - 2 <= m* <= ceil and the cluster->tail drop dominates the
                    largest gap INSIDE the cluster (internally tight rivals).
      "COLLAPSED" - m* == 1 and the lone leading mode stands dominantly above the
                    tail (the determinism-root shape).
      "FLAT"      - no dominant separation: undifferentiated mass (noise/uniform)
                    or a cluster too broad to be genuine competition.
    Exactly tau-invariant: scaling all scores by 1/tau scales every gap by 1/tau,
    so both the argmax-gap location (m*) and every gap RATIO are unchanged.
    """
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if s.size < 2:
        return s.size, "FLAT"
    gaps = s[:-1] - s[1:]
    if not np.any(gaps > 0):
        return s.size, "FLAT"                  # perfectly flat
    j = int(np.argmax(gaps))                   # cluster = s[:j+1], drop = gaps[j]
    m_star = j + 1
    drop = float(gaps[j])
    if m_star == 1:
        tail = gaps[1:]
        ref = float(np.median(tail[tail > 0])) if np.any(tail > 0) else 0.0
        return m_star, ("COLLAPSED" if (ref == 0.0 or drop >= DOMINANT_GAP_RATIO * ref) else "FLAT")
    within = float(np.max(gaps[:j]))           # largest gap inside the cluster
    tight = (within <= 0.0) or (drop >= DOMINANT_GAP_RATIO * within)
    if tight and m_star <= SOUND_MODE_CEIL:
        return m_star, "SOUND"
    return m_star, "FLAT"


_KIND_TO_VERDICT = {"SOUND": "SOUND", "COLLAPSED": "UNSOUND_COLLAPSED",
                    "FLAT": "UNSOUND_UNIFORM"}


def soundness_for_probe(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size < 2:
        return {"verdict": "UNSCORABLE", "m_star": int(scores.size), "n": int(scores.size)}
    m_star, kind = _cluster(scores)
    verdict = _KIND_TO_VERDICT[kind]
    # tau-invariance validation + entropy contrast across the sweep.
    ms, ents = [], []
    for tau in TAU_SWEEP:
        ms.append(_cluster(scores * (1.0 / tau))[0])
        ents.append(round(_entropy_nats(_softmax(scores, tau)), 4))
    return {"verdict": verdict, "m_star": int(m_star), "n": int(scores.size),
            "m_star_tau_invariant": bool(len(set(ms)) == 1),
            "m_star_over_tau": ms, "entropy_over_tau": ents}


def run(probe_export):
    """probe_export: {"probes": [{"scores": [...], "source": "A1"|"A2", "id": ...}, ...]}
    or a bare list of such probe dicts. Each probe carries the PRE-collapse score
    vector (A2 logits / A1 neg-distances). Returns per-source soundness verdicts.
    """
    probes = probe_export["probes"] if isinstance(probe_export, dict) else probe_export
    by_source = {}
    per_probe = []
    for pr in probes:
        src = pr.get("source", "UNSPECIFIED")
        res = soundness_for_probe(pr["scores"])
        res["source"] = src
        res["id"] = pr.get("id")
        per_probe.append(res)
        by_source.setdefault(src, []).append(res)

    sources = {}
    for src, items in by_source.items():
        scorable = [r for r in items if r["verdict"] != "UNSCORABLE"]
        n_sound = sum(1 for r in scorable if r["verdict"] == "SOUND")
        n = len(scorable)
        frac = (n_sound / n) if n else 0.0
        inv_ok = all(r.get("m_star_tau_invariant", True) for r in scorable)
        sources[src] = {
            "n_probes": len(items), "n_scorable": n, "n_sound": n_sound,
            "sound_fraction": round(frac, 4),
            "mean_m_star": round(float(np.mean([r["m_star"] for r in scorable])), 3) if n else None,
            "tau_invariance_holds": bool(inv_ok),
            "source_verdict": "SOUND" if (n and frac >= SOURCE_SOUND_FRACTION) else "NOT_SOUND",
            "collapsed_fraction": round(sum(1 for r in scorable if r["verdict"] == "UNSOUND_COLLAPSED") / n, 4) if n else None,
            "uniform_fraction": round(sum(1 for r in scorable if r["verdict"] == "UNSOUND_UNIFORM") / n, 4) if n else None,
        }

    # source comparison (A1 vs A2): which carries sound, non-input mode structure.
    ranked = sorted(
        [(s, v["sound_fraction"], v["source_verdict"]) for s, v in sources.items()],
        key=lambda t: t[1], reverse=True,
    )
    return {"n_probes": len(per_probe), "sources": sources,
            "source_ranking_by_sound_fraction": ranked,
            "per_probe": per_probe}


def _self_test():
    rng = np.random.RandomState(0)
    V = 64  # vocab/codebook support per probe

    def sound_probe():
        # two close leading modes, then a clean drop to a low flat tail
        s = rng.randn(V) * 0.3
        order = rng.permutation(V)
        s[order[0]] = 6.0
        s[order[1]] = 5.6
        return s.tolist()

    def deterministic_probe():
        # one dominant mode, everything else low (the determinism-root shape)
        s = rng.randn(V) * 0.3
        s[rng.randint(V)] = 9.0
        return s.tolist()

    def uniform_probe():
        # undifferentiated: tiny noise around a constant, no real cluster
        return (rng.randn(V) * 0.05).tolist()

    export = {"probes":
        [{"scores": sound_probe(), "source": "A2", "id": f"snd{i}"} for i in range(12)]
        + [{"scores": deterministic_probe(), "source": "A2", "id": f"det{i}"} for i in range(4)]
        + [{"scores": uniform_probe(), "source": "A1", "id": f"uni{i}"} for i in range(12)]
        + [{"scores": deterministic_probe(), "source": "A1", "id": f"a1det{i}"} for i in range(4)]
    }
    out = run(export)
    a2, a1 = out["sources"]["A2"], out["sources"]["A1"]
    print("A2 (12 sound + 4 deterministic) ->", a2["source_verdict"],
          "sound_frac", a2["sound_fraction"], "tau_inv", a2["tau_invariance_holds"])
    print("A1 (12 uniform + 4 deterministic) ->", a1["source_verdict"],
          "sound_frac", a1["sound_fraction"], "uniform_frac", a1["uniform_fraction"])
    assert a2["source_verdict"] == "SOUND", a2
    assert a1["source_verdict"] == "NOT_SOUND", a1
    assert a2["tau_invariance_holds"] and a1["tau_invariance_holds"]
    # a single clearly-sound probe must read SOUND and be tau-invariant
    one = soundness_for_probe(sound_probe())
    assert one["verdict"] == "SOUND" and one["m_star_tau_invariant"], one
    # a deterministic probe must read collapsed, NOT uniform
    det = soundness_for_probe(deterministic_probe())
    assert det["verdict"] == "UNSOUND_COLLAPSED", det
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            export = json.load(f)
        print(json.dumps(run(export), indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
