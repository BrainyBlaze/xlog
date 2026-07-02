#!/usr/bin/env python3
"""xlog per-clone graded-marginal readout PRIMITIVE (engine lane, @xlog-claude).

Stable entry point for the finalization coupling: DTS orchestration (@xlog-claude-2)
assembles per-clone program sources (anchor@resolution + neighbours + rules) and calls
this to get graded consequence marginals. Kept minimal and pure so orchestration can
evolve independently.

Readout mechanism: exact d-DNNF WMC marginal as a HOST FLOAT via the already-exposed
loss surface — P(query) = exp(-nll_loss(query)). No DLPack capsule, no torch/cupy, no
rebuild. Facts are `p::atom` (or hard facts); queries are ground atom strings; no `?-`
directive needed.

Scope / staged extensions:
  * shared-key joins (body vars all present in head)  -> SUPPORTED.
  * existential joins (a body var absent from head)   -> SUPPORTED on this soft-WMC
    path with ground queries (verified: chain-join h(a,c):-e1(a,Y),e2(Y,c) gives
    P=0.5*0.6=0.30; OR-aggregation over bindings gives noisy-OR). The "not yet
    supported" limit is the SEPARATE hard/Boolean epistemic query surface
    (LogicProgram.evaluate() membership), which this primitive does NOT use.
  * return_grads (dP/dp_var for trainable coupling)    -> compiled-once-grad-substrate,
    item (b); stubbed here, NotImplementedError until the grad path is exposed.
"""
from __future__ import annotations

import math

import pyxlog


def readout_marginals(
    program_src: str,
    queries: list[str],
    *,
    return_grads: bool = False,
) -> dict[str, float]:
    """{query: P(query)} exact-WMC host-float marginals for a single program.

    program_src: facts (`0.54 :: e(1,2).`) + rules (`h(X,Y) :- e(X,Y).`).
    queries: ground atom strings, e.g. ["h(1, 2)"].
    return_grads: reserved for item (b) compiled-once-grad-substrate — not yet wired.
    """
    if return_grads:
        raise NotImplementedError(
            "return_grads (compiled-once-grad-substrate, item b) not yet exposed; "
            "read-only marginals only for now"
        )
    prog = pyxlog.Program.compile(program_src)
    return {q: math.exp(-prog.nll_loss(q)) for q in queries}


def per_clone_marginals(
    pro_src: str,
    contra_src: str,
    queries: list[str],
) -> dict[str, dict[str, float]]:
    """Per-clone graded divergence: {query: {pro, contra, delta}}.

    pro_src / contra_src are the two clone program sources the orchestration built
    (contested anchor resolved pro vs contra, same neighbours + rules).
    """
    pro = readout_marginals(pro_src, queries)
    contra = readout_marginals(contra_src, queries)
    return {
        q: {"pro": pro[q], "contra": contra[q], "delta": abs(pro[q] - contra[q])}
        for q in queries
    }


def _self_test() -> None:
    # single marginal (shared-key conjunction)
    m = readout_marginals(
        "0.542 :: e(1,2).\n0.7 :: f(1,2).\ng(X,Y) :- e(X,Y), f(X,Y).", ["g(1, 2)"]
    )
    assert abs(m["g(1, 2)"] - 0.542 * 0.7) < 1e-3, m

    # per-clone graded divergence (fusion): pro asserts contested anchor, contra resolves it out
    pc = per_clone_marginals(
        pro_src="0.542 :: e(1,2).\n0.7 :: f(1,2).\ng(X,Y) :- e(X,Y), f(X,Y).",
        contra_src="0.7 :: f(1,2).\ng(X,Y) :- e(X,Y), f(X,Y).",
        queries=["g(1, 2)"],
    )
    r = pc["g(1, 2)"]
    # NOTE: a withdrawn premise gives P floored at ~1e-38 (engine keeps nll_loss=-log P
    # finite), not exactly 0.0 — orchestration should treat sub-epsilon marginals as zero.
    assert abs(r["pro"] - 0.3794) < 1e-3 and r["contra"] < 1e-6, pc

    # existential join (body var Y absent from head) — SUPPORTED on soft-WMC path
    ex = readout_marginals(
        "0.5 :: e1(a,b).\n0.6 :: e2(b,c).\nh(X,Z) :- e1(X,Y), e2(Y,Z).", ["h(a, c)"]
    )
    assert abs(ex["h(a, c)"] - 0.30) < 1e-3, ex

    print("self-test PASS:", pc, "| existential h(a,c)=%.4f" % ex["h(a, c)"])


if __name__ == "__main__":
    _self_test()
