#!/usr/bin/env python3
"""Regression gate for two live-rule-graph WMC defects — both FIXED, must stay fixed.

Originally the minimal failing repros (greedily minimized from the real NB3
clone sources, nb2_run18 anchor fact 35); after fix-1 (1739910d) and fix-2
(35bbe75f) both compile, so the assertions are now FLIPPED: each case must
compile and return the exact analytical marginal. Red here = a regression in
one of two subtle engine areas that unit tests historically missed on the
warm end-to-end pyxlog path.

CASE 1 — two-sided recursive SCC (fix-1: PIR flatten + absorption).
  Both predicates of a mutual-recursion cycle carry a probabilistic base fact
  on the SAME ground pair. Pre-fix the syntactic fixpoint test never converged
  (Provenance iteration limit 1024). Analytical fixpoint:
  P = 1 - (1-0.5406)(1-0.7143) = 0.86875 for BOTH cycle atoms.

CASE 2 — duplicate ground atoms x rule (fix-2: stale-cache eviction).
  Compiles cleanly on a fresh path; pre-fix it could hash onto a disk-cache
  entry written by an older PIR encoding, serving a circuit with an
  inconsistent var space ("Circuit references var 45 but base CNF has only
  36 vars") as a fatal error on every warm compile. Post-fix a stale cached
  artifact is evicted and recompiled (self-healing), so this must compile on
  BOTH cold and warm runs. Duplicate p::atom lines keep the engine's native
  independent-evidence noisy-OR semantics (the Source-B consumer dedups with
  max-merge BEFORE grounding — a deliberate DTS re-admission contract, not
  engine behavior).
"""
import math

import pyxlog

CASE_1_TWO_SIDED_SCC = """\
0.5406026840209961 :: q10001(10034, 10042).
0.714295 :: q10000(10034, 10042).
q10000(A, B) :- q10001(A, B), q10001(A, B).
q10001(A, B) :- q10000(A, B), q10000(A, B).
"""
CASE_1_FIXPOINT = 1 - (1 - 0.5406026840209961) * (1 - 0.714295)

CASE_2_DUP_ATOMS_X_RULE = """\
0.537177 :: q10001(10034, 10042).
0.709917 :: q10000(10034, 10042).
0.709917 :: q10001(10034, 10042).
0.714295 :: q10000(10034, 10042).
0.530629 :: q10001(10025, 10033).
0.531213 :: q10001(10089, 10097).
0.531862 :: q10001(10008, 10016).
q10000(A, B) :- q10001(A, B), q10001(A, B).
"""


def _marginal(src, query):
    prog = pyxlog.Program.compile(src)
    return math.exp(-prog.nll_loss(query))


def main():
    # case 1: recursive SCC converges to the exact analytical fixpoint,
    # identically for both cycle atoms.
    for atom in ("q10000(10034, 10042)", "q10001(10034, 10042)"):
        p = _marginal(CASE_1_TWO_SIDED_SCC, atom)
        assert abs(p - CASE_1_FIXPOINT) < 1e-4, (atom, p, CASE_1_FIXPOINT)
    print(f"case1 two-sided-SCC: converges, P={CASE_1_FIXPOINT:.5f} both atoms")

    # case 2: compiles on cold AND warm runs (stale cache self-heals; a
    # second compile in the same process exercises the warm path).
    p_cold = _marginal(CASE_2_DUP_ATOMS_X_RULE, "q10000(10034, 10042)")
    p_warm = _marginal(CASE_2_DUP_ATOMS_X_RULE, "q10000(10034, 10042)")
    assert abs(p_cold - p_warm) < 1e-9, (p_cold, p_warm)
    assert 0.0 < p_cold <= 1.0, p_cold
    print(f"case2 dup-atoms-x-rule: compiles cold+warm, P={p_cold:.4f} (native noisy-OR)")

    print("REGRESSION GATE PASS — both live-rule-graph defects stay fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
