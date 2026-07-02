#!/usr/bin/env python3
"""Graded WMC marginal readout on grounded NB2 contestation — HOST SCALAR, no torch/cupy/rebuild.

The graded probabilistic axis (uncertainty carried THROUGH logic, not collapsed by argmax).
The earlier "torch/DLPack blocker" was an artifact of reading the device-resident capsule from
`Program.compile(...).evaluate().prob`. The exact-WMC marginal is also reachable as a plain host
float via the already-exposed loss surface: `nll_loss(query) == -log P(query)`, so
`P(query) = exp(-nll_loss(query))`. No CUDA-DLPack consumer, no rebuild.

Demonstrates:
  (1) real NB2 contested fact (10001(10050,10058), pro=0.542) graded marginal carried through
      rule 0 (derived <- premise): P=0.542 preserved vs argmax-collapse 1.0 (gap = discarded uncertainty);
  (2) noisy-OR two-support: P = 1-(1-p1)(1-p2) — proves REAL WMC, not an echo of one input;
  (3) conjunction (shared key, no existential): P = p1*p2.

Requires pyxlog GPU (host-io is a default feature; nll_loss uses the exact d-DNNF WMC path).
"""
import math
import pyxlog


def marginal(program_src: str, query: str) -> float:
    """Exact WMC marginal P(query) as a host float, via nll_loss = -log P."""
    prog = pyxlog.Program.compile(program_src + f"\n?- {query}.\n")
    return math.exp(-prog.nll_loss(query))


def main() -> None:
    # (1) grounded NB2 contested fact -> graded marginal through the derivation rule
    p = marginal("0.542 :: e(10050, 10058).\nh(X, Y) :- e(X, Y).", "h(10050, 10058)")
    print(f"NB2 fact_id=50  10001(10050,10058) pro=0.542  -> graded P(derived) = {p:.4f}")
    print(f"  argmax-collapse = 1.0  => uncertainty PRESERVED; gap {1.0 - p:.4f} = what DTS argmax discards")
    assert abs(p - 0.542) < 1e-3, p

    # (2) noisy-OR two-support (real WMC)
    p_nor = marginal("0.542 :: a().\n0.55 :: b().\nh() :- a().\nh() :- b().", "h()")
    exp_nor = 1 - (1 - 0.542) * (1 - 0.55)
    print(f"noisy-OR two-support: P(h) = {p_nor:.4f}  (analytic {exp_nor:.4f})")
    assert abs(p_nor - exp_nor) < 1e-3, (p_nor, exp_nor)

    # (3) conjunction, shared key (no existential join)
    p_and = marginal("0.542 :: e(1,2).\n0.7 :: f(1,2).\ng(X,Y) :- e(X,Y), f(X,Y).", "g(1, 2)")
    print(f"conjunction (shared key): P(g) = {p_and:.4f}  (analytic {0.542 * 0.7:.4f})")
    assert abs(p_and - 0.542 * 0.7) < 1e-3, p_and

    print("\nGRADED WMC readout WORKS via nll_loss (host scalar) — engine carries grounded "
          "uncertainty through logic; no torch/cupy/rebuild needed.")


if __name__ == "__main__":
    main()
