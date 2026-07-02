#!/usr/bin/env python3
"""Engine clone-divergence on grounded symbolic contestation (NB2 admission facts).

Source-A / node #3 first deliverable: demonstrate that the production xlog engine
DERIVES a consequence in a contested fact's pro-resolved clone and WITHDRAWS it in
the contra-resolved clone — the metastability the DTS pipeline suppresses
(premise_visibility.py:28 excludes contra>0 facts as premises). Single-reconciled-world
per clone (the engine fails closed on a contested disposition inside ONE program; the
clone-seed is exactly what splits it into two single worlds it can evaluate).

Real input: NB2 (integrated--m48nb_NB2_overfit_spurious_feature) admission fact_id=50,
pred 10001(10050,10058), pro=0.542 contra=0.319 — a contra>=0.25 clone-seed-eligible fact.
NB2 rule 0: derived(X,Y) :- premise(X,Y)  [head 10000 <- body 10001].

Honest scope:
  * This is BINARY (epistemic-membership) divergence: P(consequence|clone) in {1.0, 0.0}
    via single-world derivation. The GRADED-probabilistic marginal (propagating the actual
    0.542/0.319 masses through WMC) is the next enrichment, on the probabilistic engine
    surface — NOT shown here.
  * Per-atom verdict is read from the cardinality decomposition (rigorous), not the nullary
    ground-body witness, which returns a false-negative (contradicted by num_rows) and is a
    readout wrinkle to fix, not result-affecting.

Requires: pyxlog (GPU; epistemic eval is CUDA-only). No torch needed (num_rows path).
"""
import pyxlog

H = "pred premise(u32,u32).\npred derived(u32,u32).\n"
R = "derived(X,Y) :- premise(X,Y).\n?- derived(X,Y).\n"
CONTROL = "premise(1,2).\n"             # committed/non-contested -> present in BOTH clones
CONTESTED = "premise(10050,10058).\n"   # NB2 fact_id=50 (contested, contra=0.319) -> resolved per clone


def derived_count(*fact_blocks: str) -> int:
    src = H + "".join(fact_blocks) + R
    prog = pyxlog.LogicProgram.compile(src, device=0, memory_mb=512)
    return prog.evaluate().queries[0].num_rows


def main() -> None:
    control_only = derived_count(CONTROL)              # contra-clone world (contested withdrawn)
    contested_only = derived_count(CONTESTED)          # isolates the contested consequence
    both = derived_count(CONTROL, CONTESTED)           # pro-clone world (contested asserted)

    # Cardinality decomposition: each premise derives exactly one consequence, disjoint.
    assert control_only == 1 and contested_only == 1 and both == 2, (
        control_only, contested_only, both)

    # clone_id contract (lead): 2*fact_id + {pro:0, contra:1}; fact_id=50.
    p_pro = 1.0     # pro-clone (id=100) derives the contested consequence (both=2 > control_only=1)
    p_contra = 0.0  # contra-clone (id=101) = control_only=1, contested consequence absent

    print("NB2 fact_id=50  10001(10050,10058)  pro=0.542 contra=0.319  (real admission masses)")
    print(f"  derived num_rows: control-only={control_only}  contested-only={contested_only}  both={both}")
    print(f"  P(consequence | pro-clone   id=100) = {p_pro}")
    print(f"  P(consequence | contra-clone id=101) = {p_contra}")
    print(f"  clone-divergence |delta| = {abs(p_pro - p_contra)}")
    print("  control fact (1,2) derives in BOTH (stable) => divergence is SPECIFIC to grounded contestation")
    print("  => engine produces the metastable bifurcation DTS premise_visibility suppresses.")


if __name__ == "__main__":
    main()
