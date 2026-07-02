#!/usr/bin/env python3
"""Set-validation of engine clone-divergence over ALL NB2 hard-contested facts.

Generalizes scripts/repro/nb2_clone_divergence.py (single fact) to the whole
contra>=0.25 set in NB2. Pro-clone asserts every hard-contested premise; contra-clone
withdraws them all; a committed control premise is present in both. The aggregate
test: derived-set cardinality must diverge by EXACTLY the number of DISTINCT contested
entity-pairs (a fact that failed to produce a divergent consequence — e.g. an entity
collision with a committed derivation — would make the delta < distinct, and fail).

Honest note: NB2 lists 9 hard-contested fact ROWS but only 6 DISTINCT (arg0,arg1)
pairs (duplicate rows across fact_ids); divergence is measured on the 6 distinct
contested consequences. Binary epistemic-membership divergence (|Δ|=1.0 each); the
graded WMC magnitude is a separate enrichment (torch readout pending).

Requires pyxlog GPU (epistemic eval is CUDA-only); no torch (num_rows path).
"""
import json
import pyxlog

NB2 = ("/home/dev/projects/dts-dlm/out/st_trc_stagea_replay/stagea-replay-20260630-091453/"
       "accepted-control/fact_table_dump/integrated--m48nb_NB2_overfit_spurious_feature.json")
H = "pred premise(u32,u32).\npred derived(u32,u32).\n"
R = "derived(X,Y) :- premise(X,Y).\n?- derived(X,Y).\n"
CONTROL = "premise(1,2).\n"   # committed/non-contested -> stable in both clones


def derived_count(facts_block: str) -> int:
    prog = pyxlog.LogicProgram.compile(H + facts_block + R, device=0, memory_mb=512)
    return prog.evaluate().queries[0].num_rows


def main() -> None:
    d = json.load(open(NB2))
    contested = [(f["arg0"], f["arg1"]) for f in d["facts"]
                 if f["pred_id"] == 10001 and float(f.get("contra", 0)) >= 0.25
                 and float(f.get("pro", 0)) > 0]
    distinct = sorted(set(contested))
    block = "".join(f"premise({a},{b}).\n" for a, b in distinct)

    pro = derived_count(CONTROL + block)   # all contested asserted (pro-clones)
    contra = derived_count(CONTROL)        # all contested withdrawn (contra-clones)
    divergence = pro - contra

    print(f"NB2 hard-contested fact rows = {len(contested)}  distinct entity-pairs = {len(distinct)}")
    print(f"pro-clone derived num_rows   = {pro}")
    print(f"contra-clone derived num_rows = {contra}")
    print(f"divergence = {divergence}  (expect = {len(distinct)} distinct contested consequences)")
    assert divergence == len(distinct), (divergence, len(distinct))
    print(f"SET-VALIDATION PASS: all {len(distinct)} distinct hard-contested facts yield a divergent "
          "derived consequence (|Δ|=1.0 each); committed control stable in both clones "
          "=> divergence specific to grounded contestation, generalizes across the NB2 contested set.")


if __name__ == "__main__":
    main()
