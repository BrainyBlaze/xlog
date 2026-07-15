"""Engine-mode training: a real-valued credit over the ENGINE's own candidate space.

The dILP enumerator (`valid_candidates`) already holds a neural-bodied existential
candidate and derives with it (proven by the bridge spike); what the engine's credit
kernel cannot do is carry a per-event probability -- its CSR is binary. This module
re-implements the credit NLL as a torch graph so the neural column enters as a
noisy-OR over the ENGINE-read join extension, and -- separately -- selects the rule
by K-FOLD HOLDOUT with a fit gate, never by the training weight: training credit
cannot distinguish a crisp-but-coincidental rule from a soft-but-correct one even in
principle; generalization can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyxlog.ilp.join_bodies import (
    JoinExtensionIndex,
    noisy_or_from_index,
    prepare_extension,
)


@dataclass(frozen=True)
class CandidateSpec:
    """One engine-enumerated candidate, ready for the torch credit.

    Exactly one of ``witness_index`` / ``binary_cover`` is set: a neural candidate
    scores each fact as a noisy-OR over its witnesses, a relational candidate is a
    fixed {0,1} cover. Anything else is a construction bug, checked at init.
    """

    cid: int
    left: str
    right: str
    is_neural: bool
    witness_index: JoinExtensionIndex | None
    binary_cover: Any | None

    def __post_init__(self) -> None:
        if self.is_neural != (self.witness_index is not None) or (
            self.is_neural == (self.binary_cover is not None)
        ):
            raise ValueError(
                f"candidate {self.cid} ({self.left},{self.right}): a neural candidate "
                "carries witness_index and no binary_cover; a relational one the "
                "opposite. Mixed or missing is a construction bug, refused."
            )


def credit_nll(cand_probs, specs, p_event, is_positive, gamma: float = 1.0):
    """``credit[f] = sum_c p_c * s_c(f)``, NLL over facts. A torch graph end to end,
    so the gradient reaches BOTH the candidate logits and the network behind
    ``p_event``. ``gamma`` sharpens only the neural score (calibration against
    gradient starvation next to crisp {0,1} covers; it never decides truth --
    holdout does)."""
    import torch

    credit = None
    for spec in specs:
        if spec.is_neural:
            s = noisy_or_from_index(p_event, spec.witness_index)
            if gamma != 1.0:
                s = s.clamp(1e-7, 1.0) ** gamma
        else:
            s = spec.binary_cover
        term = cand_probs[spec.cid] * s
        credit = term if credit is None else credit + term
    credit = credit.clamp(1e-8, 1 - 1e-8)
    pos = is_positive.to(credit.dtype)
    loss = -(pos * torch.log(credit) + (1 - pos) * torch.log(1 - credit))
    return loss.mean()


def enumerate_specs(prog, mask_name, facts, neural_relations, device):
    """One CandidateSpec per engine triple over the program's binary EDB relations.

    Witnesses come from the ENGINE (`relation_facts`), never from the caller: for a
    fact (h, y) and candidate (L, R) the witness set is {z : L(h, z)} scored by the
    network at label y for a neural R, and the binary cover is
    [exists z: L(h,z) and R(z,y)] for a relational R. A neural relation in the LEFT
    slot has no witness semantics in this credit and is SKIPPED — filtering an
    auto-enumerated pool is not the same as silently altering a user-declared rule;
    the engine's cross-product enumeration always contains such triples."""
    import torch

    left_ext: dict[str, dict[int, list[int]]] = {}
    right_pairs: dict[str, set[tuple[int, int]]] = {}

    def _left(name):
        if name not in left_ext:
            buckets: dict[int, list[int]] = {}
            for row in prog.relation_facts(name):
                buckets.setdefault(int(row[0]), []).append(int(row[1]))
            left_ext[name] = buckets
        return left_ext[name]

    def _pairs(name):
        if name not in right_pairs:
            right_pairs[name] = {
                (int(r[0]), int(r[1])) for r in prog.relation_facts(name)
            }
        return right_pairs[name]

    specs: list[CandidateSpec] = []
    for cand in prog.valid_candidates(mask_name):
        ln, rn = cand["left_name"], cand["right_name"]
        if ln.startswith("__xlog_") or rn.startswith("__xlog_"):
            continue                        # meta relations: arity-incompatible, skip
        if ln in neural_relations:
            continue                        # neural-in-left: no witness semantics, skip
        if rn in neural_relations:
            witnesses = [_left(ln).get(h, []) for h, _y in facts]
            idx = prepare_extension(
                witnesses, device, num_rows=neural_relations[rn]
            )
            specs.append(CandidateSpec(cand["id"], ln, rn, True, idx, None))
        else:
            pairs = _pairs(rn)
            lext = _left(ln)
            cover = torch.tensor(
                [1.0 if any((z, y) in pairs for z in lext.get(h, [])) else 0.0
                 for h, y in facts],
                device=device,
            )
            specs.append(CandidateSpec(cand["id"], ln, rn, False, None, cover))
    return specs
