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
