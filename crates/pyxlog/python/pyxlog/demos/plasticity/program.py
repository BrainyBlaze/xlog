"""Builds the xlog candidate program and neural-body spec map from a Split.

Three same-head candidates compete for ``strengthens(Edge)``:
  - CAND_PREPOST_REL    : relational-only pre-post  (over-fires on weak coincidences)
  - CAND_PREPOST_NEURAL : pre-post AND a learned saliency gate  (the TRUE rule)
  - CAND_POSTPRE_NEURAL : post-pre AND a learned gate  (wrong-timing distractor)
"""

from __future__ import annotations

from typing import Any

from pyxlog.ilp.neurosymbolic import NeuralBodySpec

from .generator import Split

TRAIN_HEAD = "strengthens"
CAND_PREPOST_REL = "cand_prepost_rel"
CAND_PREPOST_NEURAL = "cand_prepost_neural"
CAND_POSTPRE_NEURAL = "cand_postpre_neural"
CAND_POSTPRE_REL = "cand_postpre_rel"


class _OutcomeSpec:
    """The candidate layout for one plasticity direction: which relation is the
    'correct' timing, the relational-only over-firer, the true neural candidate,
    and the wrong-timing neural distractor."""

    def __init__(self, head: str, rel_only: str, correct_neural: str, distractor_neural: str,
                 correct_rel: str, distractor_rel: str) -> None:
        self.head = head
        self.rel_only = rel_only
        self.correct_neural = correct_neural
        self.distractor_neural = distractor_neural
        self.correct_rel = correct_rel
        self.distractor_rel = distractor_rel


_OUTCOMES = {
    "strengthens": _OutcomeSpec(
        head="strengthens", rel_only=CAND_PREPOST_REL, correct_neural=CAND_PREPOST_NEURAL,
        distractor_neural=CAND_POSTPRE_NEURAL, correct_rel="edge_pre_post", distractor_rel="edge_post_pre",
    ),
    "weakens": _OutcomeSpec(
        head="weakens", rel_only=CAND_POSTPRE_REL, correct_neural=CAND_POSTPRE_NEURAL,
        distractor_neural=CAND_PREPOST_NEURAL, correct_rel="edge_post_pre", distractor_rel="edge_pre_post",
    ),
}


def outcome_spec(outcome: str = "strengthens") -> "_OutcomeSpec":
    try:
        return _OUTCOMES[outcome]
    except KeyError:
        raise ValueError(f"unknown outcome: {outcome!r}") from None


def build_source(split: Split, outcome: str = "strengthens") -> str:
    spec = outcome_spec(outcome)
    pre_facts = " ".join(f"edge_pre_post({i})." for i in split.relational_pre_post_ids())
    post_facts = " ".join(f"edge_post_pre({i})." for i in split.relational_post_pre_ids())
    return f"""
        {pre_facts}
        {post_facts}
        pred edge_pre_post(i64). pred edge_post_pre(i64). pred {spec.head}(i64).
        trainable_rule({spec.rel_only}, weight=0.0) :: {spec.head}(E) :- {spec.correct_rel}(E).
        trainable_rule({spec.correct_neural}, weight=0.0) :: {spec.head}(E) :- {spec.correct_rel}(E).
        trainable_rule({spec.distractor_neural}, weight=0.0) :: {spec.head}(E) :- {spec.distractor_rel}(E).
        train({spec.head}, binary_cross_entropy).
    """


def build_neural_bodies(split: Split, outcome: str = "strengthens") -> dict[str, Any]:
    spec = outcome_spec(outcome)
    phi = split.phi()
    return {
        spec.correct_neural: NeuralBodySpec(features=phi, threshold=0.5),
        spec.distractor_neural: NeuralBodySpec(features=phi, threshold=0.5),
    }
