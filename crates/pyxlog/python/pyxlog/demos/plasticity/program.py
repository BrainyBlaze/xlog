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


def build_source(split: Split) -> str:
    pre_facts = " ".join(f"edge_pre_post({i})." for i in split.relational_pre_post_ids())
    post_facts = " ".join(f"edge_post_pre({i})." for i in split.relational_post_pre_ids())
    return f"""
        {pre_facts}
        {post_facts}
        pred edge_pre_post(i64). pred edge_post_pre(i64). pred {TRAIN_HEAD}(i64).
        trainable_rule({CAND_PREPOST_REL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_pre_post(E).
        trainable_rule({CAND_PREPOST_NEURAL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_pre_post(E).
        trainable_rule({CAND_POSTPRE_NEURAL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_post_pre(E).
        train({TRAIN_HEAD}, binary_cross_entropy).
    """


def build_neural_bodies(split: Split) -> dict[str, Any]:
    phi = split.phi()
    return {
        CAND_PREPOST_NEURAL: NeuralBodySpec(features=phi, threshold=0.5),
        CAND_POSTPRE_NEURAL: NeuralBodySpec(features=phi, threshold=0.5),
    }
