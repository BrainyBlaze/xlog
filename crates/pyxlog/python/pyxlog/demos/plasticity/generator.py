"""Synthetic STDP plasticity data with a planted ground-truth rule.

Ground truth (planted): an edge ``strengthens`` iff it has a pre-before-post
coincidence AND its saliency >= SALIENCY_THRESHOLD. The existential event->edge
aggregation is already projected here into head-bound relations
(``pre_post``/``post_pre`` membership per edge), because existential-join
trainable bodies are not expressible on the current engine (Path Y).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

SALIENCY_THRESHOLD = 0.5


@dataclass(frozen=True)
class EdgeSample:
    entity_id: str  # globally unique, stable identity (NOT the binding index)
    pre_post: bool  # has a pre-before-post coincidence (existential projection)
    post_pre: bool  # has a post-before-pre coincidence
    saliency: float  # s in [0,1]; phi column 0
    distractor: float  # phi column 1; a feature the gate must ignore


def strengthens(sample: EdgeSample) -> bool:
    """The planted ground-truth plasticity rule (LTP)."""
    return sample.pre_post and sample.saliency >= SALIENCY_THRESHOLD


def weakens(sample: EdgeSample) -> bool:
    """The planted ground-truth rule for the opposite direction (LTD): a
    post-before-pre coincidence with high saliency."""
    return sample.post_pre and sample.saliency >= SALIENCY_THRESHOLD


@dataclass
class Split:
    """One data split. The binding index of a sample is its position in
    ``samples`` (the xlog query key ``train_head(i)`` and the phi row index)."""

    samples: list[EdgeSample]

    def num_queries(self) -> int:
        return len(self.samples)

    def relational_pre_post_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.samples) if s.pre_post]

    def relational_post_pre_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.samples) if s.post_pre]

    def labels(self) -> list[bool]:
        return [strengthens(s) for s in self.samples]

    def labels_for(self, outcome: str) -> list[bool]:
        if outcome == "strengthens":
            return [strengthens(s) for s in self.samples]
        if outcome == "weakens":
            return [weakens(s) for s in self.samples]
        raise ValueError(f"unknown outcome: {outcome!r}")

    def entity_ids(self) -> set[str]:
        return {s.entity_id for s in self.samples}

    def phi(self) -> Any:
        import torch

        return torch.tensor(
            [[s.saliency, s.distractor] for s in self.samples], dtype=torch.float32
        )


def make_fixed_split(prefix: str) -> Split:
    """A small, deterministic split covering every discriminating case, so the
    recovery test is reproducible and the correct candidate provably wins."""
    rows = [
        # (pre_post, post_pre, saliency, distractor)
        (True, False, 0.90, 0.10),  # 0  strong LTP        -> label TRUE
        (True, False, 0.80, 0.90),  # 1  strong LTP        -> label TRUE
        (True, False, 0.20, 0.80),  # 2  weak pre-post     -> FALSE (relational-only over-fires)
        (True, False, 0.10, 0.20),  # 3  weak pre-post     -> FALSE
        (False, True, 0.90, 0.10),  # 4  post-pre, high s  -> FALSE (wrong timing)
        (False, True, 0.30, 0.50),  # 5  post-pre          -> FALSE
        (False, False, 0.70, 0.40),  # 6 neither           -> FALSE
        (False, False, 0.10, 0.10),  # 7 neither           -> FALSE
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_held_out_split(prefix: str) -> Split:
    """Disjoint held-out entities exercising generalization and vigilance."""
    rows = [
        (True, False, 0.95, 0.20),  # 0  NEW strong LTP     -> should FIRE (generalize)
        (True, False, 0.15, 0.90),  # 1  NEW weak pre-post  -> should NOT fire (vigilance)
        (False, True, 0.90, 0.10),  # 2  NEW post-pre       -> should NOT fire (wrong timing)
        (False, False, 0.60, 0.30),  # 3 NEW neither        -> should NOT fire
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_random_split(prefix: str, n: int, seed: int) -> Split:
    """A seeded larger split for the runnable demo at scale."""
    rng = random.Random(seed)
    samples: list[EdgeSample] = []
    for i in range(n):
        pre = rng.random() < 0.5
        post = (not pre) and rng.random() < 0.5
        saliency = round(rng.random(), 3)
        distractor = round(rng.random(), 3)
        samples.append(EdgeSample(f"{prefix}_{i}", pre, post, saliency, distractor))
    return Split(samples)


def make_demo_data() -> tuple[Split, Split]:
    """The canonical (train, held_out) pair with disjoint entity ids."""
    return make_fixed_split("e_tr"), make_held_out_split("e_ho")


def make_weakens_split(prefix: str) -> Split:
    """LTD analogue of make_fixed_split: post-before-pre is the primary timing, so
    the planted weakens rule (post_pre AND saliency>=0.5) has discriminating cases."""
    rows = [
        # (pre_post, post_pre, saliency, distractor)
        (False, True, 0.90, 0.10),  # 0  strong LTD        -> label TRUE
        (False, True, 0.80, 0.90),  # 1  strong LTD        -> label TRUE
        (False, True, 0.20, 0.80),  # 2  weak post-pre     -> FALSE (relational-only over-fires)
        (False, True, 0.10, 0.20),  # 3  weak post-pre     -> FALSE
        (True, False, 0.90, 0.10),  # 4  pre-post, high s  -> FALSE (wrong timing)
        (True, False, 0.30, 0.50),  # 5  pre-post          -> FALSE
        (False, False, 0.70, 0.40),  # 6 neither           -> FALSE
        (False, False, 0.10, 0.10),  # 7 neither           -> FALSE
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_weakens_held_out_split(prefix: str) -> Split:
    """Disjoint held-out LTD entities for generalization and vigilance."""
    rows = [
        (False, True, 0.95, 0.20),  # 0  NEW strong LTD     -> should FIRE (generalize)
        (False, True, 0.15, 0.90),  # 1  NEW weak post-pre  -> should NOT fire (vigilance)
        (True, False, 0.90, 0.10),  # 2  NEW pre-post       -> should NOT fire (wrong timing)
        (False, False, 0.60, 0.30),  # 3 NEW neither        -> should NOT fire
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_weakens_demo_data() -> tuple[Split, Split]:
    """The canonical LTD (train, held_out) pair with disjoint entity ids."""
    return make_weakens_split("w_tr"), make_weakens_held_out_split("w_ho")
