"""Plasticity & Saliency Rule Induction demo (Path Y, on the current engine)."""

from .demo import DemoReport, GROUND_TRUTH_RULE, run_demo
from .generator import (
    EdgeSample,
    Split,
    make_demo_data,
    make_fixed_split,
    make_held_out_split,
    make_random_split,
    strengthens,
)
from .program import CAND_PREPOST_NEURAL, CAND_PREPOST_REL, CAND_POSTPRE_NEURAL, TRAIN_HEAD

__all__ = [
    "DemoReport",
    "GROUND_TRUTH_RULE",
    "run_demo",
    "EdgeSample",
    "Split",
    "make_demo_data",
    "make_fixed_split",
    "make_held_out_split",
    "make_random_split",
    "strengthens",
    "CAND_PREPOST_NEURAL",
    "CAND_PREPOST_REL",
    "CAND_POSTPRE_NEURAL",
    "TRAIN_HEAD",
]
