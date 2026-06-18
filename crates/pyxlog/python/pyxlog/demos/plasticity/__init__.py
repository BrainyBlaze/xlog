"""Plasticity & Saliency Rule Induction demo (Path Y, on the current engine)."""

from .generator import (
    EdgeSample,
    Split,
    make_demo_data,
    make_fixed_split,
    make_held_out_split,
    make_random_split,
    strengthens,
)

__all__ = [
    "EdgeSample",
    "Split",
    "make_demo_data",
    "make_fixed_split",
    "make_held_out_split",
    "make_random_split",
    "strengthens",
]
