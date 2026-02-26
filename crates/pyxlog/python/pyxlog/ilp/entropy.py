"""Entropy regularization for dILP training.

See docs/plans/2026-02-26-dilp-hardening-design.md §2.2.
"""
from __future__ import annotations

import torch


def normalized_entropy(probs: torch.Tensor, C: int) -> torch.Tensor:
    """Normalized Shannon entropy H/log(C).

    Returns 1.0 for uniform distribution, 0.0 for one-hot.
    Numerically stable via clamped log.

    Parameters
    ----------
    probs : torch.Tensor
        Probability distribution over C candidates (sums to 1).
    C : int
        Number of candidates (used for normalization).

    Returns
    -------
    torch.Tensor
        Scalar in [0, 1].
    """
    log_probs = torch.log(probs.clamp(min=1e-8))
    H = -(probs * log_probs).sum()
    return H / torch.log(torch.tensor(float(C), device=probs.device))


def entropy_weight_at_step(
    step: int, total_steps: int, start: float, end: float
) -> float:
    """Linear decay of entropy weight from start to end over total_steps.

    Parameters
    ----------
    step : int
        Current training step.
    total_steps : int
        Total step budget.
    start : float
        Initial entropy weight.
    end : float
        Final entropy weight.

    Returns
    -------
    float
        Interpolated weight at the given step.
    """
    if total_steps <= 0:
        return end
    t = min(step / total_steps, 1.0)
    return start + t * (end - start)
