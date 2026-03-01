"""Holdout scoring strategies for dILP.

LOO (leave-one-out) for <= 20 examples.
"""
from __future__ import annotations

import pyxlog
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import TrainConfig


def loo_holdout_f1(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> float | None:
    """Leave-one-out cross-validation F1 over positives.

    For each positive example, train on all-but-one, evaluate the committed
    rule on the held-out example. Computes per-fold precision (against known
    positives+negatives) and recall (held-out derived?), then returns the
    mean F1 across folds.

    Returns None if |positives| < 2.
    """
    if len(positives) < 2:
        return None

    fold_f1s: list[float] = []
    pos_set = {(r, tuple(v)) for r, v in positives}
    neg_set = {(r, tuple(v)) for r, v in negatives}

    for i in range(len(positives)):
        train_pos = positives[:i] + positives[i + 1:]
        held_out = positives[i]

        result = train_only(source, mask_name, train_pos, negatives, config)
        if not (result.converged and result.discovered_rule):
            fold_f1s.append(0.0)
            continue

        trial_source = _commit_rule(source, mask_name, result.discovered_rule)
        try:
            trial = pyxlog.IlpProgramFactory.compile(
                trial_source, device=config.device, memory_mb=config.memory_mb,
            )
            trial.evaluate()
        except Exception:
            fold_f1s.append(0.0)
            continue

        # Recall: did we derive the held-out example?
        held_out_derived = trial.fact_exists(held_out[0], held_out[1])
        recall = 1.0 if held_out_derived else 0.0

        # Precision: of all derived head facts, how many are in the positive set?
        head_rel = held_out[0]
        all_derived = trial.relation_facts(head_rel)
        if all_derived:
            true_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in pos_set)
            false_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in neg_set)
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1.0
        else:
            precision = 1.0 if recall == 0.0 else 0.0

        # F1 = harmonic mean
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        fold_f1s.append(f1)

    return sum(fold_f1s) / len(fold_f1s) if fold_f1s else None


def _commit_rule(source: str, mask_name: str, rule: str) -> str:
    """Replace learnable declaration with discovered rule."""
    lines = source.splitlines()
    result = []
    for line in lines:
        if line.strip().startswith(f"learnable({mask_name})"):
            result.append(f"    {rule}")
        else:
            result.append(line)
    return "\n".join(result)
