"""Holdout scoring strategies for dILP.

Current behavior:
- LOO (leave-one-out) for <= 20 positive examples.
- k-fold for larger sets (default: 5 folds).
"""
from __future__ import annotations

import random

import pyxlog
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import TrainConfig


def holdout_f1_and_variance(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> tuple[float | None, float]:
    """Compute holdout F1 and variance using configured strategy.

    Returns (mean_f1, variance). Both are deterministic when `config.seed`
    is fixed because fold assignment is seed-driven.
    """
    if not positives:
        return None, 0.0

    if len(positives) <= 20:
        strategy = "loo"
    else:
        strategy = (config.holdout_strategy or "kfold").lower()

    if strategy == "loo":
        f1 = loo_holdout_f1(source, mask_name, positives, negatives, config)
        return (f1, 0.0) if f1 is not None else (None, 0.0)

    if strategy in {"kfold", "k-fold", "k_fold", "kf"}:
        return k_fold_holdout_f1_and_variance(
            source,
            mask_name,
            positives,
            negatives,
            config,
        )

    # Unknown strategy: safe fallback to LOO.
    f1 = loo_holdout_f1(source, mask_name, positives, negatives, config)
    return (f1, 0.0) if f1 is not None else (None, 0.0)


def loo_holdout_f1(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> float | None:
    """Leave-one-out cross-validation F1 over positives."""
    if len(positives) < 2:
        return None

    fold_f1s: list[float] = []
    pos_set = {(r, tuple(v)) for r, v in positives}
    neg_set = {(r, tuple(v)) for r, v in negatives}

    for i in range(len(positives)):
        train_pos = positives[:i] + positives[i + 1:]
        held_out = positives[i]

        result = train_only(source, mask_name, train_pos, negatives, config, _compute_holdout=False)
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

        # Precision/Recall on held-out positives for this fold.
        recall = 1.0 if trial.fact_exists(held_out[0], held_out[1]) else 0.0

        head_rel = held_out[0]
        all_derived = trial.relation_facts(head_rel)
        if all_derived:
            true_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in pos_set)
            false_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in neg_set)
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1.0
        else:
            precision = 1.0 if recall == 0.0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        fold_f1s.append(f1)

    return sum(fold_f1s) / len(fold_f1s) if fold_f1s else None


def k_fold_holdout_f1_and_variance(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> tuple[float | None, float]:
    """Compute holdout F1 across k folds.

    Stratification is performed over positive examples only.
    """
    n = len(positives)
    if n < 2:
        return None, 0.0

    rng = random.Random(config.seed)
    shuffled_idx = list(range(n))
    rng.shuffle(shuffled_idx)

    k = max(2, min(config.holdout_folds, n))
    folds = [shuffled_idx[i::k] for i in range(k)]

    fold_f1s: list[float] = []
    for fold in folds:
        if not fold:
            continue

        train_pos = [p for idx, p in enumerate(positives) if idx not in set(fold)]
        held_out = [positives[idx] for idx in fold]
        if not train_pos or not held_out:
            continue

        result = train_only(
            source, mask_name, train_pos, negatives, config, _compute_holdout=False
        )
        if not (result.converged and result.discovered_rule):
            fold_f1s.append(0.0)
            continue

        try:
            trial_source = _commit_rule(source, mask_name, result.discovered_rule)
            trial = pyxlog.IlpProgramFactory.compile(
                trial_source, device=config.device, memory_mb=config.memory_mb,
            )
            trial.evaluate()
        except Exception:
            fold_f1s.append(0.0)
            continue

        fold_f1s.append(_evaluate_holdout_f1(
            held_out,
            positives,
            negatives,
            trial,
        ))

    if not fold_f1s:
        return None, 0.0

    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    if len(fold_f1s) == 1:
        return mean_f1, 0.0

    variance = sum((v - mean_f1) ** 2 for v in fold_f1s) / len(fold_f1s)
    return mean_f1, variance


def _evaluate_holdout_f1(
    held_out_positives: list[tuple[str, list[int]]],
    all_positives: list[tuple[str, list[int]]],
    all_negatives: list[tuple[str, list[int]]],
    trial,
) -> float:
    """Evaluate holdout precision/recall for one fold and return F1."""
    pos_set = {(r, tuple(v)) for r, v in all_positives}
    neg_set = {(r, tuple(v)) for r, v in all_negatives}

    if not held_out_positives:
        return 0.0

    hits = sum(1 for sample in held_out_positives if trial.fact_exists(sample[0], sample[1]))
    recall = hits / len(held_out_positives)

    head_rel = held_out_positives[0][0]
    all_derived = trial.relation_facts(head_rel)
    if all_derived:
        true_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in pos_set)
        false_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in neg_set)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1.0
    else:
        precision = 1.0 if recall == 0.0 else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


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
