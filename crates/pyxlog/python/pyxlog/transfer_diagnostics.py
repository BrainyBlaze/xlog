"""Grouped transfer metric diagnostics for reusable pyxlog reports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PredictionRecord:
    domain: str
    variant: str
    y_true: int
    y_pred: int
    baseline_pred: int | None = None


@dataclass(frozen=True)
class GroupMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedSignTest:
    model_wins: int
    baseline_wins: int
    ties: int
    p_value: float


@dataclass
class TransferDiagnostics:
    per_domain: dict[str, GroupMetrics]
    macro_f1: float
    minimum_domain_f1: float
    baseline_macro_f1: float | None = None
    baseline_uplift: float | None = None
    bootstrap_ci: ConfidenceInterval = ConfidenceInterval(0.0, 0.0)
    paired_sign_test: PairedSignTest | None = None
    missing_domains: list[str] = field(default_factory=list)
    missing_variants: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing_domains and not self.missing_variants


def compute_transfer_diagnostics(
    records: Iterable[PredictionRecord],
    *,
    required_domains: tuple[str, ...] | list[str] = (),
    required_variants: tuple[str, ...] | list[str] = (),
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> TransferDiagnostics:
    rows = list(records)
    domains = sorted({row.domain for row in rows})
    variants = sorted({row.variant for row in rows})
    missing_domains = sorted(set(required_domains) - set(domains))
    missing_variants = sorted(set(required_variants) - set(variants))

    per_domain = {
        domain: _metrics_for([row for row in rows if row.domain == domain], use_baseline=False)
        for domain in domains
    }
    macro_f1 = _macro_f1(rows, use_baseline=False)
    minimum_domain_f1 = min((metrics.f1 for metrics in per_domain.values()), default=0.0)

    baseline_macro_f1 = None
    baseline_uplift = None
    paired = None
    if rows and all(row.baseline_pred is not None for row in rows):
        baseline_macro_f1 = _macro_f1(rows, use_baseline=True)
        baseline_uplift = macro_f1 - baseline_macro_f1
        paired = _paired_sign_test(rows)

    ci = _bootstrap_macro_f1(rows, bootstrap_samples, seed)
    return TransferDiagnostics(
        per_domain=per_domain,
        macro_f1=macro_f1,
        minimum_domain_f1=minimum_domain_f1,
        baseline_macro_f1=baseline_macro_f1,
        baseline_uplift=baseline_uplift,
        bootstrap_ci=ci,
        paired_sign_test=paired,
        missing_domains=missing_domains,
        missing_variants=missing_variants,
    )


def _metrics_for(rows: list[PredictionRecord], *, use_baseline: bool) -> GroupMetrics:
    tp = fp = fn = 0
    for row in rows:
        pred = row.baseline_pred if use_baseline else row.y_pred
        if pred is None:
            continue
        if row.y_true == 1 and pred == 1:
            tp += 1
        elif row.y_true == 0 and pred == 1:
            fp += 1
        elif row.y_true == 1 and pred == 0:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return GroupMetrics(precision=precision, recall=recall, f1=f1, support=len(rows))


def _macro_f1(rows: list[PredictionRecord], *, use_baseline: bool) -> float:
    domains = sorted({row.domain for row in rows})
    if not domains:
        return 0.0
    return sum(
        _metrics_for([row for row in rows if row.domain == domain], use_baseline=use_baseline).f1
        for domain in domains
    ) / len(domains)


def _bootstrap_macro_f1(
    rows: list[PredictionRecord], samples: int, seed: int
) -> ConfidenceInterval:
    if not rows or samples <= 0:
        value = _macro_f1(rows, use_baseline=False)
        return ConfidenceInterval(value, value)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        values.append(_macro_f1(sample, use_baseline=False))
    values.sort()
    lower_idx = int(0.025 * (len(values) - 1))
    upper_idx = int(0.975 * (len(values) - 1))
    return ConfidenceInterval(values[lower_idx], values[upper_idx])


def _paired_sign_test(rows: list[PredictionRecord]) -> PairedSignTest:
    model_wins = baseline_wins = ties = 0
    for row in rows:
        baseline = row.baseline_pred
        if baseline is None:
            ties += 1
            continue
        model_correct = row.y_pred == row.y_true
        baseline_correct = baseline == row.y_true
        if model_correct and not baseline_correct:
            model_wins += 1
        elif baseline_correct and not model_correct:
            baseline_wins += 1
        else:
            ties += 1

    discordant = model_wins + baseline_wins
    if discordant == 0:
        return PairedSignTest(model_wins, baseline_wins, ties, 1.0)
    tail = min(model_wins, baseline_wins)
    p = 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return PairedSignTest(model_wins, baseline_wins, ties, min(1.0, p))
