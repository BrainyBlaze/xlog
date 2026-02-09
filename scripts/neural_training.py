from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from scripts.neural_datasets import DatasetManifest


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_epochs(
    mode: str,
    override: Optional[int],
    ci: int,
    dev: int,
    release: int,
) -> int:
    if override is not None:
        return max(1, int(override))
    if mode == "ci":
        return ci
    if mode == "dev":
        return dev
    return release


def resolve_min_accuracy(
    mode: str,
    manifest: DatasetManifest,
    override: Optional[float],
) -> Optional[float]:
    if override is not None:
        return float(override)
    if mode != "release":
        return None
    value = manifest.metrics.get("min_accuracy")
    if value is None:
        return None
    return float(value)


def split_indices(
    n_items: int,
    eval_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_items <= 0:
        raise ValueError("n_items must be > 0")
    ratio = max(0.0, min(0.9, float(eval_ratio)))
    eval_n = int(n_items * ratio)
    if eval_n <= 0:
        idx = torch.arange(n_items, dtype=torch.long)
        return idx, idx
    if eval_n >= n_items:
        eval_n = n_items - 1

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=gen)
    eval_idx = perm[:eval_n]
    train_idx = perm[eval_n:]
    return train_idx, eval_idx


@torch.no_grad()
def classification_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 256,
) -> float:
    model.eval()
    correct = 0
    total = int(targets.numel())
    if total == 0:
        return 0.0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        probs = model(inputs[start:end])
        pred = probs.argmax(dim=1)
        correct += int((pred == targets[start:end]).sum().item())
    return correct / total


def report_and_enforce_metric(
    metric_name: str,
    metric_value: float,
    threshold: Optional[float],
) -> None:
    threshold_text = "none" if threshold is None else f"{threshold:.4f}"
    print(f"FINAL_METRIC: {metric_name}={metric_value:.4f}, threshold={threshold_text}")
    if threshold is None:
        return
    if metric_value + 1e-12 < threshold:
        raise SystemExit(
            f"accuracy gate failed: {metric_name}={metric_value:.4f} < threshold={threshold:.4f}"
        )
