from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


def write_frozen_metrics(
    metrics_path: Optional[str],
    compile_api_sec: float,
    epoch_sec: List[float],
    total_train_sec: float,
    n_queries: int,
    extra: Optional[dict] = None,
) -> None:
    """Write metrics.json with the frozen schema fields.

    Args:
        metrics_path: File path to write. If None, skip.
        compile_api_sec: Time for pyxlog.Program.compile().
        epoch_sec: List of per-epoch wall-clock times.
        total_train_sec: Wall-clock time for the full training loop.
        n_queries: Total number of training queries per epoch.
        extra: Additional key-value pairs to include.
    """
    if metrics_path is None:
        return

    first_epoch_sec = epoch_sec[0] if epoch_sec else 0.0
    if len(epoch_sec) > 1:
        steady = epoch_sec[1:]
        steady_epoch_sec_mean = round(sum(steady) / len(steady), 3)
    else:
        steady_epoch_sec_mean = round(first_epoch_sec, 3)
    warmup_sec = round(first_epoch_sec - steady_epoch_sec_mean, 3)
    if warmup_sec < 0:
        warmup_sec = 0.0

    n_epochs = len(epoch_sec)
    if n_queries > 0 and n_epochs > 0:
        per_query_ms = round(total_train_sec / (n_queries * n_epochs) * 1000, 3)
    else:
        per_query_ms = 0.0

    data = {
        "compile_api_sec": round(compile_api_sec, 3),
        "epoch_sec": [round(t, 3) for t in epoch_sec],
        "first_epoch_sec": round(first_epoch_sec, 3),
        "steady_epoch_sec_mean": steady_epoch_sec_mean,
        "warmup_sec": warmup_sec,
        "total_train_sec": round(total_train_sec, 3),
        "per_query_ms": per_query_ms,
    }
    if extra:
        data.update(extra)

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(data, indent=2) + "\n")
