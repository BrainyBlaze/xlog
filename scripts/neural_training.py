from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from scripts.neural_datasets import DatasetManifest


def neural_fixture_smoke_enabled() -> bool:
    """Return true when validators should use built-in neural smoke fixtures."""
    return os.environ.get("XLOG_NEURAL_FIXTURE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    if neural_fixture_smoke_enabled():
        return None
    if override is not None:
        return float(override)
    if mode != "release":
        return None
    value = manifest.metrics.get("min_accuracy")
    if value is None:
        return None
    return float(value)


class TensorClassificationDataset:
    """Small in-memory dataset with the ImageFolder fields used by examples."""

    def __init__(self, images: torch.Tensor, targets: List[int], classes: List[str]):
        if images.size(0) != len(targets):
            raise ValueError("images and targets length mismatch")
        self.images = images
        self.targets = [int(t) for t in targets]
        self.classes = list(classes)
        self.samples = [(f"fixture-{i}", int(t)) for i, t in enumerate(self.targets)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.images[idx], self.targets[idx]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def class_pattern_tensor(
    labels: List[int],
    num_classes: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build deterministic visual fixtures with label-encoded patterns."""
    images = torch.zeros((len(labels), channels, height, width), dtype=torch.float32)
    stripe_w = max(1, width // max(num_classes, 1))
    stripe_h = max(1, height // max(num_classes, 1))
    for row, label in enumerate(labels):
        cls = int(label) % max(num_classes, 1)
        base = (cls + 1) / (num_classes + 1)
        images[row].fill_(base * 0.35)
        channel = cls % max(channels, 1)
        x0 = min(width - 1, cls * stripe_w)
        x1 = min(width, x0 + stripe_w)
        y0 = min(height - 1, cls * stripe_h)
        y1 = min(height, y0 + stripe_h)
        images[row, channel, :, x0:x1] = 1.0
        images[row, :, y0:y1, :] = torch.maximum(
            images[row, :, y0:y1, :],
            torch.full_like(images[row, :, y0:y1, :], 0.65),
        )
    return images


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
    engine=None,
) -> None:
    """Write metrics.json with the frozen schema fields.

    Args:
        metrics_path: File path to write. If None, skip.
        compile_api_sec: Time for pyxlog.Program.compile().
        epoch_sec: List of per-epoch wall-clock times.
        total_train_sec: Wall-clock time for the full training loop.
        n_queries: Total number of training queries per epoch.
        extra: Additional key-value pairs to include.
        engine: Optional pyxlog CompiledProgram; if provided and
                XLOG_WARMUP_PROFILE=1, its warmup_breakdown() is included.
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

    # Include warmup profiling breakdown when available.
    if engine is not None:
        try:
            breakdown = engine.warmup_breakdown()
            if breakdown is not None:
                data["warmup_breakdown"] = breakdown
        except Exception:
            pass  # profiling not available; skip silently

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(data, indent=2) + "\n")
