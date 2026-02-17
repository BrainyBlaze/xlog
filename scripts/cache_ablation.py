#!/usr/bin/env python3
"""Cache ablation benchmark: circuit template caching enabled vs disabled.

Compares training wall-clock time on the 01_minimal MNIST addition example
with and without circuit template caching.  For each seed the script creates
a fresh program + network, runs training with caching (default behaviour),
then repeats with caching defeated by calling program.clear_circuit_cache()
before every epoch.

Note:
    If using a locally-built pyxlog, you may need to set LD_LIBRARY_PATH to
    include the directory containing the compiled shared library, e.g.::

        export LD_LIBRARY_PATH="$PWD/target/release:$LD_LIBRARY_PATH"

Usage:
    python scripts/cache_ablation.py --train-limit 128 --epochs 2
    python scripts/cache_ablation.py --output results/cache_ablation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the 01_minimal example importable.  The directory name starts with a
# digit so we cannot use a normal dotted import; instead we add it directly
# to sys.path and import ``train`` as a top-level module.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_DIR = _PROJECT_ROOT / "examples" / "neural" / "01_minimal"
sys.path.insert(0, str(_EXAMPLE_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402
import pyxlog  # noqa: E402

from train import MNISTNet, create_program, generate_queries, load_mnist  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_program(
    seed: int,
    train_images: torch.Tensor,
    train_labels: list[int],
    device: str,
    lr: float = 1e-3,
):
    """Create a fresh program, network, and query set (deterministic for *seed*)."""
    torch.manual_seed(seed)
    program = create_program()
    net = MNISTNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    program.register_network("mnist_net", net, optimizer)

    if device == "cuda":
        program.add_tensor_source("train", train_images.cuda())
    else:
        program.add_tensor_source("train", train_images)

    n_pairs = len(train_labels) // 2
    queries = generate_queries(n_pairs, train_labels)
    return program, queries


def _run_cached(program, queries: list[str], epochs: int, batch_size: int) -> float:
    """Train with circuit cache enabled (default) and return elapsed seconds."""
    start = time.perf_counter()
    pyxlog.train_model_tensor(
        program,
        queries,
        epochs=epochs,
        batch_size=batch_size,
        log_iter=999_999,  # suppress per-batch logging
        shuffle=False,
    )
    elapsed = time.perf_counter() - start
    return elapsed


def _run_uncached(program, queries: list[str], epochs: int, batch_size: int) -> float:
    """Train with circuit cache cleared before every epoch and return elapsed seconds.

    We cannot use ``train_model_tensor`` here because it does not clear the
    cache between epochs.  Instead we manually loop and call
    ``program.train_epoch_tensor()`` with ``program.clear_circuit_cache()``
    before each iteration.
    """
    start = time.perf_counter()
    for _epoch in range(epochs):
        program.clear_circuit_cache()
        program.train_epoch_tensor(queries, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return elapsed


def _compute_speedup(cached_times: list[float], uncached_times: list[float]) -> dict:
    """Compute speedup ratio (uncached / cached) with 95% CI.

    Uses ``scipy.stats.t.interval`` when scipy is available, otherwise falls
    back to a simple percentile bootstrap.
    """
    ratios = [u / c for c, u in zip(cached_times, uncached_times)]
    n = len(ratios)
    mean_ratio = sum(ratios) / n

    if n < 2:
        return {"ratio": mean_ratio, "ci_95_lower": mean_ratio, "ci_95_upper": mean_ratio}

    std_ratio = math.sqrt(sum((r - mean_ratio) ** 2 for r in ratios) / (n - 1))
    se = std_ratio / math.sqrt(n)

    try:
        from scipy.stats import t as t_dist
        lower, upper = t_dist.interval(0.95, df=n - 1, loc=mean_ratio, scale=se)
    except ImportError:
        # Percentile bootstrap fallback
        import random
        random.seed(0)
        n_boot = 10_000
        boot_means: list[float] = []
        for _ in range(n_boot):
            sample = [random.choice(ratios) for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        lower = boot_means[int(0.025 * n_boot)]
        upper = boot_means[int(0.975 * n_boot)]

    return {"ratio": mean_ratio, "ci_95_lower": lower, "ci_95_upper": upper}


def _stats_block(times: list[float]) -> dict:
    """Build a stats dictionary for a list of timing measurements."""
    n = len(times)
    mean = sum(times) / n
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1)) if n > 1 else 0.0
    return {"times_sec": times, "mean_sec": mean, "std_sec": std}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache ablation benchmark for circuit template caching",
    )
    parser.add_argument(
        "--train-limit", type=int, default=128,
        help="Number of training images to use (default: 128)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Epochs per run (default: 2)",
    )
    parser.add_argument(
        "--seeds", type=str, default="7,42,123",
        help="Comma-separated random seeds (default: '7,42,123')",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data/mnist",
        help="Path to MNIST data directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results (default: print to stdout)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Cache ablation benchmark")
    print(f"  device       : {device}")
    print(f"  train_limit  : {args.train_limit}")
    print(f"  epochs       : {args.epochs}")
    print(f"  seeds        : {seeds}")
    print()

    # Load data once (shared across all runs).
    print("Loading MNIST data...")
    train_images, train_labels = load_mnist(args.data_path)
    train_images = train_images[: args.train_limit]
    train_labels = train_labels[: args.train_limit]
    print(f"  Using {len(train_labels)} images ({len(train_labels) // 2} pairs)")
    print()

    cached_times: list[float] = []
    uncached_times: list[float] = []
    batch_size = 32

    for seed in seeds:
        # --- cache enabled ---
        print(f"[seed={seed}] cache ENABLED  ... ", end="", flush=True)
        program, queries = _setup_program(seed, train_images, train_labels, device)
        t = _run_cached(program, queries, args.epochs, batch_size)
        cached_times.append(t)
        print(f"{t:.3f}s")

        # --- cache disabled ---
        print(f"[seed={seed}] cache DISABLED ... ", end="", flush=True)
        program, queries = _setup_program(seed, train_images, train_labels, device)
        t = _run_uncached(program, queries, args.epochs, batch_size)
        uncached_times.append(t)
        print(f"{t:.3f}s")

    print()

    result = {
        "example": "01_minimal",
        "train_limit": args.train_limit,
        "epochs": args.epochs,
        "seeds": seeds,
        "cache_enabled": _stats_block(cached_times),
        "cache_disabled": _stats_block(uncached_times),
        "speedup": _compute_speedup(cached_times, uncached_times),
    }

    result_json = json.dumps(result, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result_json + "\n")
        print(f"Results written to {args.output}")
    else:
        print(result_json)

    # Quick summary
    ce = result["cache_enabled"]
    cd = result["cache_disabled"]
    sp = result["speedup"]
    print()
    print("Summary:")
    print(f"  Cache enabled  : {ce['mean_sec']:.3f}s +/- {ce['std_sec']:.3f}s")
    print(f"  Cache disabled : {cd['mean_sec']:.3f}s +/- {cd['std_sec']:.3f}s")
    print(f"  Speedup ratio  : {sp['ratio']:.2f}x  (95% CI: [{sp['ci_95_lower']:.2f}, {sp['ci_95_upper']:.2f}])")


if __name__ == "__main__":
    main()
