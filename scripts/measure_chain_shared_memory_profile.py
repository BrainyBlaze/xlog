"""Measure the chain-topology shared-memory exact-induction profile trigger.

The fixture routes through pyxlog's native exact-induction API and shapes
candidate rows so the chain topology performs its O(|L| * |R|) scan while the
other topologies stay linear. The script prints JSON only; evidence files are
maintained separately.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import torch
import pyxlog
from pyxlog.ilp import induce_exact


SOURCE = """
    pred p_A(u64, u64).
    pred p_B(u64, u64).
    pred p_C(u64, u64).
    learnable(W_chain_p_A)  :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
    learnable(W_star_p_A)   :: p_A(X, Y) :- bL(X, Y), bR(X, Y).
    learnable(W_fanout_p_A) :: p_A(X, Y) :- bL(X, Z), bR(X, Y).
    learnable(W_fanin_p_A)  :: p_A(X, Y) :- bL(X, Y), bR(Z, Y).
"""


def tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def build_request(rows: int, queries: int):
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=256)

    # The first query argument appears in every left row. The second query
    # argument matches only the final right row, forcing the chain predicate to
    # scan the full right relation before finding coverage for each positive query.
    left_arg0 = [1] * rows
    left_arg1 = list(range(10_000, 10_000 + rows))
    right_arg0 = list(range(10_000, 10_000 + rows))
    right_arg1 = list(range(20_000, 20_000 + rows))

    prog.put_relation("p_B", [tensor(left_arg0), tensor(left_arg1)])
    prog.put_relation("p_C", [tensor(right_arg0), tensor(right_arg1)])

    kwargs = dict(
        head_relation="p_A",
        candidate_relations=["p_B", "p_C"],
        positive_arg0=tensor([1] * queries),
        positive_arg1=tensor([20_000 + rows - 1] * queries),
        negative_arg0=tensor([2] * queries),
        negative_arg1=tensor([999_999] * queries),
        k_per_topology=2,
        deterministic=True,
    )
    return prog, kwargs


CHAIN_SHARED_MEMORY_ENV = "XLOG_ILP_EXACT_CHAIN_SMEM"


def measure(
    rows: int,
    queries: int,
    iterations: int,
    warmup: int,
    *,
    chain_shared_memory_enabled: bool,
) -> dict[str, Any]:
    previous = os.environ.get(CHAIN_SHARED_MEMORY_ENV)
    os.environ[CHAIN_SHARED_MEMORY_ENV] = "1" if chain_shared_memory_enabled else "0"
    prog, kwargs = build_request(rows, queries)
    try:
        for _ in range(warmup):
            induce_exact(prog, backend="native", **kwargs)

        torch.cuda.synchronize()
        samples: list[float] = []
        last_signature = None
        for _ in range(iterations):
            prog.reset_d2h_transfer_count()
            start = time.perf_counter()
            result = induce_exact(prog, backend="native", **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if prog.d2h_transfer_count() != 2:
                raise RuntimeError(
                    f"expected dtoh_calls=2, got {prog.d2h_transfer_count()}"
                )
            if result.total_scored != 16:
                raise RuntimeError(f"expected total_scored=16, got {result.total_scored}")
            signature = [
                (
                    c.topology,
                    c.left_relation,
                    c.right_relation,
                    c.positives_covered,
                    c.negatives_covered,
                    c.local_rank,
                )
                for c in result.candidates
            ]
            if last_signature is not None and signature != last_signature:
                raise RuntimeError("exact-induction result changed across iterations")
            last_signature = signature
            samples.append(elapsed)
    finally:
        if previous is None:
            os.environ.pop(CHAIN_SHARED_MEMORY_ENV, None)
        else:
            os.environ[CHAIN_SHARED_MEMORY_ENV] = previous

    return {
        "rows_per_candidate": rows,
        "query_pairs_positive": queries,
        "query_pairs_negative": queries,
        "iterations": iterations,
        "warmup": warmup,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "dtoh_calls": 2,
        "chain_shared_memory_enabled": chain_shared_memory_enabled,
        "result_signature": last_signature,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-rows", type=int, default=32)
    parser.add_argument("--small-queries", type=int, default=8)
    parser.add_argument("--hot-rows", type=int, default=768)
    parser.add_argument("--hot-queries", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    small_baseline = measure(
        args.small_rows,
        args.small_queries,
        args.iterations,
        args.warmup,
        chain_shared_memory_enabled=False,
    )
    small_shared_memory = measure(
        args.small_rows,
        args.small_queries,
        args.iterations,
        args.warmup,
        chain_shared_memory_enabled=True,
    )
    chain_hot_baseline = measure(
        args.hot_rows,
        args.hot_queries,
        args.iterations,
        args.warmup,
        chain_shared_memory_enabled=False,
    )
    chain_hot_shared_memory = measure(
        args.hot_rows,
        args.hot_queries,
        args.iterations,
        args.warmup,
        chain_shared_memory_enabled=True,
    )
    hot_speedup = (
        chain_hot_baseline["median_seconds"] / chain_hot_shared_memory["median_seconds"]
    )
    small_regression = (
        (small_shared_memory["median_seconds"] - small_baseline["median_seconds"])
        / small_baseline["median_seconds"]
        * 100.0
    )
    payload = {
        "small": {
            "baseline": small_baseline,
            "chain_shared_memory": small_shared_memory,
            "parity": small_baseline["result_signature"]
            == small_shared_memory["result_signature"],
            "regression_percent": small_regression,
        },
        "chain_hot": {
            "baseline": chain_hot_baseline,
            "chain_shared_memory": chain_hot_shared_memory,
            "parity": chain_hot_baseline["result_signature"]
            == chain_hot_shared_memory["result_signature"],
            "speedup_ratio": hot_speedup,
        },
        "transfer_budget": {
            "baseline_dtoh_calls": chain_hot_baseline["dtoh_calls"],
            "chain_shared_memory_dtoh_calls": chain_hot_shared_memory["dtoh_calls"],
            "added_dtoh_calls": chain_hot_shared_memory["dtoh_calls"]
            - chain_hot_baseline["dtoh_calls"],
        },
        "fallback": {
            "non_chain_uses_baseline_logic": True,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
