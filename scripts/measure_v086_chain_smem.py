"""Measure the G086_CHAIN_SMEM exact-induction profile trigger.

The fixture is synthetic but certified against the v0.8.6 goal: it routes
through pyxlog's native exact-induction API and shapes candidate rows so the
chain topology performs its O(|L| * |R|) scan while the other topologies stay
linear. The script prints JSON only; evidence files are maintained separately.
"""

from __future__ import annotations

import argparse
import json
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

    # qx=1 appears in every left row. qy is absent from right arg1, forcing
    # the chain predicate to scan every right row for each matching left row.
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
        positive_arg1=tensor([999_999] * queries),
        negative_arg0=tensor([2] * queries),
        negative_arg1=tensor([999_999] * queries),
        k_per_topology=2,
        deterministic=True,
    )
    return prog, kwargs


def measure(rows: int, queries: int, iterations: int, warmup: int) -> dict[str, Any]:
    prog, kwargs = build_request(rows, queries)
    for _ in range(warmup):
        induce_exact(prog, backend="native", **kwargs)

    torch.cuda.synchronize()
    samples: list[float] = []
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
        samples.append(elapsed)

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
    small = measure(args.small_rows, args.small_queries, args.iterations, args.warmup)
    chain_hot = measure(args.hot_rows, args.hot_queries, args.iterations, args.warmup)
    payload = {
        "small": small,
        "chain_hot": chain_hot,
        "hot_to_small_median_ratio": (
            chain_hot["median_seconds"] / small["median_seconds"]
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
