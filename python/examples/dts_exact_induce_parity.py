#!/usr/bin/env python3
"""M8 Phase 1 — canonical parity harness for induce_exact backends.

Builds one bounded DTS-shaped ILP request, runs both the python reference
and the native xlog-induce backend, and reports any structural disagreement.

Exit codes:
  0 — both backends produced the same ExactInductionResult
  1 — mismatch or native backend unavailable
  2 — setup/environment error (no CUDA, bad build)

Used as the cross-repo parity bridge. DTS integration calls this script for
confidence; DTS does not own a duplicate parity implementation.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict

try:
    import torch
except ImportError:
    print("ERROR: torch not available", file=sys.stderr)
    sys.exit(2)

try:
    import pyxlog
    from pyxlog.ilp import induce_exact
except ImportError as e:
    print(f"ERROR: pyxlog not importable: {e}", file=sys.stderr)
    sys.exit(2)


_TARGET_SOURCE = """
    pred p_A(u64, u64).
    pred p_B(u64, u64).
    pred p_C(u64, u64).
    pred p_D(u64, u64).
    learnable(W_chain_p_A)  :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
    learnable(W_star_p_A)   :: p_A(X, Y) :- bL(X, Y), bR(X, Y).
    learnable(W_fanout_p_A) :: p_A(X, Y) :- bL(X, Z), bR(X, Y).
    learnable(W_fanin_p_A)  :: p_A(X, Y) :- bL(X, Y), bR(Z, Y).
"""


def _build_request():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(2)

    prog = pyxlog.IlpProgramFactory.compile(_TARGET_SOURCE, device=0, memory_mb=64)
    dev = torch.device("cuda")

    prog.put_relation("p_B", [
        torch.tensor([1, 2], dtype=torch.int64, device=dev),
        torch.tensor([2, 3], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_C", [
        torch.tensor([2, 3, 4], dtype=torch.int64, device=dev),
        torch.tensor([4, 5, 6], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_D", [
        torch.tensor([1, 2], dtype=torch.int64, device=dev),
        torch.tensor([4, 5], dtype=torch.int64, device=dev),
    ])

    kwargs = dict(
        head_relation="p_A",
        candidate_relations=["p_B", "p_C", "p_D"],
        positive_arg0=torch.tensor([1, 2], dtype=torch.int64, device=dev),
        positive_arg1=torch.tensor([4, 5], dtype=torch.int64, device=dev),
        negative_arg0=torch.tensor([7], dtype=torch.int64, device=dev),
        negative_arg1=torch.tensor([8], dtype=torch.int64, device=dev),
        k_per_topology=2,
        deterministic=True,
    )
    return prog, kwargs


def _summarize_result(name: str, result) -> None:
    print(f"--- {name} ---")
    print(f"  total_scored={result.total_scored}")
    print(f"  candidate_count={result.candidate_count}")
    print(f"  positive_count={result.positive_count}")
    print(f"  negative_count={result.negative_count}")
    print(f"  candidates={len(result.candidates)}")
    for c in result.candidates:
        print(
            f"    [{c.topology:6s}] rank={c.local_rank} "
            f"L={c.left_relation} R={c.right_relation} "
            f"pos={c.positives_covered} neg={c.negatives_covered}"
        )


def _diff_results(py_res, native_res) -> list[str]:
    diffs = []
    for field in ("total_scored", "candidate_count", "positive_count", "negative_count"):
        pv, nv = getattr(py_res, field), getattr(native_res, field)
        if pv != nv:
            diffs.append(f"summary.{field}: python={pv} native={nv}")

    if len(py_res.candidates) != len(native_res.candidates):
        diffs.append(
            f"candidates.len: python={len(py_res.candidates)} "
            f"native={len(native_res.candidates)}"
        )
        return diffs

    for i, (p, n) in enumerate(zip(py_res.candidates, native_res.candidates)):
        for field in (
            "topology", "head_relation", "left_relation", "right_relation",
            "positives_covered", "negatives_covered", "local_rank",
        ):
            pv, nv = getattr(p, field), getattr(n, field)
            if pv != nv:
                diffs.append(
                    f"candidates[{i}].{field}: python={pv!r} native={nv!r}"
                )
    return diffs


def main() -> int:
    prog, kwargs = _build_request()

    print("Running python-reference backend...")
    os.environ.setdefault("XLOG_ALLOW_PYTHON_ILP_REFERENCE", "1")
    py_res = induce_exact(prog, backend="python", **kwargs)
    _summarize_result("python", py_res)

    print("\nRunning native backend...")
    try:
        native_res = induce_exact(prog, backend="native", **kwargs)
    except NotImplementedError as e:
        print(f"native backend not available: {e}", file=sys.stderr)
        return 1
    _summarize_result("native", native_res)

    diffs = _diff_results(py_res, native_res)
    print()
    if not diffs:
        print("PARITY: OK — backends agree")
        return 0
    print("PARITY: MISMATCH")
    for d in diffs:
        print(f"  {d}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
