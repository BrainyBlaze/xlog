#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import pyxlog
from pyxlog.ilp import induce_exact


def _candidate_tuple(candidate: object) -> tuple[object, ...]:
    return (
        candidate.topology,
        candidate.head_relation,
        candidate.left_relation,
        candidate.right_relation,
        int(candidate.positives_covered),
        int(candidate.negatives_covered),
        int(candidate.local_rank),
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v0.8.0 external consumer examples")

    source = Path(__file__).with_name("program.xlog").read_text(encoding="utf-8")
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=64)
    device = torch.device("cuda")

    prog.put_relation(
        "p_B",
        [
            torch.tensor([1, 2], dtype=torch.int64, device=device),
            torch.tensor([2, 3], dtype=torch.int64, device=device),
        ],
    )
    prog.put_relation(
        "p_C",
        [
            torch.tensor([2, 3, 4], dtype=torch.int64, device=device),
            torch.tensor([4, 5, 6], dtype=torch.int64, device=device),
        ],
    )
    prog.put_relation(
        "p_D",
        [
            torch.tensor([1, 2], dtype=torch.int64, device=device),
            torch.tensor([4, 5], dtype=torch.int64, device=device),
        ],
    )

    kwargs = {
        "head_relation": "p_A",
        "candidate_relations": ["p_B", "p_C", "p_D"],
        "positive_arg0": torch.tensor([1, 2], dtype=torch.int64, device=device),
        "positive_arg1": torch.tensor([4, 5], dtype=torch.int64, device=device),
        "negative_arg0": torch.tensor([7], dtype=torch.int64, device=device),
        "negative_arg1": torch.tensor([8], dtype=torch.int64, device=device),
        "k_per_topology": 2,
        "deterministic": True,
    }

    os.environ.setdefault("XLOG_ALLOW_PYTHON_ILP_REFERENCE", "1")
    python_result = induce_exact(prog, backend="python", strict_per_topology=True, **kwargs)
    native_result = induce_exact(prog, backend="native", **kwargs)

    python_candidates = [_candidate_tuple(candidate) for candidate in python_result.candidates]
    native_candidates = [_candidate_tuple(candidate) for candidate in native_result.candidates]
    parity = {
        "summary": native_result.total_scored == python_result.total_scored
        and native_result.candidate_count == python_result.candidate_count
        and native_result.positive_count == python_result.positive_count
        and native_result.negative_count == python_result.negative_count,
        "ordered_candidates": native_candidates == python_candidates,
    }
    if not all(parity.values()):
        raise AssertionError(parity)

    summary = {
        "example": "04_native_exact_induction",
        "status": "PASS",
        "exact_induction_parity": parity,
        "candidate_count": int(native_result.candidate_count),
        "total_scored": int(native_result.total_scored),
        "positive_count": int(native_result.positive_count),
        "negative_count": int(native_result.negative_count),
        "topologies": sorted({candidate[0] for candidate in native_candidates}),
        "first_candidate": native_candidates[0] if native_candidates else None,
        "cuda_tensor_checks": {
            "positive_examples_cuda": bool(kwargs["positive_arg0"].is_cuda),
            "negative_examples_cuda": bool(kwargs["negative_arg0"].is_cuda),
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
