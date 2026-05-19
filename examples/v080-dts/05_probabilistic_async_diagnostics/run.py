#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import pyxlog


def _dlpack_values(capsule: object) -> tuple[list[float], bool]:
    tensor = torch.from_dlpack(capsule)
    return [float(v) for v in tensor.detach().cpu().tolist()], bool(tensor.is_cuda)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v0.8.0 DTS examples")

    source = Path(__file__).with_name("program.xlog").read_text(encoding="utf-8")

    exact_program = pyxlog.Program.compile(source, device=0, memory_mb=512, prob_engine="exact_ddnnf")
    exact_program.reset_host_transfer_stats()
    exact_async = exact_program.evaluate_async(memory_mb=512)
    exact_result = exact_async.result(timeout=30)
    exact_progress = exact_program.progress_stats()
    exact_probabilities, exact_probability_tensor_cuda = _dlpack_values(exact_result.prob)

    mc_program = pyxlog.Program.compile(source, device=0, memory_mb=512, prob_engine="mc")
    mc_async = mc_program.evaluate_async(
        samples=128,
        seed=7,
        confidence=0.9,
        sampling_method="rejection",
        memory_mb=512,
    )
    mc_result = mc_async.result(timeout=30)
    mc_progress = mc_program.progress_stats()

    summary = {
        "example": "05_probabilistic_async_diagnostics",
        "status": "PASS",
        "async_completion": {
            "exact_done_after_result": bool(exact_async.done()),
            "mc_done_after_result": bool(mc_async.done()),
            "exact_evaluations_completed": int(exact_progress["evaluations_completed"]),
            "mc_evaluations_completed": int(mc_progress["evaluations_completed"]),
        },
        "probabilistic_modes": {
            "exact_atoms": [str(atom) for atom in exact_result.atoms],
            "exact_probabilities": exact_probabilities,
            "mc_atoms": [str(atom) for atom in mc_result.atoms],
            "mc_approx": bool(mc_result.approx),
            "mc_samples": int(mc_result.samples),
            "mc_seed": int(mc_result.seed),
        },
        "host_transfer_diagnostics": {
            "exact": exact_program.host_transfer_stats(),
            "mc": mc_program.host_transfer_stats(),
        },
        "graph_diagnostics": {
            "exact": exact_program.cuda_graph_stats(),
            "mc": mc_program.cuda_graph_stats(),
        },
        "memory_stats": {
            "exact": exact_program.memory_stats(),
            "mc": mc_program.memory_stats(),
        },
        "cuda_tensor_checks": {
            "exact_probability_tensor_cuda": exact_probability_tensor_cuda,
        },
    }
    if not (summary["async_completion"]["exact_done_after_result"] and summary["async_completion"]["mc_done_after_result"]):
        raise AssertionError(summary["async_completion"])
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
