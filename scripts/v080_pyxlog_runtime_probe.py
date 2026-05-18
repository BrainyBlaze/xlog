#!/usr/bin/env python3
"""Generate v0.8.0 pyxlog runtime probe evidence.

The probe is intentionally small and branch-local: it exercises the public
pyxlog runtime/session controls that are recorded in the v0.8.0 PYAPI evidence
without launching a DTS-DLM pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOGIC_SOURCE_I32 = """
pred edge(i32, i32).
pred reach(i32, i32).
reach(X, Y) :- edge(X, Y).
?- reach(X, Y).
"""


LOGIC_SOURCE_I64 = """
pred edge(i64, i64).
pred reach(i64, i64).
reach(X, Y) :- edge(X, Y).
?- reach(X, Y).
"""


PROB_SOURCE = """
0.3::rain().
query(rain()).
"""


def _stats_keys(stats: dict[str, Any]) -> list[str]:
    return list(stats.keys())


def _run_pyapi_probe() -> dict[str, Any]:
    import pyxlog
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v0.8.0 pyxlog runtime probe")

    small_program = pyxlog.LogicProgram.compile(LOGIC_SOURCE_I32, device=0, memory_mb=512)
    session = small_program.session()
    left = torch.arange(1, 9, device="cuda", dtype=torch.int32)
    right = left + 10
    session.put_relation("edge", [left, right])

    async_eval = session.evaluate_async()
    result = async_eval.result(timeout=30)
    progress_after_async = session.progress_stats()
    chunks = list(result.iter_query_chunks(chunk_rows=3))
    stream_chunks = list(session.evaluate_stream(chunk_rows=3))

    large_program = pyxlog.LogicProgram.compile(LOGIC_SOURCE_I64, device=0, memory_mb=512)
    large_left = torch.arange(200_000, device="cuda", dtype=torch.int64)
    large_right = large_left + 1
    large_result = large_program.evaluate(
        {"edge": [large_left, large_right]},
        memory_mb=512,
    )
    large_memory_stats = large_program.memory_stats()
    try:
        large_program.evaluate({"edge": [large_left, large_right]}, memory_mb=1)
    except Exception as exc:  # probe records the public Python error surface
        typed_memory_limit_error = type(exc).__name__
        typed_memory_limit_message = str(exc)
    else:
        raise RuntimeError("expected 1 MiB per-call limit to raise MemoryError")

    prob_program = pyxlog.Program.compile(PROB_SOURCE, memory_mb=512)
    program_async = prob_program.evaluate_async()
    program_result = program_async.result(timeout=30)

    session_graph = session.cuda_graph_stats()
    session_host = session.host_transfer_stats()
    program_graph = prob_program.cuda_graph_stats()
    program_host = prob_program.host_transfer_stats()

    return {
        "async_done_after_result": async_eval.done(),
        "async_logic_result_type": type(result).__name__,
        "chunk_rows": [int(chunk.num_rows) for chunk in chunks],
        "chunk_tensors_cuda": [
            bool(tensor.is_cuda) for chunk in chunks for tensor in chunk.tensors
        ],
        "large_memory_stats_before_overlimit": large_memory_stats,
        "large_rows": int(large_result.queries[0].num_rows),
        "logic_rows": int(result.queries[0].num_rows),
        "program_async_atoms": [str(atom) for atom in program_result.atoms],
        "program_cuda_graph_stats_keys": _stats_keys(program_graph),
        "program_host_transfer_stats_keys": _stats_keys(program_host),
        "program_memory_stats": prob_program.memory_stats(),
        "program_progress_after_async": prob_program.progress_stats(),
        "pyxlog_file": str(Path(pyxlog.__file__).resolve()),
        "pyxlog_version": getattr(pyxlog, "__version__", "0.7.0"),
        "session_cuda_graph_stats_keys": _stats_keys(session_graph),
        "session_host_transfer_stats_keys": _stats_keys(session_host),
        "session_memory_stats": session.memory_stats(),
        "session_progress_after_async": progress_after_async,
        "stream_chunk_rows": [int(chunk.num_rows) for chunk in stream_chunks],
        "stream_chunk_tensors_cuda": [
            bool(tensor.is_cuda) for chunk in stream_chunks for tensor in chunk.tensors
        ],
        "torch_version": torch.__version__,
        "typed_memory_limit_error": typed_memory_limit_error,
        "typed_memory_limit_message": typed_memory_limit_message,
    }


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        choices=["pyapi"],
        default="pyapi",
        help="Runtime probe to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON evidence to this path. Defaults to stdout.",
    )
    args = parser.parse_args()

    if args.probe != "pyapi":
        raise AssertionError(args.probe)
    _write_json(_run_pyapi_probe(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
