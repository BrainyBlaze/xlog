#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import pyxlog


def _pairs_from_chunks(chunks: list[object]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for chunk in chunks:
        if int(chunk.num_rows) == 0:
            continue
        left, right = chunk.tensors
        rows = torch.stack([left, right], dim=1).detach().cpu().tolist()
        pairs.extend([[int(a), int(b)] for a, b in rows])
    return sorted(pairs)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v0.8.0 DTS examples")

    source = Path(__file__).with_name("program.xlog").read_text(encoding="utf-8")
    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=512)
    session = program.session()
    session.reset_host_transfer_stats()

    edge_src = torch.tensor([1, 2, 3, 10], device="cuda", dtype=torch.int32)
    edge_dst = torch.tensor([2, 3, 4, 11], device="cuda", dtype=torch.int32)
    session.put_relation("edge", [edge_src, edge_dst])

    async_eval = session.evaluate_async(memory_mb=512)
    result = async_eval.result(timeout=30)
    progress = session.progress_stats()
    chunks = list(result.iter_query_chunks(chunk_rows=2))
    stream_chunks = list(session.evaluate_stream(chunk_rows=3))

    expected = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [10, 11]]
    async_pairs = _pairs_from_chunks(chunks)
    stream_pairs = _pairs_from_chunks(stream_chunks)
    if async_pairs != expected or stream_pairs != expected:
        raise AssertionError({"async": async_pairs, "stream": stream_pairs})

    chunk_tensors_cuda = [
        bool(tensor.is_cuda) for chunk in chunks + stream_chunks for tensor in chunk.tensors
    ]

    summary = {
        "example": "01_async_streaming_reachability",
        "status": "PASS",
        "async_completion": {
            "done_after_result": bool(async_eval.done()),
            "evaluations_completed": int(progress["evaluations_completed"]),
            "last_rows": int(progress["last_rows"]),
        },
        "streaming_chunk_counts": {
            "iter_query_chunks": [int(chunk.num_rows) for chunk in chunks],
            "evaluate_stream": [int(chunk.num_rows) for chunk in stream_chunks],
        },
        "cuda_tensor_checks": {
            "all_stream_tensors_cuda": all(chunk_tensors_cuda),
            "checked_tensor_count": len(chunk_tensors_cuda),
        },
        "host_transfer_diagnostics": session.host_transfer_stats(),
        "graph_diagnostics": session.cuda_graph_stats(),
        "memory_stats": session.memory_stats(),
        "reachability_rows": len(expected),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
