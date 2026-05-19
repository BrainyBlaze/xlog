#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import pyxlog


def _rows(result: object) -> list[list[int]]:
    query = result.queries[0]
    if int(query.num_rows) == 0:
        return []
    left = torch.utils.dlpack.from_dlpack(query.tensors[0])
    right = torch.utils.dlpack.from_dlpack(query.tensors[1])
    rows = torch.stack([left, right], dim=1).detach().cpu().tolist()
    return sorted([[int(a), int(b)] for a, b in rows])


def _cols(pairs: list[tuple[int, int]]) -> list[torch.Tensor]:
    return [
        torch.tensor([a for a, _ in pairs], device="cuda", dtype=torch.int32),
        torch.tensor([b for _, b in pairs], device="cuda", dtype=torch.int32),
    ]


def _full_replacement_rows(program: object, pairs: list[tuple[int, int]]) -> list[list[int]]:
    session = program.session()
    session.put_relation("wmir_committed", _cols(pairs))
    return _rows(session.evaluate())


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v0.8.0 DTS examples")

    source = Path(__file__).with_name("program.xlog").read_text(encoding="utf-8")
    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=512)
    session = program.session()

    initial_edges = [(1, 2), (2, 3)]
    session.put_relation("wmir_committed", _cols(initial_edges))
    initial_rows = _rows(session.evaluate())

    insert_edges = [(3, 4)]
    insert_stats = session.insert_relation("wmir_committed", _cols(insert_edges))
    after_insert_rows = _rows(session.evaluate())
    full_after_insert_rows = _full_replacement_rows(program, initial_edges + insert_edges)

    delete_edges = [(2, 3)]
    delete_stats = session.delete_relation("wmir_committed", _cols(delete_edges))
    after_delete_rows = _rows(session.evaluate())
    full_after_delete_rows = _full_replacement_rows(program, [(1, 2), (3, 4)])

    mixed_stats = session.apply_relation_delta(
        "wmir_committed",
        insert_columns=_cols([(2, 4)]),
        delete_columns=_cols([(1, 2)]),
    )
    after_mixed_rows = _rows(session.evaluate())
    full_after_mixed_rows = _full_replacement_rows(program, [(3, 4), (2, 4)])

    equivalence = {
        "insert": after_insert_rows == full_after_insert_rows,
        "delete": after_delete_rows == full_after_delete_rows,
        "mixed_apply_relation_delta": after_mixed_rows == full_after_mixed_rows,
    }
    if not all(equivalence.values()):
        raise AssertionError(equivalence)

    summary = {
        "example": "02_wmir_relation_deltas",
        "status": "PASS",
        "relation_delta_equivalence": equivalence,
        "relation_delta_row_counts": {
            "initial_reach_rows": len(initial_rows),
            "after_insert_rows": len(after_insert_rows),
            "after_delete_rows": len(after_delete_rows),
            "after_mixed_rows": len(after_mixed_rows),
            "delta_rows_uploaded": 4,
            "full_replacement_rows_for_equivalent_states": 7,
        },
        "delta_stats": {
            "insert": insert_stats,
            "delete": delete_stats,
            "mixed_apply_relation_delta": mixed_stats,
            "latest": session.delta_stats(),
        },
        "cuda_tensor_checks": {
            "input_tensors_cuda": True,
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
