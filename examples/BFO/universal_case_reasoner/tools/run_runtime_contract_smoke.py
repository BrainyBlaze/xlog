#!/usr/bin/env python3
"""Run runtime-contract evidence for delta, device residency, and determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyxlog
import torch


ROOT = Path(__file__).resolve().parents[1]


RUNTIME_SOURCE = """
pred edge(u32, u32).
pred reach(u32, u32).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(X, Y).
"""


def _cols(pairs: list[tuple[int, int]]) -> list[torch.Tensor]:
    return [
        torch.tensor([a for a, _ in pairs], device="cuda", dtype=torch.int32),
        torch.tensor([b for _, b in pairs], device="cuda", dtype=torch.int32),
    ]


def _rows(result: Any) -> list[list[int]]:
    query = result.queries[0]
    if int(query.num_rows) == 0:
        return []
    left = torch.utils.dlpack.from_dlpack(query.tensors[0])
    right = torch.utils.dlpack.from_dlpack(query.tensors[1])
    rows = torch.stack([left, right], dim=1).detach().cpu().tolist()
    return sorted([[int(a), int(b)] for a, b in rows])


def _full_recompute_rows(program: Any, pairs: list[tuple[int, int]]) -> list[list[int]]:
    session = program.session()
    session.put_relation("edge", _cols(pairs))
    return _rows(session.evaluate())


def _delta_equivalence(program: Any) -> dict[str, Any]:
    session = program.session()
    initial = [(1, 2), (2, 3)]
    session.put_relation("edge", _cols(initial))
    initial_rows = _rows(session.evaluate())

    insert = [(3, 4)]
    session.insert_relation("edge", _cols(insert))
    after_insert = _rows(session.evaluate())
    full_insert = _full_recompute_rows(program, initial + insert)

    delete = [(2, 3)]
    session.delete_relation("edge", _cols(delete))
    after_delete = _rows(session.evaluate())
    full_delete = _full_recompute_rows(program, [(1, 2), (3, 4)])

    session.apply_relation_delta(
        "edge",
        insert_columns=_cols([(2, 4)]),
        delete_columns=_cols([(1, 2)]),
    )
    after_mixed = _rows(session.evaluate())
    full_mixed = _full_recompute_rows(program, [(3, 4), (2, 4)])

    checks = {
        "insert": after_insert == full_insert,
        "delete": after_delete == full_delete,
        "mixed": after_mixed == full_mixed,
    }
    return {
        "checks": checks,
        "delta_output_equals_full_recompute_pct": 100.0
        if all(checks.values())
        else 100.0 * sum(1 for ok in checks.values() if ok) / len(checks),
        "row_counts": {
            "initial": len(initial_rows),
            "after_insert": len(after_insert),
            "after_delete": len(after_delete),
            "after_mixed": len(after_mixed),
        },
    }


def _hot_loop_transfer_stats(program: Any) -> dict[str, int]:
    session = program.session()
    session.put_relation("edge", _cols([(1, 2), (2, 3), (3, 4), (10, 11)]))
    session.evaluate()
    session.reset_host_transfer_stats()
    for _ in range(3):
        session.evaluate()
    return dict(session.host_transfer_stats())


def _deterministic_replay(program: Any) -> dict[str, Any]:
    canonical_payloads: list[str] = []
    for _ in range(5):
        session = program.session()
        session.put_relation("edge", _cols([(1, 2), (2, 3), (3, 4), (10, 11)]))
        rows = _rows(session.evaluate())
        payload = json.dumps({"seed": 0, "rows": rows}, sort_keys=True, separators=(",", ":"))
        canonical_payloads.append(payload)
    first = canonical_payloads[0]
    digests = [hashlib.sha256(payload.encode("utf-8")).hexdigest() for payload in canonical_payloads]
    matching = sum(1 for payload in canonical_payloads if payload == first)
    return {
        "runs": len(canonical_payloads),
        "matching_runs": matching,
        "byte_identical": matching == len(canonical_payloads),
        "seed": 0,
        "digests": digests,
    }


def run(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runtime contract smoke")

    program = pyxlog.LogicProgram.compile(RUNTIME_SOURCE, device=0, memory_mb=128)
    delta = _delta_equivalence(program)
    transfers = _hot_loop_transfer_stats(program)
    determinism = _deterministic_replay(program)
    zero_transfers = all(
        int(transfers.get(key, -1)) == 0
        for key in ["dtoh_calls", "htod_calls", "dtoh_bytes", "htod_bytes"]
    )

    payload = {
        "status": "PASS"
        if delta["delta_output_equals_full_recompute_pct"] == 100.0
        and zero_transfers
        and determinism["byte_identical"]
        else "FAIL",
        "delta_output_equals_full_recompute_pct": delta[
            "delta_output_equals_full_recompute_pct"
        ],
        "delta": delta,
        "hot_loop_transfer_stats": transfers,
        "determinism": determinism,
        "cuda_device": torch.cuda.get_device_name(0),
        "pyxlog_version": getattr(pyxlog, "__version__", "unknown"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "runtime_contract_smoke.json",
        help="Path for runtime contract evidence JSON.",
    )
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
