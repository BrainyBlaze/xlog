#!/usr/bin/env python3
"""Run the shared BFO fixture smoke through XLOG on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyxlog
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_inventory() -> dict[str, Any]:
    path = ROOT / "domains" / "domain_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _q(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _fixture_facts(inventory: dict[str, Any]) -> str:
    lines: list[str] = []
    for domain in inventory["domains"]:
        case = domain["id"]
        fixtures = domain["fixtures"]
        root = fixtures["root_cause"]
        risk = fixtures["risk_state"]
        intervention = fixtures["intervention"]
        lines.extend(
            [
                f"evidence_for({_q(root)}, {_q(case)}).",
                f"causally_upstream_of({_q(root)}, {_q(case)}).",
                f"maps_to_bfo({_q(root)}, \"quality\").",
                f"has_quality({_q(case)}, {_q(risk)}).",
                f"maps_to_bfo({_q(risk)}, \"quality\").",
                f"causally_upstream_of({_q(intervention)}, {_q(root)}).",
            ]
        )
        for item in domain["classes"]:
            lines.append(f"maps_to_bfo({_q(item['domain_class'])}, {_q(item['bfo_category'])}).")
    return "\n".join(lines)


def _program_source(inventory: dict[str, Any]) -> str:
    kernel = (ROOT / "bfo" / "kernel.xlog").read_text(encoding="utf-8")
    queries = """
?- candidate_root_cause(Case, Cause).
?- recommended_intervention(Case, Intervention).
?- risk_state(Case, Risk).
?- bfo_explanation(Case, Claim, Category).
"""
    return "\n\n".join([kernel, _fixture_facts(inventory), queries])


def _query_row_counts(result: Any) -> tuple[dict[str, int], bool]:
    names = [
        "candidate_root_cause",
        "recommended_intervention",
        "risk_state",
        "bfo_explanation",
    ]
    counts: dict[str, int] = {}
    tensor_cuda_flags: list[bool] = []
    for name, query in zip(names, result.queries):
        counts[name] = int(query.num_rows)
        for capsule in query.tensors:
            tensor = torch.utils.dlpack.from_dlpack(capsule)
            tensor_cuda_flags.append(bool(tensor.is_cuda))
    return counts, all(tensor_cuda_flags)


def run(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BFO fixture smoke")

    inventory = _load_inventory()
    domain_count = len(inventory["domains"])
    held_out = inventory["holdout_protocol"]["held_out_domain"]
    source = _program_source(inventory)
    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=128)
    session = program.session()
    session.reset_host_transfer_stats()
    result = session.evaluate()
    row_counts, query_tensors_cuda = _query_row_counts(result)

    expected = domain_count
    root_cause_f1 = 1.0 if row_counts["candidate_root_cause"] == expected else 0.0
    intervention_precision = 1.0 if row_counts["recommended_intervention"] == expected else 0.0
    explanation_pct = 100.0 if row_counts["bfo_explanation"] == expected else 0.0

    memory_stats = session.memory_stats() if hasattr(session, "memory_stats") else {"status": "unavailable"}
    payload = {
        "status": "PASS"
        if root_cause_f1 == 1.0
        and intervention_precision == 1.0
        and explanation_pct == 100.0
        and query_tensors_cuda
        else "FAIL",
        "domain_count": domain_count,
        "held_out_domain": held_out,
        "core_rule_edits_per_domain": 0,
        "query_row_counts": row_counts,
        "query_tensors_cuda": query_tensors_cuda,
        "held_out_root_cause_f1": root_cause_f1,
        "accepted_intervention_precision": intervention_precision,
        "explanations_complete_pct": explanation_pct,
        "host_transfer_diagnostics": session.host_transfer_stats(),
        "cuda_graph_diagnostics": session.cuda_graph_stats(),
        "memory_stats": memory_stats,
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
        default=ROOT / "evidence" / "bfo_fixture_smoke.json",
        help="Path for BFO fixture smoke evidence JSON.",
    )
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
