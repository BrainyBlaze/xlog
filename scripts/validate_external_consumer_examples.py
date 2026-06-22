#!/usr/bin/env python3
"""Run the external consumer Python examples and write one validation summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "external-consumer-python"
DEFAULT_OUTPUT = (
    ROOT / "docs" / "evidence" / "external-consumer-examples" / "validation_summary.json"
)
EXAMPLES = [
    "01_async_streaming_reachability",
    "02_relation_deltas",
    "03_neural_bridge_topk_belnap",
    "04_native_exact_induction",
    "05_probabilistic_async_diagnostics",
]


def _load_example_result(example: str, python: str, timeout: int) -> dict[str, Any]:
    script = EXAMPLE_ROOT / example / "run.py"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(
        [python, str(script)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(
            f"{example} failed with exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{example} produced no JSON output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{example} did not end with JSON: {exc}\n{proc.stdout}") from exc
    payload["duration_sec"] = round(elapsed, 6)
    if payload.get("status") != "PASS":
        raise SystemExit(f"{example} returned non-PASS status: {payload}")
    return payload


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["example"]: result for result in results}
    per_example = [
        {
            "name": name,
            "status": by_name[name]["status"],
            "duration_sec": by_name[name]["duration_sec"],
        }
        for name in EXAMPLES
    ]
    return {
        "suite": "external_consumer_examples",
        "status": "PASS",
        "example_count": len(results),
        "per_example": per_example,
        "cuda_tensor_checks": {
            name: by_name[name].get("cuda_tensor_checks", {}) for name in EXAMPLES
        },
        "zero_copy_diagnostics": {
            "01_async_streaming_reachability": by_name["01_async_streaming_reachability"][
                "cuda_tensor_checks"
            ],
        },
        "host_transfer_diagnostics": {
            name: by_name[name].get("host_transfer_diagnostics", {}) for name in EXAMPLES
        },
        "graph_diagnostics": {
            name: by_name[name].get("graph_diagnostics", {}) for name in EXAMPLES
        },
        "async_completion": {
            name: by_name[name]["async_completion"]
            for name in [
                "01_async_streaming_reachability",
                "05_probabilistic_async_diagnostics",
            ]
        },
        "streaming_chunk_counts": {
            "01_async_streaming_reachability": by_name["01_async_streaming_reachability"][
                "streaming_chunk_counts"
            ],
        },
        "relation_delta_equivalence": by_name["02_relation_deltas"][
            "relation_delta_equivalence"
        ],
        "relation_delta_row_counts": by_name["02_relation_deltas"][
            "relation_delta_row_counts"
        ],
        "deterministic_topk_selected_labels": by_name["03_neural_bridge_topk_belnap"][
            "deterministic_topk_selected_labels"
        ],
        "exact_induction_parity": by_name["04_native_exact_induction"][
            "exact_induction_parity"
        ],
        "raw_examples": by_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run examples.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Summary JSON output path.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-example timeout in seconds.")
    args = parser.parse_args(argv)

    results = [_load_example_result(example, args.python, args.timeout) for example in EXAMPLES]
    summary = _aggregate(results)
    if summary["example_count"] != 5:
        raise SystemExit(f"Expected example_count=5, got {summary['example_count']}")
    if any(item["status"] != "PASS" for item in summary["per_example"]):
        raise SystemExit(f"Expected all per-example statuses PASS: {summary['per_example']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
