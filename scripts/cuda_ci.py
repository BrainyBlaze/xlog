#!/usr/bin/env python3
"""Deterministic change classification and aggregate gating for CUDA CI."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


RELEVANT_PATH_PATTERNS = (
    ".github/workflows/cuda-ci.yml",
    "Cargo.lock",
    "Cargo.toml",
    "crates/**",
    "pytest.ini",
    "python/tests/conftest.py",
    "python/tests/test_logic_relation_provenance.py",
    "python/tests/test_pyxlog_conditioned_reuse.py",
    "python/tests/test_pyxlog_epistemic_api.py",
    "python/tests/test_pyxlog_epistemic_gpu.py",
    "python/tests/test_pyxlog_ground_term_encoding.py",
    "python/tests/test_relation_callbacks_runtime.py",
    "python/tests/test_relation_provenance_contract.py",
    "python/tests/test_relation_provenance_public_api.py",
    "python/constraints-build.txt",
    "scripts/cuda_ci.py",
    "scripts/validate_release_gpu.sh",
    "scripts/validate_reproducible_pyxlog_wheel.py",
)


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating whether the stable Python-wheel check may pass."""

    passed: bool
    message: str


def changes_are_relevant(paths: Iterable[str]) -> bool:
    """Return whether any changed repository path requires the CUDA wheel job."""

    return any(
        fnmatch.fnmatchcase(path, pattern)
        for path in paths
        for pattern in RELEVANT_PATH_PATTERNS
    )


def evaluate_python_wheel_gate(
    *,
    event_name: str,
    repository: str,
    head_repository: str,
    relevant: bool | None,
    classification_result: str,
    gpu_result: str,
) -> GateDecision:
    """Evaluate the stable Python-wheel context from prerequisite job results."""

    if classification_result != "success":
        return GateDecision(
            False,
            f"CUDA change classification did not succeed: {classification_result}",
        )
    if relevant is None:
        return GateDecision(False, "CUDA change classification produced no result")
    if not relevant:
        if gpu_result not in {"skipped", "success"}:
            return GateDecision(False, f"Unexpected GPU wheel job result: {gpu_result}")
        return GateDecision(True, "No CUDA Python-wheel inputs changed")
    if event_name == "pull_request" and head_repository != repository:
        return GateDecision(
            False,
            "Relevant CUDA Python-wheel changes from a fork require a trusted "
            "same-repository branch",
        )
    if gpu_result != "success":
        return GateDecision(
            False,
            f"Required trusted GPU wheel job did not succeed: {gpu_result}",
        )
    return GateDecision(True, "Required trusted GPU wheel job succeeded")


def _parse_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _read_paths(null_delimited: bool) -> list[str]:
    separator = b"\0" if null_delimited else b"\n"
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in sys.stdin.buffer.read().split(separator)
        if value
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser(
        "classify", help="classify changed paths from stdin"
    )
    classify.add_argument(
        "--null",
        action="store_true",
        help="read NUL-delimited paths, as emitted by git diff --name-only -z",
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="evaluate the stable Python-wheel aggregate check"
    )
    aggregate.add_argument("--event-name", required=True)
    aggregate.add_argument("--repository", required=True)
    aggregate.add_argument("--head-repository", default="")
    aggregate.add_argument("--relevant", required=True, type=_parse_bool)
    aggregate.add_argument("--classification-result", required=True)
    aggregate.add_argument("--gpu-result", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "classify":
        print(str(changes_are_relevant(_read_paths(args.null))).lower())
        return 0

    decision = evaluate_python_wheel_gate(
        event_name=args.event_name,
        repository=args.repository,
        head_repository=args.head_repository,
        relevant=args.relevant,
        classification_result=args.classification_result,
        gpu_result=args.gpu_result,
    )
    print(decision.message, file=sys.stdout if decision.passed else sys.stderr)
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
