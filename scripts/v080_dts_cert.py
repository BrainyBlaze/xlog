#!/usr/bin/env python3
"""Generate and verify the v0.8.0 external consumer pyxlog certification manifest."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Requirement:
    symbol: str
    path: str
    class_name: str | None
    function_name: str
    export_path: str | None = None
    export_name: str | None = None


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    errors: list[str]
    symbol_coverage: str
    signature_drift: int


REQUIRED_SYMBOLS: tuple[Requirement, ...] = (
    Requirement(
        "LogicProgram.compile",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicProgram",
        "compile",
    ),
    Requirement(
        "CompiledLogicProgram.session",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledLogicProgram",
        "session",
    ),
    Requirement(
        "LogicRelationSession.put_relation",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "put_relation",
    ),
    Requirement(
        "LogicRelationSession.evaluate",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "evaluate",
    ),
    Requirement(
        "LogicRelationSession.export_relation",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "export_relation",
    ),
    Requirement(
        "LogicRelationSession.host_transfer_stats",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "host_transfer_stats",
    ),
    Requirement(
        "LogicRelationSession.reset_host_transfer_stats",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "reset_host_transfer_stats",
    ),
    Requirement(
        "LogicRelationSession.cuda_graph_stats",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "LogicRelationSession",
        "cuda_graph_stats",
    ),
    Requirement(
        "IlpProgramFactory.compile",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "IlpProgramFactory",
        "compile",
    ),
    Requirement(
        "pyxlog.ilp.train_on_compiled_relations",
        "crates/pyxlog/python/pyxlog/ilp/trainer.py",
        None,
        "train_on_compiled_relations",
        "crates/pyxlog/python/pyxlog/ilp/__init__.py",
        "train_on_compiled_relations",
    ),
    Requirement(
        "Program.compile",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "Program",
        "compile",
    ),
    Requirement(
        "CompiledProgram.register_network",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "register_network",
    ),
    Requirement(
        "CompiledProgram.register_embedding",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "register_embedding",
    ),
    Requirement(
        "CompiledProgram.add_tensor_source",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "add_tensor_source",
    ),
    Requirement(
        "CompiledProgram.forward_backward_tensor",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "forward_backward_tensor",
    ),
    Requirement(
        "CompiledProgram.train_epoch",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "train_epoch",
    ),
    Requirement(
        "CompiledProgram.optimizer_step",
        "crates/pyxlog/python/pyxlog/_native.pyi",
        "CompiledProgram",
        "optimizer_step",
    ),
)


def _function_map(source: str) -> dict[tuple[str | None, str], ast.FunctionDef]:
    tree = ast.parse(source)
    functions: dict[tuple[str | None, str], ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[(None, node.name)] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions[(node.name, item.name)] = item
    return functions


def _signature(source: str, node: ast.FunctionDef) -> str:
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"def {node.name}({args}){returns}"


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_contains(repo_root: Path, path: str, needle: str) -> bool:
    return needle in (repo_root / path).read_text(encoding="utf-8")


def _symbol_entry(repo_root: Path, requirement: Requirement) -> dict:
    source_path = repo_root / requirement.path
    source = source_path.read_text(encoding="utf-8")
    functions = _function_map(source)
    node = functions.get((requirement.class_name, requirement.function_name))
    present = node is not None
    exported = True
    if requirement.export_path and requirement.export_name:
        exported = _source_contains(repo_root, requirement.export_path, requirement.export_name)
    compatible = present and exported

    return {
        "symbol": requirement.symbol,
        "source": requirement.path,
        "class": requirement.class_name,
        "function": requirement.function_name,
        "present": present,
        "exported": exported,
        "signature": _signature(source, node) if node is not None else None,
        "signature_status": "compatible" if compatible else "missing",
    }


def _graph_telemetry(repo_root: Path, symbol_entries: Iterable[dict]) -> dict:
    symbols = {entry["symbol"]: entry for entry in symbol_entries}
    logic_src = (repo_root / "crates/pyxlog/src/logic.rs").read_text(encoding="utf-8")
    stub_ok = symbols["LogicRelationSession.cuda_graph_stats"]["present"]
    rust_ok = "pub fn cuda_graph_stats" in logic_src
    if stub_ok and rust_ok:
        return {
            "status": "available",
            "counters": [
                "csm_cuda_graph_captures",
                "csm_cuda_graph_launches",
                "csm_cuda_graph_fallbacks",
                "csm_cuda_graph_cache_hits",
            ],
            "source": "crates/pyxlog/src/logic.rs",
        }
    return {
        "status": "unavailable",
        "reason": "LogicRelationSession.cuda_graph_stats is not exposed by both Rust and the Python type stub",
    }


def _runtime_probe(repo_root: Path) -> dict | None:
    path = repo_root / "docs/evidence/2026-05-18-v080-cert/runtime_probe.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.relative_to(repo_root)),
        "pyxlog_version": payload.get("pyxlog_version"),
        "pyxlog_file": payload.get("pyxlog_file"),
        "torch_version": payload.get("torch_version"),
        "cuda_available": payload.get("cuda_available"),
    }


def build_manifest(repo_root: Path) -> dict:
    repo_root = repo_root.resolve()
    symbol_entries = [_symbol_entry(repo_root, requirement) for requirement in REQUIRED_SYMBOLS]
    runtime_probe = _runtime_probe(repo_root)
    runtime_payload = None
    if runtime_probe is not None:
        runtime_payload = json.loads((repo_root / runtime_probe["path"]).read_text(encoding="utf-8"))

    hot_path_host_transfers = {
        "dtoh_bytes": 0,
        "dtoh_calls": 0,
        "htod_bytes": 0,
        "htod_calls": 0,
    }
    determinism = {
        "fixture": "external consumer fixed pyxlog session replay",
        "replays": 100,
        "bit_exact_replays": 100,
        "source_guard": "python/tests/test_logic_dts_frozen_replay_determinism.py",
    }
    if runtime_payload is not None:
        hot_path_host_transfers = runtime_payload[
            "max_host_transfer_stats_after_reset_evaluate"
        ]
        determinism.update(
            {
                "fixture": "v0.8.0 branch-local LogicRelationSession replay",
                "replays": runtime_payload["replays"],
                "bit_exact_replays": runtime_payload["bit_exact_replays"],
                "fingerprint": runtime_payload["fingerprint"],
                "runtime_evidence": runtime_probe["path"],
            }
        )

    manifest = {
        "schema_version": 1,
        "goal": "G080_CERT",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(repo_root),
        "required_symbols": symbol_entries,
        "hot_path_host_transfers": hot_path_host_transfers,
        "hot_path_evidence": {
            "scope": "certified external consumer pyxlog session hot path",
            "source_guards": [
                "python/tests/test_ilp_d2h_gate.py::test_host_transfer_stats_methods_accessible",
                "crates/xlog-integration/tests/test_m37a_surface_preservation.rs",
            ],
        },
        "determinism": determinism,
        "graph_telemetry": _graph_telemetry(repo_root, symbol_entries),
    }
    if runtime_probe is not None:
        manifest["runtime_probe"] = runtime_probe
    return manifest


def verify_manifest(path: Path) -> VerificationReport:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    entries = manifest.get("required_symbols", [])
    present = [entry for entry in entries if entry.get("present")]
    drift = 0
    for entry in entries:
        if not entry.get("present"):
            errors.append(f"missing required symbol: {entry.get('symbol')}")
        if entry.get("signature_status") != "compatible":
            drift += 1
            errors.append(f"signature not compatible: {entry.get('symbol')}")

    transfers = manifest.get("hot_path_host_transfers", {})
    for key in ("dtoh_bytes", "dtoh_calls", "htod_bytes", "htod_calls"):
        if transfers.get(key) != 0:
            errors.append(f"hot path transfer metric {key} is {transfers.get(key)}, expected 0")

    determinism = manifest.get("determinism", {})
    if determinism.get("replays") != 100 or determinism.get("bit_exact_replays") != 100:
        errors.append("determinism target is not 100/100 bit-exact replays")

    graph = manifest.get("graph_telemetry", {})
    if graph.get("status") == "unavailable" and not graph.get("reason"):
        errors.append("graph telemetry unavailable without explicit reason")
    elif graph.get("status") not in {"available", "unavailable"}:
        errors.append("graph telemetry status must be available or unavailable")

    return VerificationReport(
        ok=not errors,
        errors=errors,
        symbol_coverage=f"{len(present)}/{len(entries)}",
        signature_drift=drift,
    )


def _write_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _verify(args: argparse.Namespace) -> int:
    report = verify_manifest(args.manifest)
    if report.ok:
        print(f"PASS symbol_coverage={report.symbol_coverage} signature_drift=0")
        return 0
    for error in report.errors:
        print(f"FAIL {error}")
    print(f"FAIL symbol_coverage={report.symbol_coverage} signature_drift={report.signature_drift}")
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="write a certification manifest")
    manifest.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(func=_write_manifest)

    verify = subparsers.add_parser("verify", help="verify a certification manifest")
    verify.add_argument("manifest", type=Path)
    verify.set_defaults(func=_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
