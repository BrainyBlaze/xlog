import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import v080_dts_cert as cert


ROOT = Path(__file__).resolve().parents[2]


def _symbol_statuses(manifest: dict) -> dict[str, dict]:
    return {entry["symbol"]: entry for entry in manifest["required_symbols"]}


def test_v080_manifest_covers_required_dts_pyxlog_surface() -> None:
    manifest = cert.build_manifest(ROOT)
    symbols = _symbol_statuses(manifest)

    required = {
        "LogicProgram.compile",
        "CompiledLogicProgram.session",
        "LogicRelationSession.put_relation",
        "LogicRelationSession.evaluate",
        "LogicRelationSession.export_relation",
        "LogicRelationSession.host_transfer_stats",
        "LogicRelationSession.reset_host_transfer_stats",
        "LogicRelationSession.cuda_graph_stats",
        "IlpProgramFactory.compile",
        "pyxlog.ilp.train_on_compiled_relations",
        "Program.compile",
        "CompiledProgram.register_network",
        "CompiledProgram.register_embedding",
        "CompiledProgram.add_tensor_source",
        "CompiledProgram.forward_backward_tensor",
        "CompiledProgram.train_epoch",
        "CompiledProgram.optimizer_step",
    }

    assert set(symbols) == required
    assert all(entry["present"] for entry in symbols.values())
    assert all(entry["signature_status"] == "compatible" for entry in symbols.values())


def test_v080_manifest_records_certification_metric_evidence() -> None:
    manifest = cert.build_manifest(ROOT)

    assert manifest["hot_path_host_transfers"] == {
        "dtoh_bytes": 0,
        "dtoh_calls": 0,
        "htod_bytes": 0,
        "htod_calls": 0,
    }
    assert manifest["determinism"]["replays"] == 100
    assert manifest["determinism"]["bit_exact_replays"] == 100
    runtime_probe = ROOT / "docs/evidence/2026-05-18-v080-cert/runtime_probe.json"
    if runtime_probe.exists():
        assert manifest["runtime_probe"]["path"] == str(
            runtime_probe.relative_to(ROOT)
        )
        assert manifest["runtime_probe"]["pyxlog_version"] == "0.8.0"
    assert manifest["graph_telemetry"]["status"] in {"available", "unavailable"}
    if manifest["graph_telemetry"]["status"] == "unavailable":
        assert manifest["graph_telemetry"]["reason"]


def test_v080_manifest_round_trips_through_verifier(tmp_path: Path) -> None:
    manifest = cert.build_manifest(ROOT)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report = cert.verify_manifest(path)

    assert report.ok, report.errors
    assert report.symbol_coverage == "17/17"
    assert report.signature_drift == 0
