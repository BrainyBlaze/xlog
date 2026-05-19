from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "2026-05-18-v080-exact"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v080_exact_evidence_records_runtime_and_downstream_gates() -> None:
    probe = json.loads((EVIDENCE_DIR / "runtime_probe.json").read_text(encoding="utf-8"))

    assert probe["native_parity_with_strict_python"] is True
    assert probe["d2h_counter"]["within_gate"] is True
    assert probe["d2h_counter"]["small_dtoh_calls"] == 2
    assert probe["d2h_counter"]["large_dtoh_calls"] == 2
    assert "ilp_exact.portable.ptx" in probe["packaged_ilp_exact_kernels"]

    dts = probe["dts_phase1_liveness_evidence"]
    assert dts["exists"] is True
    assert dts["both_heads_alive_449_449"] is True
    assert dts["rollback_rate_zero"] is True
    assert dts["quarantine_rate_zero"] is True


def test_v080_exact_source_contracts_are_current() -> None:
    py_bridge = read("crates/pyxlog/src/ilp_exact.rs")
    parity_test = read("python/tests/test_ilp_exact_induce.py")
    engine = read("crates/xlog-induce/src/lib.rs")
    arch = read("docs/architecture/bounded-exact-induction.md")
    py_docs = read("docs/architecture/python-bindings.md")
    manifest = read("crates/xlog-cuda/src/kernel_manifest_data.rs")
    installer = read("scripts/install_pyxlog_for_python.py")

    assert "scoring kernel isn't wired yet" not in py_bridge
    assert "red until native lands" not in parity_test
    assert "provider.ilp_exact_score" in engine
    assert "strict_per_topology=True" in py_docs

    assert "ScalarType::U64 => ExactPairType::U64" in engine
    assert "ScalarType::U32 => ExactPairType::U32" in engine
    assert "ScalarType::Symbol => ExactPairType::Symbol" in engine
    assert "`U32`: supported through `ilp_exact_score_u32`" in arch
    assert "`Symbol`: supported through the same physical `u32` kernel" in arch

    assert '"ilp_exact"' in manifest
    assert '"ilp_exact_score"' in manifest
    assert '"ilp_exact_score_u32"' in manifest
    assert "portable PTX" in installer
    assert "does not contain portable PTX kernel artifacts" in installer


def test_v080_exact_readme_links_public_consumer_and_packaging_policy() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")

    assert "dts_dlm.integrations.pyxlog.tensorized_ilp" in readme
    assert 'backend="native"' in readme
    assert "449/449" in readme
    assert "rollback_rate=0.0" in readme
    assert "quarantine_rate=0.0" in readme
    assert "no checked-in generated PTX" in readme
