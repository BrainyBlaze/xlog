from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def exact_induction_evidence_dir() -> Path:
    matches = sorted(
        path
        for path in (ROOT / "docs" / "evidence").glob("*-exact")
        if (path / "runtime_probe.json").exists() and (path / "README.md").exists()
    )
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = exact_induction_evidence_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_exact_evidence_records_runtime_and_downstream_gates() -> None:
    probe = json.loads((EVIDENCE_DIR / "runtime_probe.json").read_text(encoding="utf-8"))

    assert probe["native_parity_with_strict_python"] is True
    assert probe["d2h_counter"]["within_gate"] is True
    assert probe["d2h_counter"]["small_dtoh_calls"] == 2
    assert probe["d2h_counter"]["large_dtoh_calls"] == 2
    assert "ilp_exact.portable.ptx" in probe["packaged_ilp_exact_kernels"]

    consumer_liveness_entries = [
        value for key, value in probe.items() if key.endswith("_liveness_evidence")
    ]
    assert len(consumer_liveness_entries) == 1
    consumer_liveness = consumer_liveness_entries[0]
    assert consumer_liveness["exists"] is True
    assert consumer_liveness["both_heads_alive_449_449"] is True
    assert consumer_liveness["rollback_rate_zero"] is True
    assert consumer_liveness["quarantine_rate_zero"] is True


def test_exact_source_contracts_are_current() -> None:
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


def test_exact_readme_links_public_consumer_and_packaging_policy() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")

    assert "imports public `pyxlog.ilp.exact_induce.induce_exact`" in readme
    assert 'backend="native"' in readme
    assert "449/449" in readme
    assert "rollback_rate=0.0" in readme
    assert "quarantine_rate=0.0" in readme
    assert "no checked-in generated PTX" in readme
