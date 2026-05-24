from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xlog_pyxlog_003_session_evidence_and_relation_provenance_api() -> None:
    init_py = (ROOT / "crates/pyxlog/python/pyxlog/__init__.py").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

    for needle in [
        "put_relation_with_provenance",
        "def evidence(",
        "def relation(",
        "class RelationEvidence",
        "def provenance(",
        "program_hash",
        "source_hash",
        "accepted_count",
        "rejected_count",
        "output_hash",
    ]:
        assert needle in init_py
        assert needle in init_stub
        assert needle in docs
