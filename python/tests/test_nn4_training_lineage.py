from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xlog_nn4_004_registration_lineage_and_influence_audit_surface() -> None:
    init_py = (ROOT / "crates/pyxlog/python/pyxlog/__init__.py").read_text()
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

    for needle in [
        "checkpoint_hash",
        "split_hashes",
        "calibration_metrics",
        "influence_audit",
        "nn4_lineage",
        "record_nn4_influence",
        "changed_acceptance",
        "cuda_device",
    ]:
        assert needle in init_py
        assert needle in native_stub
        assert needle in docs
