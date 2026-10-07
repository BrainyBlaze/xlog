import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_source_only_import_keeps_relation_evidence_types_importable():
    source_package = ROOT / "crates/pyxlog/python"
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class RejectNativeExtension(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pyxlog._native":
                    raise ModuleNotFoundError(
                        "forced source-only import",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, RejectNativeExtension())

        import pyxlog
        from pyxlog import RelationEvidence, RelationMetadataError

        assert pyxlog._NATIVE_AVAILABLE is False
        assert pyxlog.__all__.count("RelationEvidence") == 1
        assert pyxlog.__all__.count("RelationMetadataError") == 1
        assert issubclass(RelationMetadataError, ValueError)

        try:
            RelationEvidence()
        except RuntimeError as exc:
            assert str(exc) == "pyxlog._native is not available"
        else:
            raise AssertionError("source-only RelationEvidence must fail on use")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_package)
    env["PYTHONNOUSERSITE"] = "1"

    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_native_relation_provenance_contract_is_bound_to_runtime():
    native = pytest.importorskip("pyxlog._native")
    pyxlog = pytest.importorskip("pyxlog")

    assert issubclass(native.RelationMetadataError, ValueError)
    assert pyxlog.RelationEvidence is native.RelationEvidence
    assert pyxlog.RelationMetadataError is native.RelationMetadataError
    assert "provenance" in vars(native.RelationEvidence)

    expected_session_methods = {
        "put_relation_with_provenance",
        "put_relation_from_manifest",
        "relation",
        "evidence",
        "export_relation_with_provenance",
    }
    assert expected_session_methods <= vars(native.LogicRelationSession).keys()
