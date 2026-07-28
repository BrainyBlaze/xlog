from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE_STUB = ROOT / "crates" / "pyxlog" / "python" / "pyxlog" / "_native.pyi"
PACKAGE_STUB = ROOT / "crates" / "pyxlog" / "python" / "pyxlog" / "__init__.pyi"


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_joint_carrier_stub_exposes_consumer_stream_registration() -> None:
    methods = _class_methods(NATIVE_STUB, "JointConstraintCarrier")
    assert "note_producer_stream" in methods
    assert "note_consumer_stream" in methods

    package_source = PACKAGE_STUB.read_text(encoding="utf-8")
    assert "JointConstraintCarrier as JointConstraintCarrier" in package_source
