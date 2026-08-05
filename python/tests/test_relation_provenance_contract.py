import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_native_relation_provenance_contract_stays_aligned_across_public_surfaces():
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    package_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    package_runtime = (ROOT / "crates/pyxlog/python/pyxlog/__init__.py").read_text()
    docs = [
        (ROOT / path).read_text()
        for path in (
            "docs/reference/python.mdx",
            "docs/guides/diagnostics.mdx",
            "docs/guides/interop.mdx",
            "python/tests/contract_docs/python-bindings.md",
        )
    ]

    native_tree = ast.parse(native_stub)
    native_classes = {
        node.name: node for node in native_tree.body if isinstance(node, ast.ClassDef)
    }
    metadata_error = native_classes["RelationMetadataError"]
    assert [base.id for base in metadata_error.bases if isinstance(base, ast.Name)] == [
        "ValueError"
    ]
    evidence_methods = {
        node.name
        for node in native_classes["RelationEvidence"].body
        if isinstance(node, ast.FunctionDef)
    }
    assert "provenance" in evidence_methods

    session_methods = {
        node.name
        for node in native_classes["LogicRelationSession"].body
        if isinstance(node, ast.FunctionDef)
    }
    expected_session_methods = {
        "put_relation_with_provenance",
        "put_relation_from_manifest",
        "relation",
        "evidence",
        "export_relation_with_provenance",
    }
    assert expected_session_methods <= session_methods

    package_tree = ast.parse(package_stub)
    native_reexports = {
        alias.asname or alias.name
        for node in ast.walk(package_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "pyxlog._native"
        for alias in node.names
    }
    assert {
        "LogicRelationSession",
        "RelationEvidence",
        "RelationMetadataError",
    } <= native_reexports

    runtime_tree = ast.parse(package_runtime)
    native_aliases = {
        target.id: node.value.attr
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "_native"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert native_aliases["RelationEvidence"] == "RelationEvidence"
    assert native_aliases["RelationMetadataError"] == "RelationMetadataError"

    for document in docs:
        assert "RelationEvidence" in document
        assert "RelationMetadataError" in document
        assert "KeyError" in document

    for method_name in (
        "put_relation_with_provenance",
        "put_relation_from_manifest",
        "export_relation_with_provenance",
    ):
        assert method_name in docs[0]
        assert method_name in docs[3]

    for method_name in (
        "put_relation_with_provenance",
        "export_relation_with_provenance",
    ):
        assert method_name in docs[1]
        assert method_name in docs[2]
    assert "put_relation_from_manifest" in docs[2]

    for removed_sidecar in (
        "_RELATION_EVIDENCE",
        "_record_relation_evidence",
        "_logic_session_put_relation_with_provenance",
    ):
        assert removed_sidecar not in package_runtime
