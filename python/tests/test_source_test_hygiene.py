import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SUFFIXES = (".rs", ".pyi")


def _contains_forbidden_source_path(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and child.value.endswith(FORBIDDEN_SUFFIXES)
        for child in ast.walk(node)
    )


def _assigned_forbidden_paths(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if node.value is None:
                continue
            value_names = {
                child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)
            }
            if not (
                _contains_forbidden_source_path(node.value) or value_names & names
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


def _references_forbidden_path(node: ast.AST, forbidden_names: set[str]) -> bool:
    return _contains_forbidden_source_path(node) or any(
        isinstance(child, ast.Name) and child.id in forbidden_names
        for child in ast.walk(node)
    )


def _forbidden_reads_in_tree(tree: ast.AST) -> list[int]:
    forbidden_names = _assigned_forbidden_paths(tree)
    lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "open",
            "read_bytes",
            "read_text",
        }:
            if _references_forbidden_path(node.func.value, forbidden_names):
                lines.append(node.lineno)
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in {"open", "read"}
            and any(
                _references_forbidden_path(argument, forbidden_names)
                for argument in node.args
            )
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _forbidden_reads(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _forbidden_reads_in_tree(tree)


def test_source_read_detector_covers_path_aliases_and_open_variants() -> None:
    source = """
from pathlib import Path
rust_source = Path("crate.rs")
alias = rust_source
alias.read_text()
Path("binding.pyi").read_bytes()
Path("other.rs").open()
open(Path("native.pyi"))
read("module.rs")
Path("guide.md").read_text()
"""

    assert _forbidden_reads_in_tree(ast.parse(source)) == [5, 6, 7, 8, 9]


def test_python_tests_do_not_inspect_rust_or_stub_source_text() -> None:
    violations = {
        str(path.relative_to(ROOT)): lines
        for path in sorted((ROOT / "python" / "tests").rglob("*.py"))
        if (lines := _forbidden_reads(path))
    }
    assert not violations, (
        "Python tests must verify runtime behavior instead of reading Rust or stub "
        f"implementation text: {violations}"
    )
