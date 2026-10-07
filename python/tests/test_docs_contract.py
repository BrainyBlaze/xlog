from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_static_markdown_export_generator_writes_route_markdown(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/docs/build_markdown_exports.py",
            "docs",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    index = tmp_path / "index.md"
    gpu = tmp_path / "architecture/gpu-execution.md"
    assert index.exists()
    assert gpu.exists()
    assert index.read_text(encoding="utf-8").startswith("# XLOG Documentation\n\n")
    gpu_text = gpu.read_text(encoding="utf-8")
    assert gpu_text.startswith("# GPU Execution\n\n")
    assert "XLOG's deterministic runtime" in gpu_text
    assert "title:" not in gpu_text.splitlines()[:5]


def test_pyxlog_stub_generator_extracts_classes_and_methods() -> None:
    sample = '''
class LogicProgram:
    """Factory."""

    @staticmethod
    def compile(source: str, device: int = 0) -> CompiledLogicProgram: ...
'''
    result = subprocess.run(
        [sys.executable, "scripts/docs/gen_pyxlog_api.py", "--stdin"],
        cwd=ROOT,
        input=sample,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "## LogicProgram" in result.stdout
    assert "compile(source: str, device: int = 0)" in result.stdout
