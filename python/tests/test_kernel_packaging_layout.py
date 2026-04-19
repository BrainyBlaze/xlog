from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repo_does_not_require_tracked_ptx_files() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--", "kernels/*.ptx"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], f"tracked PTX files should be removed: {tracked}"


def test_stage_kernels_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/stage_kernels.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage:" in result.stdout.lower()
    assert "--from-out-dir" in result.stdout
    assert "--to" in result.stdout


def test_stage_kernels_prunes_and_emits_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        from_out_dir = tmp_root / "out"
        to_dir = tmp_root / "kernels"
        from_out_dir.mkdir()
        to_dir.mkdir()

        source_files = {
            "join.portable.ptx": b"join-ptx",
            "join.sm_75.cubin": b"join-cubin",
            "sort.portable.ptx": b"sort-ptx",
        }
        for name, content in source_files.items():
            (from_out_dir / name).write_bytes(content)

        stale_files = {
            "obsolete.portable.ptx": b"old-ptx",
            "obsolete.sm_75.cubin": b"old-cubin",
        }
        for name, content in stale_files.items():
            (to_dir / name).write_bytes(content)
        keep_file = to_dir / "notes.txt"
        keep_file.write_text("keep me")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/stage_kernels.py",
                "--from-out-dir",
                str(from_out_dir),
                "--to",
                str(to_dir),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr or result.stdout

        expected_lines = []
        for name in sorted(source_files):
            content = source_files[name]
            expected_lines.append(
                f"{name}\t{hashlib.sha256(content).hexdigest()}\t{len(content)}"
            )

        assert result.stdout.splitlines() == expected_lines
        assert (to_dir / "join.portable.ptx").read_bytes() == source_files["join.portable.ptx"]
        assert (to_dir / "join.sm_75.cubin").read_bytes() == source_files["join.sm_75.cubin"]
        assert (to_dir / "sort.portable.ptx").read_bytes() == source_files["sort.portable.ptx"]
        assert not (to_dir / "obsolete.portable.ptx").exists()
        assert not (to_dir / "obsolete.sm_75.cubin").exists()
        assert keep_file.exists()
