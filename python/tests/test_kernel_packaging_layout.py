from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYXLOG_ROOT = ROOT / "crates" / "pyxlog"
PYXLOG_PACKAGE_ROOT = PYXLOG_ROOT / "python" / "pyxlog"


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


def test_pyxlog_pyproject_includes_generated_kernels_in_wheel() -> None:
    pyproject = (PYXLOG_ROOT / "pyproject.toml").read_text()

    assert 'crate_name = "xlog-cuda"' in pyproject
    assert 'from = "out-dir"' in pyproject
    assert 'to = "pyxlog/kernels/"' in pyproject
    assert 'path = "*.portable.ptx"' in pyproject
    assert 'path = "*.cubin"' in pyproject


def test_pyxlog_kernel_path_helper_prefers_packaged_kernels() -> None:
    helper_path = PYXLOG_PACKAGE_ROOT / "_kernel_paths.py"
    spec = importlib.util.spec_from_file_location("pyxlog_kernel_paths_test", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        package_root = Path(tmp) / "pyxlog"
        kernels_dir = package_root / "kernels"
        kernels_dir.mkdir(parents=True)
        (kernels_dir / "join.portable.ptx").write_text("ptx")

        original = os.environ.get("XLOG_CUBIN_DIR")
        try:
            os.environ.pop("XLOG_CUBIN_DIR", None)
            configured = module.configure_kernel_search_path(package_root)
            assert configured == str(kernels_dir)
            assert os.environ["XLOG_CUBIN_DIR"] == str(kernels_dir)

            override_dir = package_root / "override"
            override_dir.mkdir()
            os.environ["XLOG_CUBIN_DIR"] = str(override_dir)
            configured = module.configure_kernel_search_path(package_root)
            assert configured == str(override_dir)
            assert os.environ["XLOG_CUBIN_DIR"] == str(override_dir)
        finally:
            if original is None:
                os.environ.pop("XLOG_CUBIN_DIR", None)
            else:
                os.environ["XLOG_CUBIN_DIR"] = original


def test_package_cli_release_help_works() -> None:
    result = subprocess.run(
        ["bash", "scripts/package_cli_release.sh", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage:" in result.stdout.lower()
    assert "--output" in result.stdout


def test_package_cli_release_stages_layout_and_tarball() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        target_dir = tmp_root / "target"
        release_dir = target_dir / "release"
        build_out_dir = release_dir / "build" / "xlog-cuda-test" / "out"
        binary_path = release_dir / "xlog"
        output_dir = tmp_root / "dist"

        build_out_dir.mkdir(parents=True)
        release_dir.mkdir(parents=True, exist_ok=True)
        binary_path.write_text("#!/usr/bin/env bash\nexit 0\n")
        binary_path.chmod(0o755)

        (build_out_dir / "join.portable.ptx").write_text("join-ptx")
        (build_out_dir / "join.sm_75.cubin").write_text("join-cubin")

        env = os.environ.copy()
        env.update(
            {
                "XLOG_PACKAGE_SKIP_BUILD": "1",
                "XLOG_PACKAGE_TARGET_DIR": str(target_dir),
                "XLOG_PACKAGE_BINARY_PATH": str(binary_path),
                "XLOG_PACKAGE_KERNEL_OUT_DIR": str(build_out_dir),
                "XLOG_PACKAGE_VERSION": "9.9.9",
                "XLOG_PACKAGE_HOST_TRIPLE": "x86_64-unknown-linux-gnu",
            }
        )

        result = subprocess.run(
            ["bash", "scripts/package_cli_release.sh", "--output", str(output_dir)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr or result.stdout

        bundle_root = output_dir / "xlog-v9.9.9-x86_64-unknown-linux-gnu"
        tarball_path = output_dir / "xlog-v9.9.9-x86_64-unknown-linux-gnu.tar.gz"
        assert (bundle_root / "xlog").is_file()
        assert (bundle_root / "kernels" / "join.portable.ptx").read_text() == "join-ptx"
        assert (bundle_root / "kernels" / "join.sm_75.cubin").read_text() == "join-cubin"
        assert (bundle_root / "README.md").is_file()
        assert (bundle_root / "LICENSE-APACHE").is_file()
        assert (bundle_root / "LICENSE-MIT").is_file()
        assert tarball_path.is_file()

        with tarfile.open(tarball_path, "r:gz") as archive:
            names = set(archive.getnames())

        assert "xlog-v9.9.9-x86_64-unknown-linux-gnu/xlog" in names
        assert "xlog-v9.9.9-x86_64-unknown-linux-gnu/kernels/join.portable.ptx" in names
        assert "xlog-v9.9.9-x86_64-unknown-linux-gnu/kernels/join.sm_75.cubin" in names
