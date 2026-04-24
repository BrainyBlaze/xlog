#!/usr/bin/env python3
"""Build and install local pyxlog for one explicit Python interpreter."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYXLOG_MANIFEST = ROOT / "crates" / "pyxlog" / "Cargo.toml"
DEFAULT_WHEEL_DIR = ROOT / "target" / "pyxlog-wheels"

VERIFY_IMPORT_SNIPPET = r"""
import pathlib
import pyxlog

package_dir = pathlib.Path(pyxlog.__file__).resolve().parent
kernels_dir = package_dir / "kernels"
if not kernels_dir.is_dir():
    raise SystemExit(f"pyxlog kernels directory is missing: {kernels_dir}")
if not any(kernels_dir.glob("*.portable.ptx")):
    raise SystemExit(f"pyxlog kernels directory has no portable PTX: {kernels_dir}")
print(f"pyxlog package: {package_dir}")
print(f"pyxlog kernels: {kernels_dir}")
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage CUDA kernels, run `maturin build` for one explicit Python "
            "interpreter, install the resulting pyxlog wheel into that same "
            "interpreter, and verify the installed package contains kernels."
        ),
        epilog=(
            "Example: python scripts/install_pyxlog_for_python.py --python "
            "/usr/local/bin/python --user"
        ),
    )
    parser.add_argument(
        "--python",
        required=True,
        help=(
            "Target Python executable that will receive pyxlog. Use the exact "
            "interpreter your downstream project uses."
        ),
    )
    parser.add_argument(
        "--wheel-dir",
        default=str(DEFAULT_WHEEL_DIR),
        help="Directory for the built wheel. Existing pyxlog wheels there are pruned.",
    )
    parser.add_argument(
        "--compatibility",
        default="linux",
        help="maturin wheel compatibility tag for local installs (default: linux).",
    )
    parser.add_argument(
        "--features",
        help="Optional additional Cargo features to pass to maturin build.",
    )
    parser.add_argument(
        "--maturin",
        default=None,
        help="Path to a maturin executable. Defaults to maturin on PATH.",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="Pass --user to pip install for user-site installs.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip install.",
    )
    parser.add_argument(
        "--skip-stage",
        action="store_true",
        help="Do not restage kernels before building the wheel.",
    )
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Do not import pyxlog after installation to verify packaged kernels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without running them.",
    )
    return parser.parse_args(argv)


def quote_cmd(command: list[str | Path]) -> str:
    return shlex.join(str(part) for part in command)


def run(command: list[str | Path], *, dry_run: bool) -> None:
    print(f"+ {quote_cmd(command)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def maturin_command(args: argparse.Namespace) -> list[str]:
    if args.maturin is not None:
        return [args.maturin]
    if shutil.which("maturin") is not None:
        return ["maturin"]
    return [args.python, "-m", "maturin"]


def prune_existing_pyxlog_wheels(wheel_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    wheel_dir.mkdir(parents=True, exist_ok=True)
    for wheel in wheel_dir.glob("pyxlog-*.whl"):
        wheel.unlink()


def built_wheel(wheel_dir: Path) -> Path:
    wheels = sorted(wheel_dir.glob("pyxlog-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected exactly one pyxlog wheel in {wheel_dir}, found {len(wheels)}."
        )
    return wheels[0]


def validate_wheel_contains_kernels(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    kernel_names = [
        name
        for name in names
        if name.startswith("pyxlog/kernels/") and not name.endswith("/")
    ]
    if not kernel_names:
        raise SystemExit(f"{wheel} does not contain pyxlog/kernels/ artifacts.")
    if not any(name.endswith(".portable.ptx") for name in kernel_names):
        raise SystemExit(f"{wheel} does not contain portable PTX kernel artifacts.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wheel_dir = Path(args.wheel_dir).resolve()
    wheel_placeholder = wheel_dir / "pyxlog-<built>.whl"

    run([args.python, "-c", "import sys; print(sys.executable)"], dry_run=args.dry_run)

    if not args.skip_stage:
        run(["bash", "scripts/stage_pyxlog_kernels.sh"], dry_run=args.dry_run)

    prune_existing_pyxlog_wheels(wheel_dir, dry_run=args.dry_run)

    build_cmd: list[str | Path] = [
        *maturin_command(args),
        "build",
        "-m",
        PYXLOG_MANIFEST,
        "--release",
        "--locked",
        "--compatibility",
        args.compatibility,
        "--out",
        wheel_dir,
        "-i",
        args.python,
    ]
    if args.features:
        build_cmd.extend(["--features", args.features])
    run(build_cmd, dry_run=args.dry_run)

    wheel = wheel_placeholder if args.dry_run else built_wheel(wheel_dir)
    if not args.dry_run:
        validate_wheel_contains_kernels(wheel)

    install_cmd: list[str | Path] = [
        args.python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
    ]
    if args.user:
        install_cmd.append("--user")
    if args.no_deps:
        install_cmd.append("--no-deps")
    install_cmd.append(wheel)
    run(install_cmd, dry_run=args.dry_run)

    if not args.skip_import_check:
        run([args.python, "-c", VERIFY_IMPORT_SNIPPET], dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
