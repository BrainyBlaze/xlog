#!/usr/bin/env python3
"""Regenerate unpublished package notes without changing source Git metadata."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-date", help="Release-note date (YYYY-MM-DD).")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    head = git(repository, "rev-parse", "HEAD")
    branch = git(repository, "symbolic-ref", "--short", "HEAD")
    git(repository, "diff", "--quiet", "HEAD", "--", "Cargo.toml", "Cargo.lock", "crates")
    manifest = tomllib.loads((repository / "Cargo.toml").read_text())
    config_path = repository / "release-plz.toml"
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    version = manifest["workspace"]["package"]["version"]
    packages = config["package"]
    (tag_owner,) = [package for package in packages if package.get("git_tag_enable")]
    tag = tag_owner["git_tag_name"].replace("{{ version }}", version)
    local_tag = git(repository, "tag", "--list", tag)
    remote_tag = git(repository, "ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    if local_tag or remote_tag:
        raise RuntimeError(f"Release tag {tag} exists; published notes are immutable.")
    changelog_packages = {
        package["name"] for package in packages
        if package.get("release") and package.get("changelog_update", True)
    }
    changelog_path = repository / config["workspace"]["changelog_path"]
    original = changelog_path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)(?=^## )", original)
    unreleased, = [part for part in parts if part.startswith("## [Unreleased]\n")]
    retained = [
        part for part in parts
        if part != unreleased and not (
            part.startswith(f"## [{version}](") and any(
                f"...{package}-v{version}) - " in part.splitlines()[0]
                for package in changelog_packages
            )
        )
    ]
    marker = "## [Unreleased]\n\n"
    baseline = retained[0] + marker + "".join(retained[1:])
    with tempfile.TemporaryDirectory(prefix="xlog-release-changelog-") as directory:
        checkout = Path(directory) / "source"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", "--branch", branch,
             str(repository), str(checkout)], check=True
        )
        git(
            checkout, "remote", "set-url", "origin",
            git(repository, "remote", "get-url", "origin")
        )
        (checkout / "release-plz.toml").write_bytes(config_bytes)
        generated_path = checkout / config["workspace"]["changelog_path"]
        generated_path.write_text(baseline, encoding="utf-8")
        command = ["release-plz", "update", "--allow-dirty"]
        if args.release_date:
            command.extend(["--release-date", args.release_date])
        subprocess.run(command, cwd=checkout, check=True)
        generated = generated_path.read_text(encoding="utf-8")
        if generated.count(marker) != 1 or not generated.endswith("".join(retained[1:])):
            raise RuntimeError("Generator changed retained release history or its boundary.")
        updated = generated.replace(marker, unreleased, 1)
    if (
        git(repository, "rev-parse", "HEAD") != head
        or config_path.read_bytes() != config_bytes
        or changelog_path.read_text(encoding="utf-8") != original
    ):
        raise RuntimeError("Release inputs changed during generation; no notes were replaced.")
    changelog_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
