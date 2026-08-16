#!/usr/bin/env python3
"""Remove generated sections for the current unpublished workspace version."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 local fallback
    import tomli as tomllib


RELEASE_HEADING = re.compile(
    r"^## (?:(?P<package>[A-Za-z0-9][A-Za-z0-9_-]*) )?"
    r"\[(?P<version>[^\]]+)\](?:\((?P<url>[^)]+)\))?"
    r" - (?P<date>\d{4}-\d{2}-\d{2})\s*$"
)


@dataclass(frozen=True)
class ReleaseContext:
    version: str
    changelog_packages: frozenset[str]
    authoritative_package: str
    release_tag: str


@dataclass(frozen=True)
class ReleaseSection:
    version: str
    package: str
    start: int
    end: int


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_release_context(manifest_path: Path, config_path: Path) -> ReleaseContext:
    manifest = _load_toml(manifest_path)
    config = _load_toml(config_path)
    try:
        version = manifest["workspace"]["package"]["version"]
    except KeyError as exc:
        raise ValueError(
            f"{manifest_path} must define workspace.package.version"
        ) from exc

    workspace = config.get("workspace", {})
    changelog_packages: set[str] = set()
    tag_packages: list[tuple[str, str]] = []
    for package in config.get("package", []):
        if not package.get("release", workspace.get("release", True)):
            continue
        if package.get("changelog_update", workspace.get("changelog_update", True)):
            changelog_packages.add(package["name"])
        if package.get("git_tag_enable", workspace.get("git_tag_enable", True)):
            tag_template = package.get("git_tag_name", workspace.get("git_tag_name"))
            if not tag_template:
                raise ValueError(
                    f"release tag package {package['name']} needs git_tag_name"
                )
            tag_packages.append((package["name"], tag_template))

    if not changelog_packages:
        raise ValueError(f"{config_path} enables no package changelogs")
    if len(tag_packages) != 1:
        raise ValueError(
            f"{config_path} must enable exactly one authoritative release tag; "
            f"found {len(tag_packages)}"
        )
    authoritative_package, tag_template = tag_packages[0]
    if "{{ version }}" not in tag_template:
        raise ValueError("the authoritative git_tag_name must contain {{ version }}")

    return ReleaseContext(
        version=version,
        changelog_packages=frozenset(changelog_packages),
        authoritative_package=authoritative_package,
        release_tag=tag_template.replace("{{ version }}", version),
    )


def _parse_compare_url(url: str, version: str) -> str | None:
    if "/compare/" not in url:
        return None
    comparison = url.rsplit("/compare/", 1)[1]
    if comparison.count("...") != 1:
        return None
    previous, current = comparison.split("...", 1)
    if "-v" not in previous or "-v" not in current:
        return None
    previous_package, _ = previous.rsplit("-v", 1)
    current_package, current_version = current.rsplit("-v", 1)
    if previous_package != current_package or current_version != version:
        return None
    return current_package


def _parse_release_heading(line: str) -> tuple[str, str] | None:
    match = RELEASE_HEADING.fullmatch(line.rstrip("\n"))
    if match is None:
        return None
    version = match.group("version")
    labelled_package = match.group("package")
    url = match.group("url")

    if url is None:
        if labelled_package is None:
            return None
        return version, labelled_package

    linked_package = _parse_compare_url(url, version)
    if linked_package is None:
        return None
    if labelled_package is not None and labelled_package != linked_package:
        return None
    return version, labelled_package or linked_package


def release_sections(changelog: str) -> list[ReleaseSection]:
    lines = changelog.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    heading_lines = [
        index for index, line in enumerate(lines) if line.startswith("## ")
    ]
    sections: list[ReleaseSection] = []
    for heading_index, line_index in enumerate(heading_lines):
        parsed = _parse_release_heading(lines[line_index])
        if parsed is None:
            continue
        next_line = (
            heading_lines[heading_index + 1]
            if heading_index + 1 < len(heading_lines)
            else len(lines)
        )
        end = offsets[next_line] if next_line < len(lines) else len(changelog)
        version, package = parsed
        sections.append(
            ReleaseSection(
                version=version,
                package=package,
                start=offsets[line_index],
                end=end,
            )
        )
    return sections


def remove_unpublished_release_sections(
    changelog: str,
    *,
    version: str,
    packages: set[str] | frozenset[str],
    release_tag_exists: bool,
) -> tuple[str, int]:
    if release_tag_exists:
        return changelog, 0

    stale_sections = [
        section
        for section in release_sections(changelog)
        if section.version == version and section.package in packages
    ]
    if not stale_sections:
        return changelog, 0

    pieces: list[str] = []
    cursor = 0
    for section in stale_sections:
        pieces.append(changelog[cursor : section.start])
        cursor = section.end
    pieces.append(changelog[cursor:])
    return "".join(pieces), len(stale_sections)


def _tag_exists(repository: Path, tag: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        detail = proc.stderr.strip() or proc.stdout.strip() or "git failed"
        raise RuntimeError(f"could not inspect release tag {tag}: {detail}")
    return proc.returncode == 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--config", type=Path, default=Path("release-plz.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = args.repository.resolve()
    manifest = repository / args.manifest
    config = repository / args.config
    changelog_path = repository / args.changelog
    context = load_release_context(manifest, config)
    original = changelog_path.read_text(encoding="utf-8")
    tag_exists = _tag_exists(repository, context.release_tag)
    updated, removed = remove_unpublished_release_sections(
        original,
        version=context.version,
        packages=context.changelog_packages,
        release_tag_exists=tag_exists,
    )
    if updated != original:
        changelog_path.write_text(updated, encoding="utf-8")

    if tag_exists:
        print(
            f"Release tag {context.release_tag} exists; preserved published "
            f"{context.version} changelog sections."
        )
    else:
        print(
            f"Removed {removed} generated changelog section(s) for unpublished "
            f"workspace version {context.version}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
