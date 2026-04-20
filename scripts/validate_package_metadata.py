"""Validate README quickstart and package metadata invariants used by CI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_version_support import README_BADGE_VERSION_RE, workspace_version

REQUIRED_SNIPPETS = (
    "python scripts/xlog_doctor.py",
    "cargo build --release",
    "cargo build --release -p xlog-cli --features host-io",
    "./target/release/xlog",
)


def validate_package_metadata(
    *,
    readme: str,
    workspace_version: str,
    metadata: dict,
) -> list[str]:
    errors: list[str] = []
    readme_plain = readme.replace("**", "")

    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in readme]
    if missing:
        errors.append("README quickstart is missing required snippets:")
        errors.extend(f"  - {snippet}" for snippet in missing)

    badge_match = README_BADGE_VERSION_RE.search(readme)
    if not badge_match:
        errors.append("Could not find README version badge.")
    elif badge_match.group(1) != workspace_version:
        errors.append(
            f"README version badge ({badge_match.group(1)}) does not match workspace version ({workspace_version})."
        )

    status_match = re.search(r"Release status:\s*`v([^`]+)`", readme_plain)
    if not status_match:
        errors.append("Could not find README release status line.")
    elif status_match.group(1) != workspace_version:
        errors.append(
            f"README release status ({status_match.group(1)}) does not match workspace version ({workspace_version})."
        )

    xlog_cli = next(
        (
            package
            for package in metadata["packages"]
            if package["name"] == "xlog-cli" and package.get("source") is None
        ),
        None,
    )
    if xlog_cli is None:
        errors.append("cargo metadata did not include the local xlog-cli package.")
        return errors

    bin_targets = {
        target["name"] for target in xlog_cli["targets"] if "bin" in target["kind"]
    }
    if "xlog" not in bin_targets:
        errors.append(
            f"xlog-cli binary targets do not include `xlog`: {sorted(bin_targets)}"
        )

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--cargo", default="Cargo.toml")
    parser.add_argument("--metadata", default="cargo-metadata.json")
    return parser.parse_args(argv)
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    readme = Path(args.readme).read_text(encoding="utf-8")
    cargo_text = Path(args.cargo).read_text(encoding="utf-8")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    current_workspace_version = workspace_version(cargo_text)

    errors = validate_package_metadata(
        readme=readme,
        workspace_version=current_workspace_version,
        metadata=metadata,
    )
    if errors:
        print("\n".join(errors))
        return 1

    print("README quickstart assumptions and workspace package metadata validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
