#!/usr/bin/env python3
"""Generate static Markdown exports for the exported docs bundle.

Mintlify exposes per-page Markdown through `.md` URLs on hosted deployments, but
`mint export` emits only HTML. This script mirrors those `.md` URLs from the
source pages listed in docs.json so the static site can support copy-as-Markdown.

Usage: build_markdown_exports.py <docs-site-dir> <bundle-dir>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def collect_pages(value: object, pages: list[str]) -> None:
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "mailto:")):
            return
        pages.append(value.split("#", 1)[0].strip("/"))
        return
    if isinstance(value, list):
        for item in value:
            collect_pages(item, pages)
        return
    if isinstance(value, dict):
        if "pages" in value:
            collect_pages(value["pages"], pages)
        for key in ("tabs", "groups", "anchors", "dropdowns", "products", "versions", "languages"):
            if key in value:
                collect_pages(value[key], pages)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    raw = text[4:end].strip()
    body = text[end + len("\n---") :].lstrip()
    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def render_markdown(source: Path) -> str:
    metadata, body = parse_frontmatter(source.read_text(encoding="utf-8"))
    parts: list[str] = []
    title = metadata.get("title")
    description = metadata.get("description")
    if title:
        parts.append(f"# {title}\n")
    if description:
        parts.append(f"{description}\n")
    parts.append(body.rstrip())
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 64

    docs_site = Path(sys.argv[1])
    bundle = Path(sys.argv[2])
    config_path = docs_site / "docs.json"
    if not config_path.exists() or not bundle.exists():
        print("error: expected docs-site dir and exported bundle dir", file=sys.stderr)
        return 65

    config = json.loads(config_path.read_text(encoding="utf-8"))
    pages: list[str] = []
    collect_pages(config.get("navigation", {}), pages)

    written = 0
    seen: set[str] = set()
    for route in pages:
        if route in seen:
            continue
        seen.add(route)
        source = docs_site / f"{route}.mdx"
        if not source.exists():
            source = docs_site / f"{route}.md"
        if not source.exists():
            print(f"error: no source page for route {route}", file=sys.stderr)
            return 66

        target = bundle / f"{route}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_markdown(source), encoding="utf-8")
        written += 1

    print(f"markdown-pages={written}")
    if written == 0:
        print("error: no markdown pages written", file=sys.stderr)
        return 67
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
