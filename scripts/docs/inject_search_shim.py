#!/usr/bin/env python3
"""Prepare the exported docs bundle for self-hosted Pagefind search.

Run BEFORE `pagefind --site <bundle>` and BEFORE the rustdoc graft:

1. Tags each page's main content container with `data-pagefind-body` so
   Pagefind indexes article content only (not the sidebar/nav chrome).
   Because the attribute is present, Pagefind ignores any page without it
   (e.g. later-grafted rustdoc pages) by design.
2. Injects the search-shim stylesheet/script tags before </head>.
3. Copies search-shim.js / search-shim.css into the bundle root.

Usage: inject_search_shim.py <bundle-dir>
"""

import pathlib
import shutil
import sys

HEAD_SNIPPET = (
    '<link rel="stylesheet" href="/search-shim.css">'
    '<script src="/search-shim.js" defer></script></head>'
)
BODY_MARK = '<main id="content-container"'
BODY_MARK_TAGGED = '<main data-pagefind-body id="content-container"'


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 64
    bundle = pathlib.Path(sys.argv[1])
    if not (bundle / "index.html").exists():
        print(f"error: {bundle} does not look like an export bundle", file=sys.stderr)
        return 65

    here = pathlib.Path(__file__).resolve().parent
    pages = tagged = injected = 0
    for page in bundle.rglob("*.html"):
        pages += 1
        text = page.read_text(encoding="utf-8")
        changed = False
        if BODY_MARK in text and BODY_MARK_TAGGED not in text:
            text = text.replace(BODY_MARK, BODY_MARK_TAGGED, 1)
            tagged += 1
            changed = True
        if "</head>" in text and "/search-shim.js" not in text:
            text = text.replace("</head>", HEAD_SNIPPET, 1)
            injected += 1
            changed = True
        if changed:
            page.write_text(text, encoding="utf-8")

    for asset in ("search-shim.js", "search-shim.css"):
        shutil.copy2(here / asset, bundle / asset)

    print(f"pages={pages} tagged={tagged} head-injected={injected}")
    if tagged == 0 or injected == 0:
        print("error: no pages tagged/injected - export layout changed?", file=sys.stderr)
        return 66
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
