#!/usr/bin/env python3
"""Materialize docs.json redirects and a custom 404 page in the export bundle.

Static exports do not implement Mintlify's `redirects` config, so legacy
mkdocs URLs would 404. This script reads the redirects from docs.json and
writes a stub index.html (canonical link + meta refresh + JS replace) at
each old path, plus a branded 404.html for the App Platform error_document.

Run AFTER Pagefind indexing (stubs carry no data-pagefind-body, so they are
never indexed either way) and before the rustdoc graft.

Usage: build_redirect_stubs.py <docs-dir> <bundle-dir>
"""

import html
import json
import pathlib
import sys

STUB_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting&#8230;</title>
<link rel="canonical" href="{dest}">
<meta http-equiv="refresh" content="0; url={dest}">
<script>location.replace({dest_js});</script>
</head>
<body>
<p>This page has moved to <a href="{dest}">{dest_text}</a>.</p>
</body>
</html>
"""

PAGE_404 = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page not found &#8211; XLOG</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: Inter, system-ui, sans-serif; background: #fbfbfd; color: #1e1b2e;
  }
  @media (prefers-color-scheme: dark) { body { background: #0e0e16; color: #e7e6f0; } }
  main { text-align: center; padding: 2rem; }
  .code { font-size: 4rem; font-weight: 700; color: #4f46e5; margin: 0; }
  @media (prefers-color-scheme: dark) { .code { color: #8b83f0; } }
  p { color: #6b6880; margin: 0.75rem 0 1.5rem; }
  @media (prefers-color-scheme: dark) { p { color: #9b98b4; } }
  a.button {
    display: inline-block; padding: 0.6rem 1.2rem; border-radius: 8px; text-decoration: none;
    background: #4f46e5; color: #fff; font-weight: 600; font-size: 0.95rem;
  }
</style>
</head>
<body>
<main>
  <p class="code">404</p>
  <p>That page does not exist. It may have moved &#8212; try the search (Ctrl&#8202;K) on any page.</p>
  <a class="button" href="/">Back to the documentation</a>
</main>
</body>
</html>
"""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 64
    site_dir = pathlib.Path(sys.argv[1])
    bundle = pathlib.Path(sys.argv[2])
    docs_json = site_dir / "docs.json"
    if not docs_json.exists() or not (bundle / "index.html").exists():
        print("error: bad docs or bundle dir", file=sys.stderr)
        return 65

    redirects = json.loads(docs_json.read_text(encoding="utf-8")).get("redirects", [])
    written = skipped = 0
    for entry in redirects:
        source = entry["source"].strip("/")
        dest = entry["destination"]
        if not source:
            skipped += 1
            continue
        target_dir = bundle / source
        target = target_dir / "index.html"
        # Never clobber a real exported page.
        if target.exists():
            skipped += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            STUB_TEMPLATE.format(
                dest=html.escape(dest, quote=True),
                dest_js=json.dumps(dest),
                dest_text=html.escape(dest),
            ),
            encoding="utf-8",
        )
        written += 1

    (bundle / "404.html").write_text(PAGE_404, encoding="utf-8")
    print(f"redirect stubs written={written} skipped={skipped}; 404.html written")
    if written == 0 and redirects:
        print("error: no stubs written despite configured redirects", file=sys.stderr)
        return 66
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
