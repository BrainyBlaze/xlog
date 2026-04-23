# xlog v0.5.0 Whitepaper

arxiv-style single-column LaTeX source for the xlog v0.5.0 technical whitepaper: *"xlog: A GPU-Native Logic Programming Language for Unified Symbolic Reasoning"*.

The rendered PDF is [`main.pdf`](main.pdf). The validated xlog code snippets referenced by Section 3 live under [`examples/`](examples/).

## Source

The LaTeX source (preamble, per-section `.tex` files, `refs.bib`, `arxiv.sty`, Mermaid figure sources, `latexmkrc`, `Makefile`) is **not tracked on the default branch**. It lives on the `whitepaper-source` branch, which preserves the full pipeline for anyone who needs to rebuild or edit:

```bash
# Extract the source into your working tree without changing branches
git fetch origin whitepaper-source
git checkout origin/whitepaper-source -- docs/whitepaper/

# Or check out the source branch directly
git checkout whitepaper-source
```

This keeps the default branch free of LaTeX plumbing while preserving full reproducibility.

## Build

From a worktree that has the source checked in:

```bash
cd docs/whitepaper
latexmk -pdf main.tex        # -> main.pdf
```

Requires a working LaTeX distribution (MiKTeX, TeX Live) with `pdflatex`, `latexmk`, and `biber`. Mermaid figure regeneration (optional, only after editing `figures/*.mmd`) requires `@mermaid-js/mermaid-cli`:

```bash
npm install -g @mermaid-js/mermaid-cli
make figures
```
