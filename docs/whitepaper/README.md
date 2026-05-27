# xlog v0.9.0-rc Whitepaper (LaTeX)

arxiv-style single-column LaTeX source for the xlog v0.9.0-rc technical whitepaper: *"xlog: A GPU-Native Logic Programming Language for Unified Symbolic Reasoning"*.

The rendered PDF is committed as `main.pdf`; rebuild only when sources change.

## Build

```bash
cd docs/whitepaper
latexmk -pdf main.tex        # -> main.pdf
```

Requires a working LaTeX distribution (MiKTeX, TeX Live) with `pdflatex`, `latexmk`, and `biber`.

## Figures

Diagram sources live under `figures/*.mmd` (Mermaid). Rendered PDFs are committed so the document builds without `mmdc`. To regenerate after editing a source:

```bash
npm install -g @mermaid-js/mermaid-cli    # one-time
make figures
```

## Layout

| Path | Purpose |
|---|---|
| `main.tex` | Preamble + `\input{sections/*}` |
| `arxiv.sty` | Vendored arXiv-style preamble |
| `refs.bib` | Bibliography (biblatex) |
| `sections/*.tex` | One file per whitepaper section (10 sections: abstract, intro, architecture, language, Datalog evaluation, probabilistic, neural-symbolic, interop, evaluation, related work, limitations) |
| `figures/*.mmd` | Mermaid diagram sources |
| `figures/*.pdf` | Rendered figures (checked in) |
| `latexmkrc` / `Makefile` | Build automation |
