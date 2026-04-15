# xlog v0.5.0 Whitepaper (LaTeX)

arxiv-style single-column LaTeX port of `docs/whitepaper-v050.md`.

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
| `sections/*.tex` | One file per whitepaper section |
| `figures/*.mmd` | Mermaid diagram sources |
| `figures/*.pdf` | Rendered figures |
| `latexmkrc` / `Makefile` | Build automation |
