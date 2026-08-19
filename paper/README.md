# XLOG Whitepaper (LaTeX)

ArXiv-style two-column LaTeX source for the XLOG technical whitepaper: *"XLOG: A Universal GPU-Native Engine for Neurosymbolic Integration"*.

The rendered PDF is committed as `main.pdf`; rebuild only when sources change. Generated figure PDFs (`figures/results.pdf`) are not tracked — regenerate before building.

## Build

```bash
cd paper
(cd figures && python3 make_results.py)  # render figures/results.pdf from encoded values
latexmk -pdf main.tex               # -> main.pdf
```

Requires a working LaTeX distribution (MiKTeX, TeX Live) with `pdflatex`, `latexmk`, and `biber`. `make_results.py` needs Python with `matplotlib`.

## Figures

`figures/results.pdf` is rendered by `figures/make_results.py` from benchmark values encoded in that script; the generator does not read measurement artifacts automatically. Any Mermaid diagram sources (`figures/*.mmd`) can be re-rendered with:

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
| `sections/*.tex` | One file per whitepaper section (abstract, intro, architecture, language, Datalog evaluation, probabilistic, neural-symbolic, Event-Calculus induction on CAVIAR, maritime rule induction at scale, epistemic, evaluation, related work, limitations) |
| `figures/make_results.py` | Renders `figures/results.pdf` from benchmark values encoded in the script |
| `artifacts/head-to-head/` | Benchmark measurement JSONs + runner scripts backing the evaluation section |
| `latexmkrc` / `Makefile` | Build automation |
