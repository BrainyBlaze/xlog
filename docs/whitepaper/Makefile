# Makefile for xlog v0.5.0 whitepaper (arxiv-style LaTeX).
# Requires: latexmk, pdflatex, biber (MiKTeX or TeX Live).
# Optional: mmdc (@mermaid-js/mermaid-cli) for rebuilding figures from .mmd sources.

MAIN    := main
TEX     := $(MAIN).tex
PDF     := $(MAIN).pdf
MMDS    := $(wildcard figures/*.mmd)
FIGPDFS := $(MMDS:.mmd=.pdf)

.PHONY: all figures clean distclean watch

all: $(PDF)

$(PDF): $(TEX) refs.bib $(wildcard sections/*.tex) $(FIGPDFS) arxiv.sty
	latexmk -pdf $(TEX)

figures: $(FIGPDFS)

figures/%.pdf: figures/%.mmd
	@command -v mmdc >/dev/null 2>&1 && \
		mmdc -i $< -o $@ -b transparent --pdfFit || \
		echo "mmdc not found; keeping committed $@"

watch:
	latexmk -pdf -pvc $(TEX)

clean:
	latexmk -c $(TEX)

distclean:
	latexmk -C $(TEX)
