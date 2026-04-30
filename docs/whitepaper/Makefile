# Makefile for xlog v0.6.1 whitepaper (arxiv-style LaTeX).
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

# On Windows / git-bash the `mmdc` shim mangles paths; invoke via cmd.exe so
# the Windows .cmd wrapper resolves correctly. On POSIX, fall back to plain mmdc.
MMDC := $(shell command -v mmdc 2>/dev/null)
ifeq ($(OS),Windows_NT)
  MMDC_INVOKE := cmd //c "mmdc -i $$< -o $$@ -b transparent --pdfFit"
else
  MMDC_INVOKE := mmdc -i $$< -o $$@ -b transparent --pdfFit
endif

figures/%.pdf: figures/%.mmd
	@if [ -n "$(MMDC)" ]; then \
		echo "Rendering $< -> $@"; \
		$(MMDC_INVOKE); \
	else \
		echo "mmdc not found; keeping committed $@"; \
	fi

watch:
	latexmk -pdf -pvc $(TEX)

clean:
	latexmk -c $(TEX)

distclean:
	latexmk -C $(TEX)
