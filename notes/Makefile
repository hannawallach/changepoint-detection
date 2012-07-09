# Makefile for LaTeX files
# (c) 2002 Hanna Wallach

FILENAME = changepoint

.PHONY: all pdf clean

all: pdf

pdf: $(FILENAME).pdf

$(FILENAME).pdf: *.tex *.bib
	echo "Running latex...."
	pdflatex -halt-on-error $(FILENAME).tex
	echo "Running bibtex...."
	bibtex $(FILENAME)
	echo "Rerunning latex...."
	pdflatex -halt-on-error $(FILENAME).tex
	pdflatex -halt-on-error $(FILENAME).tex

clean:
	rm -f *.aux *.dvi *.log *.toc *.blg *.bbl $(FILENAME).pdf *~
