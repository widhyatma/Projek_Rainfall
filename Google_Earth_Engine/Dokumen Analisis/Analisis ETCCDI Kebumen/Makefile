# Makefile untuk kompilasi dokumen LaTeX ETCCDI Kebumen
# Jalankan: pdflatex main.tex (minimal 2x untuk referensi silang)
# Atau gunakan latexmk: latexmk -pdf main.tex

MAIN = Analisis EETCDI Kebumen
PDFLATEX = pdflatex
BIBTEX = bibtex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	del /f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

cleanall: clean
	del /f $(MAIN).pdf

.PHONY: all clean cleanall
