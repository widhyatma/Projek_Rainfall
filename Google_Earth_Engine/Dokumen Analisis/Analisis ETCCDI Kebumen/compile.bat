@echo off
echo === Kompilasi LaTeX: Analisis Iklim ETCCDI Kebumen ===
echo.
pdflatex -interaction=nonstopmode "Analisis Iklim ETCCDI Kebumen.tex"
bibtex "Analisis Iklim ETCCDI Kebumen"
pdflatex -interaction=nonstopmode "Analisis Iklim ETCCDI Kebumen.tex"
pdflatex -interaction=nonstopmode "Analisis Iklim ETCCDI Kebumen.tex"
echo.
echo === Kompilasi Selesai! Silakan buka Analisis Iklim ETCCDI Kebumen.pdf ===
pause
