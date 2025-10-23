#!/bin/bash
cd "$(dirname "$0")"
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
echo "Done! Check report.pdf"
