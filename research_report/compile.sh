#!/bin/bash
cd "$(dirname "$0")"
pdflatex research_report.tex
bibtex research_report
pdflatex research_report.tex
pdflatex research_report.tex

# Move PDF to root level for easy GitHub viewing
mv research_report.pdf ../research_report.pdf

echo "Done! PDF saved to research_report.pdf (root level)"
