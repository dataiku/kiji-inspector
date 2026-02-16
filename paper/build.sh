#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TEX="Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.tex"
BASE="${TEX%.tex}"

pdflatex "$TEX"
bibtex "$BASE"
pdflatex "$TEX"
pdflatex "$TEX"

# Clean up temporary files
rm -f "$BASE".{aux,bbl,blg,log,out,toc,lof,lot,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}

echo "Done — output: $BASE.pdf"
