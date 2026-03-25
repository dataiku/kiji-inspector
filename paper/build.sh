#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TEX="Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.tex"
BASE="${TEX%.tex}"

compile_pdf() {
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null
    bibtex "$BASE" > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null
    pdflatex -interaction=nonstopmode "$TEX" > /dev/null
    rm -f "$BASE".{aux,bbl,blg,log,out,toc,lof,lot,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
}

# ── 1. Render color PNGs from SVGs, generate color charts, compile color PDF ─
echo "==> Rendering color images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode color
echo "==> Generating color charts..."
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode color

echo "==> Compiling color PDF..."
compile_pdf
mv "$BASE.pdf" "$BASE (color).pdf"
echo "    ✓ $BASE (color).pdf"

# ── 2. Render B&W PNGs, generate B&W charts, compile B&W PDF ─────────────────
echo "==> Rendering B&W images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode bw
echo "==> Generating B&W charts..."
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode bw

echo "==> Compiling B&W PDF..."
compile_pdf
mv "$BASE.pdf" "$BASE (bw).pdf"
echo "    ✓ $BASE (bw).pdf"

# ── 3. Restore color PNGs as the default working state ────────────────
echo "==> Restoring color images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode color > /dev/null
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode color > /dev/null

echo "Done."
