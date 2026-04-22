#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TEX="Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.tex"
BASE="${TEX%.tex}"
TEX_SHORT="${BASE}_short.tex"
BASE_SHORT="${BASE}_short"

# Figures the short paper embeds as _bw.png variants (see _short.tex).
SHORT_BW_IMAGES=(training_pipeline sae_architecture)

compile_pdf() {
    local tex="$1"
    local base="${tex%.tex}"
    pdflatex -interaction=nonstopmode "$tex" > /dev/null
    bibtex "$base" > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$tex" > /dev/null
    pdflatex -interaction=nonstopmode "$tex" > /dev/null
    rm -f "$base".{aux,bbl,blg,log,out,toc,lof,lot,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
}

# ── 1. Render color PNGs from SVGs, generate color charts, compile color PDF ─
echo "==> Rendering color images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode color
echo "==> Generating color charts..."
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode color

echo "==> Compiling color PDF..."
compile_pdf "$TEX"
mv "$BASE.pdf" "$BASE (color).pdf"
echo "    ✓ $BASE (color).pdf"

# ── 2. Render B&W PNGs, compile B&W PDF, stash _bw.png for the short paper ──
echo "==> Rendering B&W images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode bw
echo "==> Generating B&W charts..."
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode bw

echo "==> Stashing _bw.png copies for short paper..."
for name in "${SHORT_BW_IMAGES[@]}"; do
    cp "images/${name}.png" "images/${name}_bw.png"
done

echo "==> Compiling B&W PDF..."
compile_pdf "$TEX"
mv "$BASE.pdf" "$BASE (bw).pdf"
echo "    ✓ $BASE (bw).pdf"

# ── 3. Compile short (B&W) PDF — references _bw.png variants directly ───────
echo "==> Compiling short (B&W) PDF..."
compile_pdf "$TEX_SHORT"
echo "    ✓ $BASE_SHORT.pdf"

# ── 4. Restore color PNGs as the default working state ──────────────────────
echo "==> Restoring color images..."
uv run --no-project --with playwright --with Pillow python3 render_svgs.py --mode color > /dev/null
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode color > /dev/null

echo "Done."
