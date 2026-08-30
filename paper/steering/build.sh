#!/usr/bin/env bash
# Build the steering paper. The NeurIPS style/checklist are kept in
# neurips_2026_formatting; the shared bibliography is reused from ../.
#
# Uses a local pdflatex if there is one, otherwise the texlive Docker image.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

TEX="Steering Agent Tool Selection.tex"
BASE="${TEX%.tex}"
MODE="${1:-color}"
TEX_IMAGE="${TEX_IMAGE:-texlive/texlive:latest}"

echo "==> Extracting results from demo/steering/..."
(cd "$REPO_ROOT" && python3 paper/steering/extract_results.py)

echo "==> Generating charts ($MODE)..."
uv run --no-project --with matplotlib --with numpy python3 generate_charts.py --mode "$MODE"

# ── pick a LaTeX toolchain ──────────────────────────────────────────────────
if command -v pdflatex >/dev/null 2>&1; then
    tex() { TEXINPUTS=".:neurips_2026_formatting:..:" BSTINPUTS=".:..:" BIBINPUTS=".:..:" "$@"; }
elif command -v docker >/dev/null 2>&1; then
    echo "==> No local pdflatex; using $TEX_IMAGE"
    docker image inspect "$TEX_IMAGE" >/dev/null 2>&1 || docker pull "$TEX_IMAGE"
    tex() {
        docker run --rm \
            -v "$REPO_ROOT:/work" -w /work/paper/steering \
            -u "$(id -u):$(id -g)" \
            -e TEXINPUTS=".:neurips_2026_formatting:..:" -e BSTINPUTS=".:..:" -e BIBINPUTS=".:..:" \
            "$TEX_IMAGE" "$@"
    }
else
    echo "!! Neither pdflatex nor docker found — results and figures are up to"
    echo "   date, but the PDF was not built.  See ../README.md."
    exit 0
fi

echo "==> Compiling PDF..."
tex pdflatex -interaction=nonstopmode "$TEX" > /dev/null
tex bibtex "$BASE" > /dev/null 2>&1 || true
tex pdflatex -interaction=nonstopmode "$TEX" > /dev/null
tex pdflatex -interaction=nonstopmode "$TEX" > /dev/null

# Surface anything that would show as a defect on the printed page before the
# log is deleted.
if grep -E '^!' "$BASE.log"; then echo "^^ LaTeX errors"; exit 1; fi
grep -E '^Overfull \\hbox' "$BASE.log" || true
grep -iE 'undefined (control sequence|reference|citation)' "$BASE.log" || true

rm -f "$BASE".{aux,bbl,blg,log,out,toc,fls,fdb_latexmk,synctex.gz}
echo "    ✓ $BASE.pdf"
