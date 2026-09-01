#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../.. && pwd)"
TEX="From Readable to Causal.tex"
BASE="From Readable to Causal"
TEX_IMAGE="texlive/texlive:latest"

if command -v pdflatex >/dev/null 2>&1; then
    tex() { TEXINPUTS=".:neurips_2026_formatting:..:" BSTINPUTS=".:..:" BIBINPUTS=".:..:" "$@"; }
elif command -v docker >/dev/null 2>&1; then
    docker image inspect "$TEX_IMAGE" >/dev/null 2>&1 || docker pull "$TEX_IMAGE"
    tex() {
        docker run --rm \
            -v "$REPO_ROOT:/work" -w /work/paper/steering_workshop \
            -u "$(id -u):$(id -g)" \
            -e TEXINPUTS=".:neurips_2026_formatting:..:" \
            -e BSTINPUTS=".:..:" -e BIBINPUTS=".:..:" \
            "$TEX_IMAGE" "$@"
    }
else
    echo "Neither pdflatex nor docker is available."
    exit 1
fi

tex pdflatex -interaction=nonstopmode "$TEX" >/dev/null
tex bibtex "$BASE" >/dev/null
tex pdflatex -interaction=nonstopmode "$TEX" >/dev/null
tex pdflatex -interaction=nonstopmode "$TEX" >/dev/null

if grep -E '^!' "$BASE.log"; then
    echo "LaTeX errors found"
    exit 1
fi
grep -E '^Overfull \\hbox' "$BASE.log" || true
grep -iE 'undefined (control sequence|reference|citation)' "$BASE.log" || true
rm -f "$BASE.aux" "$BASE.bbl" "$BASE.blg" "$BASE.log" "$BASE.out"
echo "Built $BASE.pdf"
