#!/usr/bin/env bash
set -euo pipefail

# Generate a single PDF from all documentation markdown files.
#
# Prerequisites:
#   brew install pandoc
#   brew install --cask mactex   # or: brew install tectonic

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SCRIPT_DIR}/yakk_inspector_documentation.pdf"

# Ordered list of input files (index first, then steps in order)
INPUT_FILES=(
    "${SCRIPT_DIR}/index.md"
    "${SCRIPT_DIR}/step1_contrastive_pair_generation.md"
    "${SCRIPT_DIR}/step2_activation_extraction.md"
    "${SCRIPT_DIR}/step3_sae_training.md"
    "${SCRIPT_DIR}/step4_contrastive_activation_analysis.md"
    "${SCRIPT_DIR}/step5_feature_interpretation.md"
    "${SCRIPT_DIR}/step6_fuzzing_evaluation.md"
)

# Check prerequisites
if ! command -v pandoc &>/dev/null; then
    echo "Error: pandoc is not installed. Install with: brew install pandoc" >&2
    exit 1
fi

# Verify all input files exist
for f in "${INPUT_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: missing input file: $f" >&2
        exit 1
    fi
done

pandoc "${INPUT_FILES[@]}" \
    -o "$OUTPUT" \
    --toc \
    --toc-depth=3 \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=report \
    --highlight-style=tango \
    --pdf-engine=pdflatex

echo "Generated: $OUTPUT"
