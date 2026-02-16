# Paper

LaTeX source for the LNCS paper, using the Springer `llncs.cls` document class.

## Prerequisites

A LaTeX distribution with `pdflatex` and `bibtex`. For example:

- **macOS:** [MacTeX](https://tug.org/mactex/) or `brew install --cask mactex`
- **Linux:** `sudo apt install texlive-full` (or a smaller subset like `texlive-latex-recommended texlive-latex-extra texlive-bibtex-extra`)

All required style files (`llncs.cls`, `splncs04.bst`) are included in this directory.

## Compiling to PDF

A convenience script wraps the full build and cleans up temporary files:

```bash
./build.sh
```

The triple `pdflatex` pass resolves all cross-references and citations.

## Generating Charts

The figures in `images/` can be regenerated from the raw data in `results/`:

```bash
python generate_charts.py
```
