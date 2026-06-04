# Poster: Opening the Black Box

A conference poster summarising the kiji-inspector paper *"Opening the Black
Box: Mechanistic Interpretability of Agent Tool Selection with Sparse
Autoencoders"* (Hapke & Cardozo, Dataiku 575 Lab).

The PDF (`poster.pdf`) is committed; the LaTeX source is self-contained and
rebuilds with `make` if you have a TeX Live distribution with LuaLaTeX and
`latexmk`.

## Build

```
make            # produces poster.pdf via latexmk + lualatex
make clean      # remove build artefacts
```

Required TeX packages: `beamerposter`, `tikz`, `pgfplots`, `booktabs`,
`graphicx`, `qrcode`, `xcolor`, and the Raleway / Lato fonts (OFL).

## Files

- `poster.tex` — main document; 3-column 118.9 × 84.1 cm beamerposter
- `poster.bib` — bibliography (subset of `../paper/references.bib`)
- `images/` — figures used by the poster (mostly copies of `../paper/images/`
  plus the 575 Lab logo)
- `beamerthemegemini.sty`, `colorthemes/` — the
  [gemini](https://github.com/anishathalye/gemini) beamerposter theme
  (MIT, Anish Athalye); see `LICENSE.md` for terms
- `Makefile`, `.latexmkrc` — build configuration

## Hero result

> **10.1 %** of agent tool choices reverse when 10 SAE features are ablated.
> Random ablation: 0 %. SAE round-trip: 0 %. *p* = 0.002.
> *(fundamental vs. technical analysis contrast, Nemotron-3-Nano-30B, layer 20;
> up to 17.1 % on other contrast types.)*

## Attribution

Theme: [anishathalye/gemini](https://github.com/anishathalye/gemini) (MIT).
Color theme used: `dart` (Dartmouth green).
