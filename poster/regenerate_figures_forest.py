#!/usr/bin/env python3
"""Regenerate the poster's figures in the Forest palette (poster-only).

Forest = the green secondary brand palette from colors.md:
  Dark Green #06312E, Green #3EDAB2, Light Green #C7FFF1, Beige #F8F4E4,
  Core Black #1A1A1A, Core White #FFFEF9 (+ documented neutral greys).

Charts: imports paper/generate_charts.py, overrides its COLORS dict, and
re-renders chart_layer_sweep and chart_ablation into poster/images/.
paper/images/ is left untouched.

Diagrams: recolors the *current* poster-local SVGs (poster/images/*.svg, which
already carry the enlarged fonts + layout fixes) from their Dataiku colors to
the Forest palette, then re-renders to PNG with rsvg-convert.

Run:
    uv run --no-project --with matplotlib python3 poster/regenerate_figures_forest.py
"""
import subprocess
import sys
from pathlib import Path

# Forest palette --- colors.md tokens. Greens carry no semantic warning/error
# hue, so the hero/selected series uses bright Green against Dark-Green context.
DARK = "#06312E"    # Dark Green  (primary / context / text-on-light)
MINT = "#3EDAB2"    # Green       (hero / selected / accents)
LIGHT = "#C7FFF1"   # Light Green (light accent)
BEIGE = "#F8F4E4"   # Beige       (warm fill)
INK = "#1A1A1A"     # Core Black
WHITE = "#FFFEF9"   # Core White
GREY = "#929088"    # Grey        (muted baseline series / fine axis text)
GREY_LT = "#EEEDEA" # Light Grey  (borders / gridlines)

HERE = Path(__file__).resolve().parent  # poster/
ROOT = HERE.parent
IMAGES = HERE / "images"

# --- Charts (matplotlib) ----------------------------------------------------
FOREST_COLORS = {
    "primary": DARK,         # context bars, titles, axis labels
    "primary_light": DARK,   # non-selected layer bars (so the mint hero pops)
    "accent": MINT,          # selected layer / contrastive / MSE line (hero)
    "accent_light": LIGHT,
    "text": INK,
    "text_light": GREY,      # muted "random/baseline" series + 2nd-axis text
    "border": GREY_LT,
}


def regenerate_charts() -> None:
    sys.path.insert(0, str(ROOT / "paper"))
    import generate_charts as gc
    import matplotlib.axes as maxes
    import matplotlib.figure as mfig
    import matplotlib.pyplot as plt

    gc.COLORS = FOREST_COLORS
    gc.bw_mode = False
    gc.dpi = 300

    # Each chart's background matches the poster block it sits in, so the figure
    # blends into the beige page instead of showing a pale rectangle. The
    # ablation chart lives in a light-green alert card; the rest are on beige.
    chart_bg = {"v": BEIGE}

    # Restyle the hardcoded white "dead: X%" bar labels (generate_charts.py:238)
    # as dark ink in a light chip -- readable on both the dark-green context bars
    # and the mint selected bar. Intercepted here so generate_charts.py stays
    # unmodified.
    orig_text = maxes.Axes.text
    orig_annotate = maxes.Axes.annotate
    orig_savefig = mfig.Figure.savefig

    def savefig_with_bg(self, *args, **kwargs):
        # generate_charts.py hardcodes facecolor="white"; override per chart.
        kwargs["facecolor"] = chart_bg["v"]
        # Recolor the MSE line + its right-hand axis to Core Black so it no
        # longer matches the mint selected bar. Only the marker-bearing series
        # (the MSE line) is touched; the selected bar and "selected" label
        # stay mint. Charts without such a line (ablation) are unaffected.
        for ax in self.axes:
            mse_lines = [ln for ln in ax.get_lines()
                         if ln.get_marker() not in (None, "None", "")]
            if not mse_lines:
                continue
            for ln in mse_lines:
                ln.set_color(INK)
                ln.set_markerfacecolor(INK)
                ln.set_markeredgecolor(INK)
            ax.yaxis.label.set_color(INK)
            ax.tick_params(axis="y", colors=INK)
            ax.spines["right"].set_color(INK)
        return orig_savefig(self, *args, **kwargs)

    def text_with_dead_chip(self, *args, **kwargs):
        s = args[2] if len(args) >= 3 else kwargs.get("s")
        if isinstance(s, str) and s.startswith("dead:"):
            kwargs["color"] = INK
            kwargs["bbox"] = dict(
                boxstyle="round,pad=0.3", facecolor=WHITE,
                edgecolor="none", alpha=0.92,
            )
        return orig_text(self, *args, **kwargs)

    def annotate_with_chip(self, *args, **kwargs):
        # The MSE value labels (e.g. "0.574", "1,524") are drawn in the accent
        # color and land on top of bars -- on the mint selected bar the accent
        # text would vanish. Chip every numeric annotation (dark ink in a light
        # pill) so it reads on any bar; leave "selected"/stars in the accent.
        s = args[0] if args else kwargs.get("text")
        if isinstance(s, str) and s.replace(",", "").replace(".", "").isdigit():
            kwargs["color"] = INK
            kwargs["bbox"] = dict(
                boxstyle="round,pad=0.2", facecolor=WHITE,
                edgecolor="none", alpha=0.9,
            )
        return orig_annotate(self, *args, **kwargs)

    maxes.Axes.text = text_with_dead_chip
    maxes.Axes.annotate = annotate_with_chip
    mfig.Figure.savefig = savefig_with_bg
    try:
        plt.rcParams["axes.facecolor"] = BEIGE
        plt.rcParams["figure.facecolor"] = BEIGE
        chart_bg["v"] = BEIGE
        gc.chart_layer_sweep(IMAGES)

        plt.rcParams["axes.facecolor"] = LIGHT
        plt.rcParams["figure.facecolor"] = LIGHT
        chart_bg["v"] = LIGHT
        gc.chart_ablation(IMAGES)
    finally:
        maxes.Axes.text = orig_text
        maxes.Axes.annotate = orig_annotate
        mfig.Figure.savefig = orig_savefig
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
    print("charts -> chart_layer_sweep.png, chart_ablation.png")


# --- Diagrams (SVG) ---------------------------------------------------------
# Order matters: the Green-pop accents (strokes + arrowheads, which never sit
# behind white text) are applied FIRST, before the blanket map sends every
# remaining brand color to Dark Green / Core Black (legible behind white box
# fills and as text).
ACCENT_REPLACE = [
    ('stroke="#c51d54"', f'stroke="{MINT}"'),                 # flow arrow lines
    ('stroke="#2AB1AC"', f'stroke="{MINT}"'),                 # output border + bend
    ('points="0 0, 10 3.5, 0 7" fill="#c51d54"',             # flow arrowheads
     f'points="0 0, 10 3.5, 0 7" fill="{MINT}"'),
]
# Text that sits ON a dark box: the blanket map would send it to Dark Green and
# make it vanish, so retint to the light mint accent. Applied before the map.
TARGETED = [
    ('fill="#d24f7a" font-weight="600">Extract at decision token',
     f'fill="{MINT}" font-weight="600">Extract at decision token'),
]
HEX_MAP = {
    "#045067": DARK,   # deep brand (box gradients, text, navy arrows)
    "#1f7a76": DARK,   # gradient tops (kept dark for white-text contrast)
    "#176b67": DARK,   # teal-dark (recon stroke/text, gradient ends)
    "#2c7a7b": DARK,
    "#c51d54": DARK,   # remaining magenta: circles, spikes, layer-20, text
    "#d24f7a": DARK,
    "#2d3748": INK,
    "#222222": INK,
    "#718096": GREY,   # neutral gray text
    "#e2e8f0": GREY_LT,  # neutral borders
    "#f7fafc": WHITE,  # light card fills
}
SVGS = ["sae_architecture", "inference_pipeline"]

# The diagram page background (only this rect -- NOT the white text/cards, which
# also use #ffffff) is recolored to beige so each diagram blends into the page.
BG_REPLACE = (
    '<rect width="1100" height="560" fill="#ffffff" rx="8"/>',
    f'<rect width="1100" height="560" fill="{BEIGE}" rx="8"/>',
)


def recolor_svgs() -> None:
    for name in SVGS:
        svg = IMAGES / f"{name}.svg"
        text = svg.read_text(encoding="utf-8")
        text = text.replace(*BG_REPLACE)
        for old, new in TARGETED:
            text = text.replace(old, new)
        for old, new in ACCENT_REPLACE:
            text = text.replace(old, new)
        for old, new in HEX_MAP.items():
            text = text.replace(old, new).replace(old.upper(), new)
        svg.write_text(text, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-w", "3000", str(svg), "-o", str(IMAGES / f"{name}.png")],
            check=True,
        )
        print(f"diagram -> {name}.svg recolored + {name}.png rendered")


if __name__ == "__main__":
    regenerate_charts()
    recolor_svgs()
