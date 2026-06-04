"""Regenerate the poster's figures in the Dataiku brand palette (poster-only).

Charts: imports paper/generate_charts.py, overrides its COLORS dict with
DESIGN.md-documented Dataiku tokens, and re-renders chart_layer_sweep and
chart_ablation into poster/images/. paper/images/ is left untouched.

Diagrams: recolors the poster-local SVG copies (poster/images/*.svg) via a hex
map and re-renders them to PNG with rsvg-convert.

Run:
    uv run --no-project --with matplotlib python3 poster/regenerate_figures_dataiku.py
"""

import subprocess
import sys
from pathlib import Path

# Dataiku brand palette --- DESIGN.md-documented tokens, plus dark/light shades
# derived from them (teal darkened so white on-bar labels stay WCAG-readable).
TEAL_BLUE = "#045067"  # @teal-blue-base
TEAL = "#2AB1AC"  # @teal-base
TEAL_DK = "#1f7a76"  # darkened @teal-base (white-on-bar readable)
TEAL_DK2 = "#176b67"  # darker teal (gradient ends)
MAGENTA = "#c51d54"  # brand accent
MAGENTA_L = "#d24f7a"  # lightened brand accent
INK = "#222222"  # @grey-base

HERE = Path(__file__).resolve().parent  # poster/
ROOT = HERE.parent
IMAGES = HERE / "images"

# --- Charts (matplotlib) ----------------------------------------------------
DATAIKU_COLORS = {
    "primary": TEAL_BLUE,
    "primary_light": TEAL_DK,
    "accent": MAGENTA,
    "accent_light": MAGENTA_L,
    "text": INK,
    "text_light": "#718096",  # neutral gray (kept)
    "border": "#e2e8f0",  # neutral light gray (kept)
}


def regenerate_charts() -> None:
    sys.path.insert(0, str(ROOT / "paper"))
    import generate_charts as gc
    import matplotlib.axes as maxes

    gc.COLORS = DATAIKU_COLORS
    gc.bw_mode = False
    gc.dpi = 300

    # Restyle the hardcoded white "dead: X%" bar labels (generate_charts.py:238)
    # as dark ink inside a light rounded chip -- readable on both the teal bars
    # and the magenta selected bar. Intercepted here so generate_charts.py is
    # left unmodified.
    orig_text = maxes.Axes.text

    def text_with_dead_chip(self, *args, **kwargs):
        s = args[2] if len(args) >= 3 else kwargs.get("s")
        if isinstance(s, str) and s.startswith("dead:"):
            kwargs["color"] = INK
            kwargs["bbox"] = dict(
                boxstyle="round,pad=0.3",
                facecolor="#f7fafc",
                edgecolor="none",
                alpha=0.92,
            )
        return orig_text(self, *args, **kwargs)

    maxes.Axes.text = text_with_dead_chip
    try:
        gc.chart_layer_sweep(IMAGES)
        gc.chart_ablation(IMAGES)
    finally:
        maxes.Axes.text = orig_text
    print("charts -> chart_layer_sweep.png, chart_ablation.png")


# --- Diagrams (SVG) ---------------------------------------------------------
# Lightness is preserved per mapping so existing white-on-box contrast holds.
HEX_MAP = {
    "#1a2744": TEAL_BLUE,
    "#2d4a7a": TEAL_DK,
    "#e8563a": MAGENTA,
    "#ff7b5f": MAGENTA_L,
    "#38b2ac": TEAL,
    "#2c7a7b": TEAL_DK2,
    "#2d3748": INK,
}
SVGS = ["sae_architecture", "inference_pipeline"]


def recolor_svgs() -> None:
    for name in SVGS:
        svg = IMAGES / f"{name}.svg"
        text = svg.read_text(encoding="utf-8")
        for old, new in HEX_MAP.items():
            text = text.replace(old, new).replace(old.upper(), new)
        # Darken the grad-teal top stop so white text on the teal boxes
        # (FEATURE THEMES / "features (sparse)") meets WCAG AA contrast.
        # Solid teal accents (arrows, borders) keep the brighter brand teal.
        text = text.replace(f'stop-color="{TEAL}"', f'stop-color="{TEAL_DK}"')
        svg.write_text(text, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-w", "3000", str(svg), "-o", str(IMAGES / f"{name}.png")],
            check=True,
        )
        print(f"diagram -> {name}.svg recolored + {name}.png rendered")


if __name__ == "__main__":
    regenerate_charts()
    recolor_svgs()
