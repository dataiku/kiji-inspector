#!/usr/bin/env python3
"""Render SVGs to PNGs (color and grayscale) for paper figures.

Usage:
    python3 render_svgs.py --mode color   # render color PNGs into images/
    python3 render_svgs.py --mode bw       # render grayscale PNGs into images/
"""
import argparse
import os
import re
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

# SVG files that get rendered to PNG for the paper
SVG_NAMES = [
    "sae_architecture",
    "inference_pipeline",
    "training_pipeline",
    "nemotron_architecture",
]

# ---------------------------------------------------------------------------
# Grayscale color mapping
# ---------------------------------------------------------------------------
# Applied in order.  For SVG gradients where two original colours map to the
# same gray, we handle gradients first with context-aware replacement.

# Gradient-level mapping (id → {original_color: bw_color})
GRADIENT_MAP = {
    "grad-primary": {"#2d4a7a": "#333333", "#1a2744": "#000000"},
    "grad-accent": {"#ff7b5f": "#999999", "#e8563a": "#666666"},
    "grad-teal": {"#38b2ac": "#cccccc", "#2c7a7b": "#999999"},
    "grad-topbar": {"#1a2744": "#000000", "#e8563a": "#666666"},
}

# Global (non-gradient) replacements — applied after gradient replacement
# so that gradient stop-colors are already handled.
# High-contrast BW: dark colors → black, accent → mid-grey, light → near-white.
GLOBAL_MAP = [
    ("#2d4a7a", "#333333"),
    ("#1a2744", "#000000"),
    ("#ff7b5f", "#999999"),
    ("#e8563a", "#666666"),
    ("#38b2ac", "#cccccc"),
    ("#2c7a7b", "#999999"),
    ("#cbd5e0", "#e0e0e0"),
    ("#718096", "#666666"),
    ("#2d3748", "#222222"),
    ("#e2e8f0", "#e0e0e0"),
]

# Text-contrast fixes: on light-gray (teal) backgrounds, swap white text to dark.
# Each entry is (old_substring, new_substring) applied to the full SVG text.
CONTRAST_FIXES = {
    # --- training_pipeline ---
    "training_pipeline": [
        # Step 1 – Contrastive Pair Generation (teal box)
        ('y="76" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="700">Contrastive Pair<',
         'y="76" text-anchor="middle" font-size="12" fill="#222222" font-weight="700">Contrastive Pair<'),
        ('y="93" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="700">Generation<',
         'y="93" text-anchor="middle" font-size="12" fill="#222222" font-weight="700">Generation<'),
        ('y="113" text-anchor="middle" font-size="9.5" fill="#ffffff" fill-opacity="0.85">500K',
         'y="113" text-anchor="middle" font-size="9.5" fill="#333333">500K'),
        ('y="127" text-anchor="middle" font-size="9.5" fill="#ffffff" fill-opacity="0.85">32 tools',
         'y="127" text-anchor="middle" font-size="9.5" fill="#333333">32 tools'),
        # Step 8 – Feature Interpretation (teal box)
        ('y="486" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="700">Feature Interpretation<',
         'y="486" text-anchor="middle" font-size="12" fill="#222222" font-weight="700">Feature Interpretation<'),
        ('fill="#ffffff" fill-opacity="0.85">LLM labeling',
         'fill="#333333">LLM labeling'),
        ('fill="#ffffff" fill-opacity="0.85">top-20 activating',
         'fill="#333333">top-20 activating'),
    ],
    # --- sae_architecture ---
    "sae_architecture": [
        ('y="225" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="700">features (sparse)<',
         'y="225" text-anchor="middle" font-size="12" fill="#222222" font-weight="700">features (sparse)<'),
        ('fill="#ffffff" fill-opacity="0.9">d_sae = 10,752<',
         'fill="#333333">d_sae = 10,752<'),
        ('fill="#ffffff" fill-opacity="0.85">~668 active (~6%)<',
         'fill="#333333">~668 active (~6%)<'),
    ],
    # --- inference_pipeline ---
    "inference_pipeline": [
        ('y="310" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="700">FEATURE THEMES<',
         'y="310" text-anchor="middle" font-size="12" fill="#222222" font-weight="700">FEATURE THEMES<'),
        ('fill="#ffffff" fill-opacity="0.8">pre-computed labels<',
         'fill="#333333">pre-computed labels<'),
        ('fill="#ffffff" font-weight="600">"internal data access"<',
         'fill="#222222" font-weight="600">"internal data access"<'),
        ('fill="#ffffff" font-weight="600">"financial analysis"<',
         'fill="#222222" font-weight="600">"financial analysis"<'),
        ('fill="#ffffff" font-weight="600">"temporal reasoning"<',
         'fill="#222222" font-weight="600">"temporal reasoning"<'),
        ('fill="#ffffff" fill-opacity="0.6">...</',
         'fill="#444444">...</'),
    ],
}


def _replace_gradient_colors(svg_text: str, color_map: dict) -> str:
    """Replace stop-color values inside specific gradient definitions."""
    for grad_id, mapping in color_map.items():
        # Match the <linearGradient id="...">...</linearGradient> block
        pattern = re.compile(
            rf'(<linearGradient\s+id="{re.escape(grad_id)}"[^>]*>)(.*?)(</linearGradient>)',
            re.DOTALL,
        )
        def _repl(m):
            inner = m.group(2)
            for orig, gray in mapping.items():
                inner = inner.replace(orig, gray)
            return m.group(1) + inner + m.group(3)
        svg_text = pattern.sub(_repl, svg_text)
    return svg_text


def convert_to_grayscale(svg_text: str, svg_name: str) -> str:
    """Convert a color SVG string to its grayscale version."""
    # 1. Gradient-level replacements (context-aware, no ambiguity)
    svg_text = _replace_gradient_colors(svg_text, GRADIENT_MAP)
    # 2. Global color replacements for remaining inline uses
    for orig, gray in GLOBAL_MAP:
        svg_text = svg_text.replace(orig, gray)
    # 3. Text-contrast fixes for light-gray backgrounds
    for old, new in CONTRAST_FIXES.get(svg_name, []):
        svg_text = svg_text.replace(old, new)
    return svg_text


def render_svgs_to_png(svg_paths: list[tuple[str, str]], out_dir: str):
    """Render a list of (name, svg_path) to PNG files in out_dir using Playwright."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, svg_path in svg_paths:
            page = browser.new_page(viewport={"width": 2200, "height": 1200})
            page.goto(f"file://{svg_path}", wait_until="networkidle")
            out_path = os.path.join(out_dir, f"{name}.png")
            page.locator("svg").screenshot(path=out_path, timeout=10000)
            print(f"  {name}.png")
            page.close()
        browser.close()


def main():
    parser = argparse.ArgumentParser(description="Render SVGs to PNGs for the paper.")
    parser.add_argument(
        "--mode",
        choices=["color", "bw"],
        required=True,
        help="color = render as-is; bw = apply grayscale mapping first",
    )
    args = parser.parse_args()

    print(f"Rendering SVGs ({args.mode})...")

    if args.mode == "color":
        pairs = []
        for name in SVG_NAMES:
            svg_path = os.path.join(IMAGES_DIR, f"{name}.svg")
            if os.path.exists(svg_path):
                pairs.append((name, svg_path))
        render_svgs_to_png(pairs, IMAGES_DIR)
    else:
        # Create temp dir for grayscale SVGs
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = []
            for name in SVG_NAMES:
                svg_path = os.path.join(IMAGES_DIR, f"{name}.svg")
                if not os.path.exists(svg_path):
                    continue
                with open(svg_path) as f:
                    svg_text = f.read()
                bw_text = convert_to_grayscale(svg_text, name)
                bw_path = os.path.join(tmpdir, f"{name}.svg")
                with open(bw_path, "w") as f:
                    f.write(bw_text)
                pairs.append((name, bw_path))
            render_svgs_to_png(pairs, IMAGES_DIR)

        # Also convert non-SVG PNGs (charts, demo) to grayscale
        try:
            from PIL import Image
        except ImportError:
            print("  (pillow not available — skipping non-SVG grayscale conversion)")
            return

        svg_png_names = {f"{n}.png" for n in SVG_NAMES}
        for fname in os.listdir(IMAGES_DIR):
            if fname.endswith(".png") and fname not in svg_png_names:
                img_path = os.path.join(IMAGES_DIR, fname)
                img = Image.open(img_path)
                gray = img.convert("L").convert("RGB")
                gray.save(img_path)
                print(f"  {fname} (desaturated)")

    print("Done.")


if __name__ == "__main__":
    main()
