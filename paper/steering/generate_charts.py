#!/usr/bin/env python3
"""Figures for the steering paper.

Every chart is driven by ``results/steering_report.json`` — run
``extract_results.py`` first.  Palette and styling follow ``../generate_charts.py``
so the two papers look like siblings.

Usage:
    uv run --no-project --with matplotlib --with numpy python3 \
        paper/steering/generate_charts.py [--mode color|bw]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / "results" / "steering_report.json"
IMAGES = HERE / "images"

COLORS = {
    "primary": "#1a2744",
    "primary_light": "#2d4a7a",
    "accent": "#e8563a",
    "accent_light": "#ff7b5f",
    "text": "#2d3748",
    "text_light": "#718096",
    "border": "#e2e8f0",
}
BW_COLORS = {
    "primary": "#000000",
    "primary_light": "#444444",
    "accent": "#000000",
    "accent_light": "#777777",
    "text": "#000000",
    "text_light": "#555555",
    "border": "#999999",
}

DPI = 400
bw_mode = False

# Charts are authored at roughly the LNCS text width (4.8 in) so that scaling to
# \textwidth barely shrinks them.  Sizes are set once here rather than per-call:
# set_xlabel/set_ylabel take rcParams, not tick_params.
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.titlesize": 9,
    }
)
LAYERS = [6, 13, 20, 27, 34, 43]
SCENARIOS = ["tool_selection", "supply_chain", "customer_support"]
NICE = {
    "tool_selection": "tool selection",
    "supply_chain": "supply chain",
    "customer_support": "customer support",
}


def C(key: str) -> str:
    return (BW_COLORS if bw_mode else COLORS)[key]


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C("border"))
    ax.spines["bottom"].set_color(C("border"))
    ax.tick_params(colors=C("text"), labelsize=7)
    ax.yaxis.label.set_color(C("text"))
    ax.xaxis.label.set_color(C("text"))


def save(fig, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(HERE.parents[1])}")


# ---------------------------------------------------------------------------
# Figure 1: causal leverage against depth
# ---------------------------------------------------------------------------
def chart_depth(rep: dict) -> None:
    """Fraction of interventions that flip the tool, per layer, per scenario.

    Plotted as a rate rather than a count because the three scenarios contribute
    different numbers of pairs (7, 4, 6).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    markers = ["o", "s", "^"]
    styles = ["-", "--", ":"]
    shades = ["primary", "accent", "primary_light"]

    for ax, key, denom, title in (
        (ax1, "crossPatchFlips", "crossPatchDirections", "Cross-patch"),
        (ax2, "ablationFlips", "ablationSides", "Ablation"),
    ):
        for i, sc in enumerate(SCENARIOS):
            layers = rep["scenarios"][sc]["layers"]
            xs = [l for l in LAYERS if str(l) in layers]
            ys = [
                100.0 * layers[str(l)][key] / max(1, layers[str(l)][denom]) for l in xs
            ]
            ax.plot(
                xs,
                ys,
                marker=markers[i],
                linestyle=styles[i],
                color=C(shades[i]),
                linewidth=1.4,
                markersize=4,
                label=NICE[sc],
            )
        ax.axvspan(5, 21, color=C("border"), alpha=0.45, zorder=0)
        ax.set_xlabel("SAE layer (of 52)")
        ax.set_title(title, color=C("text"), fontsize=7, pad=8)
        ax.set_xticks(LAYERS)
        style_ax(ax)

    ax1.set_ylabel("directed flips (% of interventions)")
    ax1.text(
        13,
        62,
        "inert:\n0 / 204",
        ha="center",
        va="center",
        fontsize=7,
        color=C("text_light"),
    )
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Causal leverage over tool choice is a late-layer phenomenon",
        color=C("text"),
        fontsize=9.5,
        y=1.02,
    )
    save(fig, "chart_depth")


# ---------------------------------------------------------------------------
# Figure 2: dose-response for the single strongest feature
# ---------------------------------------------------------------------------
def chart_dose(rep: dict) -> None:
    """One feature, clamped at multiples of its donor-side value.

    Two shaded bands, because the panel carries two curves and they need
    different references.  The inner band is the random control matched to one
    cue family -- the comparison for the single-feature curve.  The outer band
    is matched to the whole cue set's count and mass, which is what the
    all-features curve clamps; comparing that curve with the inner band would
    hold it to a lighter perturbation than the one it makes.
    """
    rows = rep["scenarios"]["supply_chain"]["dose"]
    row = next(r for r in rows if r["pair"] == "demand_pull_vs_supply_push" and r["direction"] == "a_into_b")
    best = row["bestSingleFeature"]

    scales = sorted(float(s) for s in row["curve"])
    ys_all = [row["curve"][str(s) if str(s) in row["curve"] else s] for s in scales]
    ys_one = [best["curve"][str(s) if str(s) in best["curve"] else s] for s in scales]
    band = row["controlBand"]
    set_band = row.get("setControlBand")
    base = row["baselineP"]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    if set_band:
        ax.fill_between(
            scales,
            [max(0.0, base - b) for b in set_band],
            [min(1.0, base + b) for b in set_band],
            color=C("border"),
            alpha=0.35,
            label=f"random control matched to all {row['numFeatures']} cue features",
        )
    ax.fill_between(
        scales,
        [max(0.0, base - b) for b in band],
        [min(1.0, base + b) for b in band],
        color=C("border"),
        alpha=0.7,
        label="random control matched to one cue family",
    )
    ax.axhline(0.5, color=C("text_light"), linewidth=1, linestyle=":")
    ax.plot(scales, ys_all, marker="o", color=C("primary"), linewidth=1.4, label=f"all {row['numFeatures']} cue features")
    ax.plot(
        scales,
        ys_one,
        marker="s",
        linestyle="--",
        color=C("accent"),
        linewidth=1.4,
        label=f"feature {best['index']} alone",
    )
    ax.annotate(
        "tool flips",
        xy=(1.0, ys_one[scales.index(1.0)]),
        xytext=(1.35, 0.30),
        fontsize=7,
        color=C("text"),
        arrowprops=dict(arrowstyle="->", color=C("text_light"), lw=1),
    )
    ax.set_xlabel("clamp scale (× the value the feature takes on the donor request)")
    ax.set_ylabel("p(inventory_manager)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(scales)
    ax.set_title(
        "Adding one feature to an unmodified request flips the tool",
        color=C("text"),
        fontsize=7,
        pad=10,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_ax(ax)
    save(fig, "chart_dose")


# ---------------------------------------------------------------------------
# Figure 3: the degeneracy signature
# ---------------------------------------------------------------------------
def chart_degeneracy(rep: dict) -> None:
    """Why reconstruction quality is the wrong health check.

    Left: explained variance per layer — customer support peaks exactly where
    the dictionary is unusable.  Right: cross-patch and ablation agree on
    healthy layers and invert on the collapsed one.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    fig.subplots_adjust(wspace=0.28)

    markers = ["o", "s", "^"]
    styles = ["-", "--", ":"]
    shades = ["primary", "accent", "primary_light"]
    for i, sc in enumerate(SCENARIOS):
        d = rep["scenarios"][sc]["dictionary"]
        xs = [l for l in LAYERS if str(l) in d]
        ax1.plot(
            xs,
            [d[str(l)]["explainedVariance"] for l in xs],
            marker=markers[i],
            linestyle=styles[i],
            color=C(shades[i]),
            linewidth=1.4,
            markersize=4,
            label=NICE[sc],
        )
    # Label the collapsed point tersely and let the caption carry the numbers —
    # at LNCS width a multi-line callout collides with the legend.
    cs43 = rep["scenarios"]["customer_support"]["dictionary"]["43"]
    ax1.annotate(
        "degenerate",
        xy=(43, cs43["explainedVariance"]),
        xytext=(-2, 11),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color=C("accent"),
    )
    ax1.plot(
        [43],
        [cs43["explainedVariance"]],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor=C("accent"),
        markeredgewidth=1.2,
        zorder=5,
    )
    ax1.set_ylim(top=0.95)
    ax1.set_xlabel("SAE layer")
    ax1.set_ylabel("explained variance")
    ax1.set_xticks(LAYERS)
    ax1.set_title("Reconstruction quality", color=C("text"), fontsize=7, pad=14)
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    style_ax(ax1)

    cells = [
        ("supply\nchain\nL43", "supply_chain", 43),
        ("tool\nselection\nL43", "tool_selection", 43),
        ("customer\nsupport\nL34", "customer_support", 34),
        ("customer\nsupport\nL43", "customer_support", 43),
    ]
    x = np.arange(len(cells))
    cp = []
    ab = []
    for _, sc, l in cells:
        b = rep["scenarios"][sc]["layers"][str(l)]
        cp.append(100.0 * b["crossPatchFlips"] / b["crossPatchDirections"])
        ab.append(100.0 * b["ablationFlips"] / b["ablationSides"])
    w = 0.36
    ax2.bar(x - w / 2, cp, w, color=C("primary"), label="cross-patch")
    ax2.bar(x + w / 2, ab, w, color=C("accent"), label="ablation")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[0] for c in cells], fontsize=6.5)
    ax2.set_ylabel("directed flips (%)")
    ax2.set_title("Agreement, and where it breaks", color=C("text"), fontsize=7, pad=14)
    ax2.legend(frameon=False, fontsize=7)
    ax2.annotate(
        r"$9\times$ gap",
        xy=(3, max(cp[3], ab[3]) + 4),
        ha="center",
        fontsize=7,
        color=C("accent"),
    )
    style_ax(ax2)
    save(fig, "chart_degeneracy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["color", "bw"], default="color")
    args = ap.parse_args()
    global bw_mode
    bw_mode = args.mode == "bw"

    rep = json.loads(REPORT.read_text())
    print(f"Generating steering charts ({args.mode})...")
    chart_depth(rep)
    chart_dose(rep)
    chart_degeneracy(rep)


if __name__ == "__main__":
    main()
