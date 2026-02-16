#!/usr/bin/env python3
"""Generate presentation charts for the mechanistic interpretability paper.

Produces four PNG charts in paper/images/ matching the presentation theme.
Requires matplotlib and numpy.

Usage:
    uv run --no-project --with matplotlib python3 paper/generate_charts.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Theme palette (matches theme.css)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1a2744",
    "primary_light": "#2d4a7a",
    "accent": "#e8563a",
    "accent_light": "#ff7b5f",
    "text": "#2d3748",
    "text_light": "#718096",
    "border": "#e2e8f0",
}

DEFAULT_DPI = 200
dpi = DEFAULT_DPI


def style_ax(ax: plt.Axes) -> None:
    """Apply shared axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])


# ---------------------------------------------------------------------------
# Chart 1: Feature ablation (grouped bar)
# ---------------------------------------------------------------------------
def chart_ablation(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))

    labels = [
        "fundamental\nvs. technical\n(p=0.002)",
        "single vs.\nmulti-tool\n(p=0.029)",
        "query vs.\nmutate\n(p=0.077)",
        "single-stock\nvs. portfolio\n(p=0.265)",
    ]
    contrastive = [10.1, 17.1, 77.8, 10.8]
    directed = [9.0, 2.4, 0.0, 0.0]
    random_rec = [0.0, 2.4, 33.3, 6.2]
    pvals = [0.002, 0.029, 0.077, 0.265]

    x = np.arange(len(labels))
    w = 0.24

    bar_groups = [
        ax.bar(
            x - w,
            contrastive,
            w,
            label="Contrastive ablation",
            color=COLORS["accent"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        ),
        ax.bar(
            x,
            directed,
            w,
            label="Directed flips",
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        ),
        ax.bar(
            x + w,
            random_rec,
            w,
            label="Random / Recon.",
            color=COLORS["text_light"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        ),
    ]

    ax.set_ylabel("Prediction flip rate (%)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(
        fontsize=11, frameon=True, fancybox=True, edgecolor=COLORS["border"], loc="upper right"
    )
    ax.set_title(
        "Feature Ablation: Causal Evidence",
        fontsize=16,
        fontweight="bold",
        color=COLORS["primary"],
        pad=14,
    )

    # Significance stars
    for i, p in enumerate(pvals):
        if p < 0.05:
            stars = "**" if p < 0.01 else "*"
            ymax = max(contrastive[i], directed[i], random_rec[i])
            ax.annotate(
                stars,
                xy=(x[i] - w, ymax + 1.5),
                fontsize=16,
                fontweight="bold",
                color=COLORS["accent"],
                ha="center",
                va="bottom",
            )

    # Value labels
    for bars in bar_groups:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COLORS["text"],
                )

    ax.axhline(y=0, color=COLORS["border"], linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, color=COLORS["border"], zorder=0)
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(outdir / "chart_ablation.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  chart_ablation.png")


# ---------------------------------------------------------------------------
# Chart 2: Layer sweep (dual-axis)
# ---------------------------------------------------------------------------
def chart_layer_sweep(outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    layers = [8, 16, 20, 32, 44]
    alive = [85.3, 63.2, 81.2, 69.5, 72.0]
    dead = [0.10, 0.85, 0.19, 2.00, 1.01]
    mse = [0.031, 0.333, 0.574, 1524, 508]

    ax2 = ax1.twinx()

    bar_colors = [COLORS["accent"] if l == 20 else COLORS["primary_light"] for l in layers]
    ax1.bar(
        range(len(layers)),
        alive,
        0.5,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        alpha=0.85,
    )
    ax1.set_ylabel("Alive features (%)", fontsize=13, fontweight="bold", color=COLORS["primary"])
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([str(l) for l in layers], fontsize=12)
    ax1.set_xlabel("Transformer Layer", fontsize=13, fontweight="bold")

    # MSE line (log scale)
    ax2.plot(
        range(len(layers)), mse, "o-", color=COLORS["accent"], linewidth=2.5, markersize=9, zorder=4
    )
    ax2.set_ylabel(
        "Reconstruction MSE (log scale)", fontsize=13, fontweight="bold", color=COLORS["accent"]
    )
    ax2.set_yscale("log")
    ax2.set_ylim(0.01, 5000)

    # MSE value annotations
    for i, (m, l) in enumerate(zip(mse, layers)):
        offset = -16 if l == 44 else 12
        label = f"{m:.3f}" if m < 1 else f"{m:,.0f}"
        ax2.annotate(
            label,
            xy=(i, m),
            fontsize=9,
            color=COLORS["accent"],
            ha="center",
            fontweight="bold",
            xytext=(0, offset),
            textcoords="offset points",
        )

    # Dead % inside bars
    for i, d in enumerate(dead):
        ax1.text(
            i,
            alive[i] - 4,
            f"dead: {d:.2f}%",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    # Highlight selected layer
    ax1.annotate(
        "selected",
        xy=(2, alive[2] + 1),
        fontsize=11,
        fontweight="bold",
        color=COLORS["accent"],
        ha="center",
        va="bottom",
        xytext=(0, 8),
        textcoords="offset points",
    )

    ax1.set_title(
        "Layer Sweep: Alive Features vs. Reconstruction Error",
        fontsize=15,
        fontweight="bold",
        color=COLORS["primary"],
        pad=14,
    )
    ax1.grid(axis="y", alpha=0.2, color=COLORS["border"], zorder=0)
    style_ax(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["accent"])
    ax2.tick_params(axis="y", colors=COLORS["accent"], labelsize=10)

    fig.tight_layout()
    fig.savefig(outdir / "chart_layer_sweep.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  chart_layer_sweep.png")


# ---------------------------------------------------------------------------
# Chart 3: Fuzzing quality tiers (donut)
# ---------------------------------------------------------------------------
def chart_fuzzing_tiers(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))

    sizes = [84.3, 9.4, 6.2]
    tier_labels = [
        "Excellent (>0.8)\n339 features",
        "Good (0.6-0.8)\n38 features",
        "Poor (<0.6)\n25 features",
    ]
    colors = [COLORS["primary"], COLORS["primary_light"], COLORS["text_light"]]
    explode = (0.03, 0.03, 0.03)

    _, texts, autotexts = ax.pie(
        sizes,
        labels=tier_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in texts:
        t.set_fontsize(11)
        t.set_color(COLORS["text"])
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.text(
        0,
        0,
        "91.2%\ncombined\nscore",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["primary"],
    )
    ax.set_title(
        "Feature Label Quality (Token-Level Fuzzing)",
        fontsize=14,
        fontweight="bold",
        color=COLORS["primary"],
        pad=16,
    )

    fig.tight_layout()
    fig.savefig(outdir / "chart_fuzzing_tiers.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  chart_fuzzing_tiers.png")


# ---------------------------------------------------------------------------
# Chart 4: Baselines comparison (horizontal bar)
# ---------------------------------------------------------------------------
def chart_baselines(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))

    methods = [
        "SAE\n(combined fuzzing)",
        "Linear Probe\n(accuracy)",
        "Linear Probe\n(macro F1)",
        "PCA+k-means\n(NMI)",
        "PCA+k-means\n(ARI)",
    ]
    values = [0.912, 0.796, 0.765, 0.225, 0.068]
    bar_colors = [
        COLORS["accent"],
        COLORS["primary"],
        COLORS["primary"],
        COLORS["text_light"],
        COLORS["text_light"],
    ]

    y = np.arange(len(methods))
    bars = ax.barh(y, values, 0.55, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlabel("Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(
            v + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
        )

    # Interpretability annotations (spaced away from bars)
    ax.annotate(
        "interpretable",
        xy=(0.70, 0.5),
        fontsize=9,
        color=COLORS["accent"],
        fontweight="bold",
        ha="right",
        annotation_clip=False,
    )
    ax.annotate(
        "not interpretable",
        xy=(0.73, 1.55),
        fontsize=9,
        color=COLORS["primary_light"],
        fontweight="bold",
        ha="right",
    )

    ax.set_title(
        "Method Comparison: Accuracy vs. Interpretability",
        fontsize=15,
        fontweight="bold",
        color=COLORS["primary"],
        pad=14,
    )
    ax.grid(axis="x", alpha=0.3, color=COLORS["border"], zorder=0)
    ax.axvline(
        x=0.5,
        color=COLORS["accent_light"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Random baseline",
    )
    ax.legend(
        fontsize=10, loc="lower right", frameon=True, fancybox=True, edgecolor=COLORS["border"]
    )
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(outdir / "chart_baselines.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  chart_baselines.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CHARTS = {
    "ablation": chart_ablation,
    "layer_sweep": chart_layer_sweep,
    "fuzzing_tiers": chart_fuzzing_tiers,
    "baselines": chart_baselines,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation charts.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent / "images",
        help="Output directory for PNGs (default: paper/images/)",
    )
    parser.add_argument(
        "--only",
        choices=list(CHARTS.keys()),
        nargs="+",
        help="Generate only specific charts (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output resolution (default: {DEFAULT_DPI})",
    )
    args = parser.parse_args()

    global dpi
    dpi = args.dpi

    args.outdir.mkdir(parents=True, exist_ok=True)
    targets = args.only or list(CHARTS.keys())

    print(f"Generating charts in {args.outdir}/")
    for name in targets:
        CHARTS[name](args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
