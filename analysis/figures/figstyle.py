"""Reusable publication figure helpers — figures4papers house style
(https://github.com/ChenLiu-1996/figures4papers, scientific-figure-making skill).

Implements the skill's API (apply_publication_style / PALETTE / make_lines /
make_grouped_bar / annotate_bars / finalize_figure) so report figures match the repo look:
helvetica/sans, no top/right spines, thick axes, frameless legend, vector-friendly export.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]]


def apply_publication_style(font_size=16, axes_linewidth=2.5):
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "axes.grid": False,
    })


def _ema(y, alpha=0.35):
    y = np.asarray(y, float)
    if len(y) == 0:
        return y
    out = np.empty_like(y); out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def make_lines(ax, series, ylabel=None, xlabel=None, colors=None, smooth=True,
               markers=True, raw_alpha=0.16, lw=2.6):
    """series: list of dicts {x, y, label}. Plots an (optionally EMA-smoothed) line per series,
    with faint raw points behind it. Handles different-length x per series (training curves)."""
    colors = colors or DEFAULT_COLORS
    for i, s in enumerate(series):
        c = s.get("color", colors[i % len(colors)])
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        if raw_alpha and smooth:
            ax.plot(x, y, color=c, alpha=raw_alpha, lw=1.2, zorder=1)
        yy = _ema(y) if smooth else y
        ax.plot(x, yy, color=c, lw=lw, label=s.get("label"), zorder=3,
                marker="o" if markers else None, markersize=4, markevery=max(1, len(x) // 12))
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None, annotate=False):
    colors = colors or DEFAULT_COLORS
    n_groups = len(series); n_cat = len(categories)
    width = 0.8 / n_groups
    x = np.arange(n_cat)
    last = None
    for g, (vals, lab) in enumerate(zip(series, labels)):
        offset = (g - (n_groups - 1) / 2) * width
        last = ax.bar(x + offset, vals, width, label=lab, color=colors[g % len(colors)],
                      edgecolor="black", linewidth=1.2)
        if annotate:
            annotate_bars(ax, last)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    return last


def make_single_bar(ax, categories, values, colors=None, ylabel="Value", annotate=True, fmt="{:.2f}"):
    colors = colors or DEFAULT_COLORS
    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(categories))],
                  edgecolor="black", linewidth=1.4, width=0.66)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if annotate:
        annotate_bars(ax, bars, fmt=fmt)
    return bars


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=12, padding=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, padding),
                    ha="center", va="bottom", fontsize=fontsize)


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, pad=0.08, close=True):
    base, _ = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout(pad=1.2)
    saved = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved
