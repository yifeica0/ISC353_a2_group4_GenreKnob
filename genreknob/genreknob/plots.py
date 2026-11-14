"""Shared matplotlib style for figures (light, print-friendly).

Palette: validated categorical slots 1-3 (blue, orange, aqua) and the blue
sequential ramp from the reference palette; text never takes series colours.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, GRAY = "#6250d6", "#8f8d87"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def ink_on(cmap, x):
    """Text colour (white/black) that stays legible on cmap(x), x in [0, 1]."""
    r, g, b, _ = cmap(float(min(max(x, 0.0), 1.0)))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else TEXT
