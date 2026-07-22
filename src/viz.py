"""Matplotlib styling shared by all figures.

Palette is a colorblind-validated categorical set (fixed assignment order,
validated for CVD separation and normal-vision distance). Identity is never
carried by color alone: every multi-series figure also has a legend and/or
direct labels.
"""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

BLUE = "#2a78d6"    # series 1
GREEN = "#008300"   # series 2
MAGENTA = "#e87ba4" # series 3 (low surface contrast: always direct-label)
YELLOW = "#eda100"  # series 4 (low surface contrast: always direct-label)

SERIES = [BLUE, GREEN, MAGENTA, YELLOW]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def new_fig(width: float = 7.0, height: float = 4.2):
    return plt.subplots(figsize=(width, height))
