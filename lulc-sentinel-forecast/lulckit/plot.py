"""Shared figure style: class colours, map panels, recessive axes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .sites import CLASS_COLORS, CLASSES  # noqa: E402

INK = "#1f1f1f"
INK2 = "#55534e"
GRID = "#e4e2dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})


def class_cmap():
    cols = ["#ffffff"] + [CLASS_COLORS[k] for k in sorted(CLASSES)]
    return ListedColormap(cols), np.arange(-0.5, len(cols) + 0.5)


def show_map(ax, arr, title=None, extent=None):
    cmap, _ = class_cmap()
    a = np.ma.masked_equal(arr, 0)
    ax.imshow(a, cmap=cmap, vmin=-0.5, vmax=len(CLASSES) + 0.5, interpolation="nearest",
              extent=extent)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, color=INK, loc="left")


def class_legend(fig_or_ax, classes, **kw):
    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="none", label=CLASSES[k])
               for k in classes]
    return fig_or_ax.legend(handles=handles, **kw)


def scalebar(ax, res_m, length_km, loc=(0.05, 0.05)):
    """Simple scale bar for an imshow axis in pixel coordinates."""
    px = length_km * 1000 / res_m
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    w = ax.get_xlim()[1] - x0
    h = ax.get_ylim()[1] - y0
    x = x0 + loc[0] * w
    y = y0 + loc[1] * h
    ax.plot([x, x + px], [y, y], color=INK, lw=2, solid_capstyle="butt")
    ax.text(x + px / 2, y + 0.02 * h, f"{length_km:g} km", ha="center", va="bottom",
            color=INK, fontsize=8)


def save(fig, path):
    fig.savefig(str(path) + ".png")
    fig.savefig(str(path) + ".pdf")
    plt.close(fig)
