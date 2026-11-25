"""Figure style of the paper: Arial 7 pt, 8 pt bold panel labels, printed at 100 %."""
import logging

import matplotlib.pyplot as plt
import matplotlib._mathtext as _mathtext

TEXT_WIDTH_MM = 173.8          # \textwidth of the journal template
MM = 1.0 / 25.4

# Benchmark comparison (Fig. 7)
SIM = "#0072B2"
EXP = "#D55E00"
BLACK = "#222222"

def apply(*, interactive: bool = False):
    """Set the rcParams used by all figures."""
    if not interactive:
        plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    # sub- and superscripts at 6 pt next to 7 pt text (journal minimum)
    _mathtext.SHRINK_FACTOR = 6 / 7
    # fontTools warns about the old 'created' date in the Arial file when it embeds the font
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def figure(width_mm, height_mm):
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")


def panel_label(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="bottom")


def save(fig, path, *, show: bool = False):
    """Write path.pdf (vector, TrueType fonts) and path.png (600 dpi preview)."""
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=600)
    if show:
        plt.show()
    plt.close(fig)
    print("  figure:", path + ".pdf")
