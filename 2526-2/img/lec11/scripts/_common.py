"""Shared style helpers for lecture 11 figures (Neural Networks).

Palette đồng bộ với deck (lecture-style.css):
    C_BLUE   #1E93AB — màu chính
    C_RED    #E62727 — màu nhấn / cảnh báo
    C_BG     #F3F2EC — nền nhạt
    C_GRAY   #666666 — màu trung tính
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BLUE = "#1E93AB"
C_RED = "#E62727"
C_BG = "#F3F2EC"
C_GRAY = "#666666"
C_LIGHT = "#cccccc"

# Visible cell / hidden cell colors for graph figures
C_VISIBLE = "#1E93AB"   # blue — visible/input neurons
C_HIDDEN = "#f1c232"    # warm yellow — hidden neurons
C_OUTPUT = "#e8732a"    # orange — output neurons
C_NODE_EDGE = "#444444"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
})


def save_svg(fig, path):
    """Save with consistent tight layout + transparent bg."""
    fig.savefig(path, format="svg", bbox_inches="tight",
                pad_inches=0.15, transparent=False, facecolor="white")
    plt.close(fig)
