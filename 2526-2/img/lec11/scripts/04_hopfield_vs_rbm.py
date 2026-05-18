"""Hopfield Network vs Restricted Boltzmann Machine — 2 panel side by side.

Hopfield: 8 neurons trên vòng tròn, fully-connected (every pair).
RBM: 2 lớp bipartite (4 visible bên trái + 4 hidden bên phải), chỉ nối visible↔hidden.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from _common import (C_BLUE, C_VISIBLE, C_HIDDEN, C_NODE_EDGE,
                     C_GRAY, save_svg)


def hopfield_panel(ax, n=8, radius=1.4):
    """Vẽ Hopfield network — n nodes trên vòng tròn, fully connected."""
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    pts = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    # Edges — every unordered pair
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            ax.plot([x1, x2], [y1, y2], color="#666",
                    lw=0.6, alpha=0.55, zorder=1)
    # Nodes
    for (x, y) in pts:
        c = Circle((x, y), 0.18, facecolor=C_VISIBLE,
                   edgecolor="#1a4855", lw=1.2, zorder=3)
        ax.add_patch(c)
    # Self-loops symbolic — small arc on each node
    for (x, y) in pts:
        # Tiny curved arrow to suggest recurrent connection
        rx, ry = x * 1.18, y * 1.18
        ax.annotate("", xy=(rx, ry), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#444",
                                    lw=0.7, alpha=0.7,
                                    connectionstyle="arc3,rad=0.2"),
                    zorder=2)

    ax.set_title("Hopfield Network", fontsize=12)
    ax.text(0, -2.0, "Mỗi cặp neuron liên kết — không lớp ẩn",
            ha="center", va="top", fontsize=9.5, color="#555", style="italic")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.4, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")


def rbm_panel(ax, n_v=4, n_h=4):
    """Vẽ RBM bipartite — visible bên trái, hidden bên phải, chỉ nối chéo."""
    # Visible neurons (left column)
    y_v = np.linspace(1.4, -1.4, n_v)
    visible = [(-1.1, y) for y in y_v]
    # Hidden neurons (right column)
    y_h = np.linspace(1.4, -1.4, n_h)
    hidden = [(1.1, y) for y in y_h]

    # Bipartite edges
    for (x1, y1) in visible:
        for (x2, y2) in hidden:
            ax.plot([x1, x2], [y1, y2], color="#666",
                    lw=0.7, alpha=0.6, zorder=1)

    # Visible nodes
    for (x, y) in visible:
        c = Circle((x, y), 0.18, facecolor=C_VISIBLE,
                   edgecolor="#1a4855", lw=1.2, zorder=3)
        ax.add_patch(c)
    # Hidden nodes
    for (x, y) in hidden:
        c = Circle((x, y), 0.18, facecolor=C_HIDDEN,
                   edgecolor="#7a6300", lw=1.2, zorder=3)
        ax.add_patch(c)

    # Layer labels
    ax.text(-1.1, 1.85, "Visible", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=C_VISIBLE)
    ax.text(1.1, 1.85, "Hidden", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#b58d00")

    ax.set_title("Restricted Boltzmann Machine", fontsize=12)
    ax.text(0, -2.0, "Tách 2 lớp — nối chéo, không nối trong cùng lớp",
            ha="center", va="top", fontsize=9.5, color="#555", style="italic")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.4, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    hopfield_panel(axes[0])
    rbm_panel(axes[1])
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "hopfield-vs-rbm.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
