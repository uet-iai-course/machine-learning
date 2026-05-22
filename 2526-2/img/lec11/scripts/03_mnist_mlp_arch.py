"""MLP kiến trúc cho MNIST: 784 → 16 → 16 → 10.

Vẽ 4 cột nodes với edges thưa, label số neurons + tên lớp.
Input/output column được vẽ rút gọn (3 trên, 3 dưới, dấu ...) để khả thi.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from _common import C_BLUE, C_RED, C_GRAY, C_VISIBLE, C_HIDDEN, C_OUTPUT, save_svg


def draw_layer(ax, x, n_display, n_total, color, label_n, label_name,
               show_dots=True, node_radius=0.14):
    """Draw a column of nodes centered at given x. Returns list of (x, y) of drawn nodes."""
    y_positions = np.linspace(2.5, -2.5, n_display)
    coords = []
    for y in y_positions:
        circ = Circle((x, y), node_radius, facecolor=color,
                      edgecolor="#333", lw=1.0, zorder=3)
        ax.add_patch(circ)
        coords.append((x, y))
    if show_dots and n_total > n_display:
        ax.text(x, -0.05, "⋮", ha="center", va="center",
                fontsize=14, color="#666", zorder=4)
    # Label: count + name
    ax.text(x, -3.25, label_n, ha="center", va="top",
            fontsize=12, fontweight="bold", color="#333")
    ax.text(x, -3.7, label_name, ha="center", va="top",
            fontsize=10, color="#666", style="italic")
    return coords


def draw_edges(ax, src, dst, alpha=0.18, color="#999"):
    for (x1, y1) in src:
        for (x2, y2) in dst:
            ax.plot([x1, x2], [y1, y2], color=color, lw=0.4,
                    alpha=alpha, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(9.6, 4.5))

    # Column x-positions — wider spacing so labels don't collide
    x_in, x_h1, x_h2, x_out = 0.0, 3.0, 6.0, 9.0

    # 4 layers
    in_coords = draw_layer(ax, x_in, n_display=7, n_total=784,
                           color=C_VISIBLE,
                           label_n="784", label_name="Lớp đầu vào")
    h1_coords = draw_layer(ax, x_h1, n_display=8, n_total=16,
                           color=C_HIDDEN, label_n="16",
                           label_name="Lớp ẩn 1", show_dots=False)
    h2_coords = draw_layer(ax, x_h2, n_display=8, n_total=16,
                           color=C_HIDDEN, label_n="16",
                           label_name="Lớp ẩn 2", show_dots=False)
    out_coords = draw_layer(ax, x_out, n_display=10, n_total=10,
                            color=C_OUTPUT, label_n="10",
                            label_name="Lớp đầu ra (chữ số 0–9)",
                            show_dots=False, node_radius=0.16)

    # Edges between layers
    draw_edges(ax, in_coords, h1_coords, alpha=0.22)
    draw_edges(ax, h1_coords, h2_coords, alpha=0.35)
    draw_edges(ax, h2_coords, out_coords, alpha=0.35)

    # Top labels — vector notation per layer
    ax.text(x_in, 3.15, "$x_1, \\ldots, x_{784}$", ha="center", fontsize=11,
            fontweight="bold", color=C_BLUE)
    ax.text(x_h1, 3.15, "$a^{(1)}$", ha="center", fontsize=12,
            fontweight="bold", color="#b58d00")
    ax.text(x_h2, 3.15, "$a^{(2)}$", ha="center", fontsize=12,
            fontweight="bold", color="#b58d00")
    ax.text(x_out, 3.15, "$\\hat{y}$", ha="center", fontsize=12,
            fontweight="bold", color=C_OUTPUT)

    # Forward pass arrow at bottom
    ax.annotate("", xy=(x_out + 0.5, -4.3), xytext=(x_in - 0.5, -4.3),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2))
    ax.text((x_in + x_out) / 2, -4.55, "lan truyền tiến",
            ha="center", va="top", fontsize=10, color=C_GRAY, style="italic")

    ax.set_xlim(-0.9, 9.9)
    ax.set_ylim(-4.9, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "mnist-mlp.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
