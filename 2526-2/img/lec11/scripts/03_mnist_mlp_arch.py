"""MLP kiến trúc cho MNIST: 784 → 16 → 16 → 10.

Vẽ 4 cột nodes với edges thưa, label số neurons + tên lớp.
Input/output column được vẽ rút gọn (3 trên, 3 dưới, dấu ...) để khả thi.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from _common import C_BLUE, C_RED, C_GRAY, C_VISIBLE, C_HIDDEN, C_OUTPUT, save_svg


def draw_layer(ax, x, n_display, n_total, color, label, show_dots=True,
               node_radius=0.12):
    """Draw a column of nodes centered at given x. Returns list of (x, y) of drawn nodes."""
    y_positions = np.linspace(2.5, -2.5, n_display)
    coords = []
    for y in y_positions:
        circ = Circle((x, y), node_radius, facecolor=color,
                      edgecolor="#333", lw=1.0, zorder=3)
        ax.add_patch(circ)
        coords.append((x, y))
    if show_dots and n_total > n_display:
        # Dots in middle if many neurons elided
        mid_idx = n_display // 2
        ax.text(x, -0.05, "⋮", ha="center", va="center",
                fontsize=14, color="#666", zorder=4)
    # Label number of neurons
    ax.text(x, -3.2, f"{n_total}", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#333")
    ax.text(x, -3.6, label, ha="center", va="top",
            fontsize=9.5, color="#666", style="italic")
    return coords


def draw_edges(ax, src, dst, alpha=0.18, color="#999"):
    for (x1, y1) in src:
        for (x2, y2) in dst:
            ax.plot([x1, x2], [y1, y2], color=color, lw=0.4,
                    alpha=alpha, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))

    # Column x-positions
    x_in, x_h1, x_h2, x_out = 0.0, 2.4, 4.8, 7.2

    # 4 layers (display fewer nodes than total for clarity)
    in_coords = draw_layer(ax, x_in, n_display=7, n_total=784,
                           color=C_VISIBLE, label="Input layer (28×28 pixels)")
    h1_coords = draw_layer(ax, x_h1, n_display=8, n_total=16,
                           color=C_HIDDEN, label="Hidden 1", show_dots=False)
    h2_coords = draw_layer(ax, x_h2, n_display=8, n_total=16,
                           color=C_HIDDEN, label="Hidden 2", show_dots=False)
    out_coords = draw_layer(ax, x_out, n_display=10, n_total=10,
                            color=C_OUTPUT, label="Output (digit 0–9)",
                            show_dots=False, node_radius=0.14)

    # Edges between layers — thin gray, low alpha for "many connections" feel
    draw_edges(ax, in_coords, h1_coords, alpha=0.22)
    draw_edges(ax, h1_coords, h2_coords, alpha=0.35)
    draw_edges(ax, h2_coords, out_coords, alpha=0.35)

    # Top labels
    ax.text(x_in, 3.1, "x₁, …, x₇₈₄", ha="center", fontsize=10,
            fontweight="bold", color=C_BLUE)
    ax.text(x_h1, 3.1, "a⁽¹⁾", ha="center", fontsize=11,
            fontweight="bold", color="#b58d00")
    ax.text(x_h2, 3.1, "a⁽²⁾", ha="center", fontsize=11,
            fontweight="bold", color="#b58d00")
    ax.text(x_out, 3.1, "ŷ", ha="center", fontsize=11,
            fontweight="bold", color=C_OUTPUT)

    # Arrow showing direction
    ax.annotate("", xy=(7.7, -3.9), xytext=(-0.4, -3.9),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.0))
    ax.text(3.6, -4.1, "forward pass", ha="center", va="top",
            fontsize=9, color=C_GRAY, style="italic")

    ax.set_xlim(-0.8, 8.0)
    ax.set_ylim(-4.5, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "mnist-mlp.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
