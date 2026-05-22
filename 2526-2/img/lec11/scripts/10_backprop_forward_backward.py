"""Forward + backward pass minh hoạ — đơn giản, dễ hiểu.

2 panel xếp ngang:
- Trái: forward pass (input → hidden → output → loss), mũi tên màu xanh chỉ phải.
- Phải: backward pass (loss → output → hidden → input), mũi tên màu đỏ chỉ trái.
- Mỗi panel có 3 layer + 1 loss node.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def draw_layer(ax, x, n, color, label, label_color=None):
    if label_color is None:
        label_color = color
    ys = np.linspace(1.0, -1.0, n)
    coords = []
    for y in ys:
        c = Circle((x, y), 0.16, facecolor=color, edgecolor="#333", lw=1.0, zorder=3)
        ax.add_patch(c)
        coords.append((x, y))
    ax.text(x, -1.6, label, ha="center", va="top", fontsize=10,
            color=label_color, style="italic")
    return coords


def draw_loss_box(ax, x, y, color, edge, label_top, math_text):
    box = FancyBboxPatch((x - 0.4, y - 0.4), 0.8, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=edge, lw=1.4)
    ax.add_patch(box)
    ax.text(x, y + 0.15, label_top, ha="center", va="center",
            fontsize=10, fontweight="bold", color=edge)
    ax.text(x, y - 0.15, math_text, ha="center", va="center",
            fontsize=11, color="#333")


def panel(ax, direction, title, arrow_color, panel_color):
    """direction='forward' or 'backward'."""
    # Layer positions
    x_in, x_h, x_out, x_loss = 0.0, 1.6, 3.2, 4.6
    # Draw layers
    in_coords = draw_layer(ax, x_in, 4, "#bfe2eb", "Đầu vào")
    h_coords = draw_layer(ax, x_h, 5, "#fff4d2", "Ẩn", label_color="#b58d00")
    out_coords = draw_layer(ax, x_out, 3, "#fde2c4", "Đầu ra",
                            label_color="#e8732a")
    draw_loss_box(ax, x_loss, 0, "#ffe0e0", C_RED, "$\\mathcal{L}$", "mất mát")

    # Connect with light gray lines (always present, light)
    for src in in_coords:
        for dst in h_coords:
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color="#ddd", lw=0.4, zorder=1)
    for src in h_coords:
        for dst in out_coords:
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color="#ddd", lw=0.4, zorder=1)
    # out → loss
    for src in out_coords:
        ax.plot([src[0], x_loss - 0.4], [src[1], 0], color="#ddd", lw=0.4, zorder=1)

    # Direction arrows — bigger, colored
    arrow_y = 1.6
    if direction == "forward":
        # Left to right
        ax.add_patch(FancyArrowPatch((-0.4, arrow_y), (x_loss + 0.5, arrow_y),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color=arrow_color, lw=3.0))
        ax.text((x_in + x_loss) / 2, arrow_y + 0.35,
                r"Lan truyền tiến — tính $\hat y$ và $\mathcal{L}$",
                ha="center", fontsize=10, color=arrow_color, fontweight="bold")
    else:
        # Right to left
        ax.add_patch(FancyArrowPatch((x_loss + 0.5, arrow_y), (-0.4, arrow_y),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color=arrow_color, lw=3.0))
        ax.text((x_in + x_loss) / 2, arrow_y + 0.35,
                "Lan truyền ngược — chia gradient cho từng trọng số",
                ha="center", fontsize=10, color=arrow_color, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", color=arrow_color)
    ax.set_xlim(-0.7, x_loss + 0.8)
    ax.set_ylim(-2.0, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    panel(axes[0], "forward", "1. Lan truyền tiến", C_BLUE, "#e6f2f5")
    panel(axes[1], "backward", "2. Lan truyền ngược", C_RED, "#fef0f0")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "backprop-forward-backward.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
