"""CNN pipeline — chuỗi block từ input image → conv/pool stack → FC → softmax.

Layout: input → [Conv+ReLU → Pool] ×2 → Flatten → FC → Output.
Mỗi block là một hình hộp (đại diện feature maps), với label kích thước.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from _common import C_BLUE, C_RED, C_GRAY, C_VISIBLE, C_HIDDEN, C_OUTPUT, save_svg


def draw_cube(ax, x, y, w, h, depth, color, edge="#333"):
    """Draw a pseudo-3D cube to represent a feature map block."""
    # Back face (offset)
    back = Rectangle((x + depth * 0.4, y + depth * 0.5), w, h,
                     facecolor=color, edgecolor=edge, lw=1.0, alpha=0.55)
    ax.add_patch(back)
    # Front face
    front = Rectangle((x, y), w, h,
                      facecolor=color, edgecolor=edge, lw=1.2)
    ax.add_patch(front)
    # Connecting lines
    for (xs, ys) in [(x, y + h), (x + w, y + h), (x + w, y)]:
        ax.plot([xs, xs + depth * 0.4], [ys, ys + depth * 0.5],
                color=edge, lw=0.8)


def draw_arrow(ax, x1, x2, y, label=None, color=C_GRAY):
    arr = FancyArrowPatch((x1, y), (x2, y),
                          arrowstyle="-|>", mutation_scale=12,
                          color=color, lw=1.2)
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, y + 0.3, label, ha="center", va="bottom",
                fontsize=9, color="#555", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(10, 3.6))

    # Y center
    yc = 0
    h_box = 1.8

    # ===== Input image =====
    x = 0
    w_in = 1.0
    inp = Rectangle((x, yc - h_box / 2), w_in, h_box,
                    facecolor="#e6f2f5", edgecolor=C_BLUE, lw=1.3)
    ax.add_patch(inp)
    # Mini image icon — checkered pattern
    for ix in range(3):
        for iy in range(3):
            shade = "#1E93AB" if (ix + iy) % 2 == 0 else "#bfe2eb"
            ax.add_patch(Rectangle((x + 0.15 + ix * 0.23, yc - 0.35 + iy * 0.23),
                                   0.23, 0.23, facecolor=shade, edgecolor="none"))
    ax.text(x + w_in / 2, yc - h_box / 2 - 0.25, "Input\n28×28×1",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Conv1 + ReLU =====
    x1 = x + w_in + 0.7
    draw_arrow(ax, x + w_in + 0.05, x1 - 0.05, yc)
    draw_cube(ax, x1, yc - 0.75, 0.9, 1.5, 0.5, "#bfe2eb")
    ax.text(x1 + 0.65, yc - h_box / 2 - 0.25, "Conv + ReLU\n28×28×6",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Pool1 =====
    x2 = x1 + 1.7
    draw_arrow(ax, x1 + 1.4, x2 - 0.05, yc)
    draw_cube(ax, x2, yc - 0.55, 0.7, 1.1, 0.5, "#e0e0e0")
    ax.text(x2 + 0.55, yc - h_box / 2 - 0.25, "Pool\n14×14×6",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Conv2 + ReLU =====
    x3 = x2 + 1.4
    draw_arrow(ax, x2 + 1.2, x3 - 0.05, yc)
    draw_cube(ax, x3, yc - 0.45, 0.6, 0.9, 0.5, "#bfe2eb")
    ax.text(x3 + 0.5, yc - h_box / 2 - 0.25, "Conv + ReLU\n10×10×16",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Pool2 =====
    x4 = x3 + 1.2
    draw_arrow(ax, x3 + 1.0, x4 - 0.05, yc)
    draw_cube(ax, x4, yc - 0.3, 0.45, 0.6, 0.5, "#e0e0e0")
    ax.text(x4 + 0.45, yc - h_box / 2 - 0.25, "Pool\n5×5×16",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Flatten + FC =====
    x5 = x4 + 1.1
    draw_arrow(ax, x4 + 1.0, x5 - 0.05, yc, label="flatten")
    fc = Rectangle((x5, yc - 0.9), 0.32, 1.8,
                   facecolor="#fff4d2", edgecolor="#b58d00", lw=1.2)
    ax.add_patch(fc)
    ax.text(x5 + 0.16, yc - h_box / 2 - 0.25, "FC\n120",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Output / Softmax =====
    x6 = x5 + 1.0
    draw_arrow(ax, x5 + 0.42, x6 - 0.05, yc, label="softmax")
    out = Rectangle((x6, yc - 0.6), 0.28, 1.2,
                    facecolor="#fde2c4", edgecolor=C_OUTPUT, lw=1.2)
    ax.add_patch(out)
    ax.text(x6 + 0.14, yc - h_box / 2 - 0.25, "Output\n10 classes",
            ha="center", va="top", fontsize=9, color="#333")

    # ===== Stage labels (top) =====
    ax.text(2.85, yc + 1.5, "Feature learning", ha="center", fontsize=11,
            fontweight="bold", color=C_BLUE)
    ax.text(x5 + 0.16 + (x6 - x5) / 2, yc + 1.5, "Classification",
            ha="center", fontsize=11, fontweight="bold", color=C_OUTPUT)
    # Bracket lines
    ax.plot([x1, x4 + 0.85], [yc + 1.2, yc + 1.2], color=C_BLUE, lw=1.0)
    ax.plot([x5, x6 + 0.28], [yc + 1.2, yc + 1.2], color=C_OUTPUT, lw=1.0)

    ax.set_xlim(-0.3, x6 + 0.6)
    ax.set_ylim(-2.0, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "cnn-pipeline.svg"
    save_svg(fig, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
