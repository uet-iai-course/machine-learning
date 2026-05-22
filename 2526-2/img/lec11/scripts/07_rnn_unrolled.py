"""RNN unrolled — cùng cell áp dụng qua các time step.

Layout:
- Hàng input x_1, x_2, x_3, x_4 ở dưới.
- Cell A ở giữa (hộp), state h_t truyền ngang.
- Hàng output y_1, y_2, y_3, y_4 ở trên.
- Mũi tên ngang giữa các A để chỉ "memory" h_{t-1} → h_t.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def main():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))

    T = 4
    x_step = 1.8
    y_in = 0
    y_cell = 1.4
    y_out = 2.8

    cell_w, cell_h = 1.0, 0.8
    box_w, box_h = 0.4, 0.4

    for t in range(1, T + 1):
        x = (t - 1) * x_step + 0.5
        # Input x_t (small square)
        ax.add_patch(Rectangle((x - box_w / 2, y_in - box_h / 2),
                               box_w, box_h, facecolor="#e0e0e0",
                               edgecolor=C_GRAY, lw=1.0))
        ax.text(x, y_in, f"$x_{t}$", ha="center", va="center", fontsize=11)

        # Cell A (RNN cell)
        ax.add_patch(Rectangle((x - cell_w / 2, y_cell - cell_h / 2),
                               cell_w, cell_h, facecolor="#bfe2eb",
                               edgecolor=C_BLUE, lw=1.4))
        ax.text(x, y_cell, "$A$", ha="center", va="center", fontsize=14, fontweight="bold")

        # Output y_t (small square)
        ax.add_patch(Rectangle((x - box_w / 2, y_out - box_h / 2),
                               box_w, box_h, facecolor="#fde2c4",
                               edgecolor="#e8732a", lw=1.0))
        ax.text(x, y_out, f"$\\hat y_{t}$", ha="center", va="center", fontsize=11)

        # x -> A
        ax.add_patch(FancyArrowPatch((x, y_in + box_h / 2),
                                     (x, y_cell - cell_h / 2),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#666", lw=1.2))
        # A -> y
        ax.add_patch(FancyArrowPatch((x, y_cell + cell_h / 2),
                                     (x, y_out - box_h / 2),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#666", lw=1.2))

        # Hidden state arrow to next cell
        if t < T:
            x_next = t * x_step + 0.5
            ax.add_patch(FancyArrowPatch((x + cell_w / 2, y_cell),
                                         (x_next - cell_w / 2, y_cell),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color=C_RED, lw=1.5))
            ax.text((x + x_next) / 2, y_cell + 0.32, f"$h_{t}$",
                    ha="center", fontsize=10, color=C_RED, style="italic")

    # Initial h_0 input from left
    x0 = 0.5
    ax.add_patch(FancyArrowPatch((-0.15, y_cell), (x0 - cell_w / 2, y_cell),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_RED, lw=1.5))
    ax.text(-0.05, y_cell + 0.32, "$h_0$", ha="center", fontsize=10,
            color=C_RED, style="italic")

    # x_t labels go BELOW the box (avoid collision with caption)
    # Note: x_t labels already rendered inside boxes via text() above.

    # Caption — well below x_t row
    ax.text(T * x_step / 2, -1.5,
            "Đầu vào theo thời gian — cùng một khối $A$ áp dụng lặp lại",
            ha="center", va="top", fontsize=10, color="#555", style="italic")

    ax.set_xlim(-0.6, T * x_step + 0.3)
    ax.set_ylim(-2.0, y_out + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "rnn-unrolled.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
