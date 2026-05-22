"""LSTM cell — minh hoạ 3 cổng (quên/nhập/xuất) + cell state băng chuyền.

Layout đơn giản hoá:
- Cell state c_{t-1} → c_t ở đầu (băng chuyền ngang).
- 3 cổng vẽ bên dưới: quên (forget), nhập (input), xuất (output).
- Mỗi cổng là σ block + arrow nhân vào cell state.
- Input x_t và hidden h_{t-1} feed vào.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def main():
    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    # Cell boundary (rounded box)
    cell_box = FancyBboxPatch((0.2, 0.3), 7.8, 3.7,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor="#f7fbfc", edgecolor=C_BLUE, lw=1.5)
    ax.add_patch(cell_box)
    ax.text(4.1, 4.15, "LSTM cell tại bước $t$", ha="center", fontsize=11,
            fontweight="bold", color=C_BLUE)

    # Cell state c band (top)
    y_c = 3.2
    ax.add_patch(FancyArrowPatch((-0.3, y_c), (8.6, y_c),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color="#e8732a", lw=2.5))
    ax.text(-0.45, y_c, "$c_{t-1}$", ha="right", va="center",
            fontsize=11, color="#e8732a", fontweight="bold")
    ax.text(8.75, y_c, "$c_t$", ha="left", va="center",
            fontsize=11, color="#e8732a", fontweight="bold")

    # Forget gate (left)
    def gate(x, label, vi_label, color):
        gate_box = FancyBboxPatch((x - 0.4, 1.5), 0.8, 0.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor="#444", lw=1.0)
        ax.add_patch(gate_box)
        ax.text(x, 1.75, label, ha="center", va="center", fontsize=10,
                fontweight="bold")
        ax.text(x, 1.2, vi_label, ha="center", va="top", fontsize=8.5,
                color="#555", style="italic")
        # Arrow up to cell state band
        circ = Circle((x, y_c), 0.16, facecolor="white",
                      edgecolor="#444", lw=1.0, zorder=4)
        ax.add_patch(circ)
        op_symbol = "×" if "quên" in vi_label or "xuất" in vi_label else "+"
        ax.text(x, y_c, op_symbol, ha="center", va="center", fontsize=12,
                fontweight="bold", zorder=5)
        ax.add_patch(FancyArrowPatch((x, 2.0), (x, y_c - 0.15),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#666", lw=1.2))

    gate(1.5, "$f_t$", "Cổng quên", "#fde2c4")     # forget
    gate(3.6, "$i_t$", "Cổng nhập", "#bfe2eb")    # input
    gate(6.5, "$o_t$", "Cổng xuất", "#d4e8c0")    # output

    # Hidden state output (bottom)
    y_h = 0.7
    ax.add_patch(FancyArrowPatch((-0.3, y_h), (8.6, y_h),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=C_RED, lw=2.0))
    ax.text(-0.45, y_h, "$h_{t-1}$", ha="right", va="center",
            fontsize=11, color=C_RED, fontweight="bold")
    ax.text(8.75, y_h, "$h_t$", ha="left", va="center",
            fontsize=11, color=C_RED, fontweight="bold")

    # Input x_t from below
    ax.add_patch(FancyArrowPatch((4.1, -0.4), (4.1, y_h - 0.15),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#666", lw=1.2))
    ax.text(4.1, -0.55, "$x_t$", ha="center", va="top", fontsize=12,
            fontweight="bold", color="#444")

    # Connection from h_{t-1} / x_t feeds to gates
    for x in [1.5, 3.6, 6.5]:
        ax.plot([x, x], [y_h + 0.1, 1.45], color="#999", lw=0.6,
                linestyle="--", zorder=1)

    # Caption
    ax.text(4.1, -1.1,
            "3 cổng học cách quên / nhập / xuất → giữ thông tin dài hạn",
            ha="center", va="top", fontsize=10, color="#555", style="italic")

    ax.set_xlim(-0.9, 9.2)
    ax.set_ylim(-1.4, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "lstm-cell.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
