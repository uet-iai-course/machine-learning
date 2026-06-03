"""LSTM cell — minh hoạ ĐÚNG luồng dữ liệu.

- Cell state (băng chuyền trên): c_{t-1} → ×(cổng quên f_t) → +(cổng nhập i_t)
  → c_t đi thẳng ra. Output gate KHÔNG nằm trên đường này.
- Hidden output: nhánh rẽ từ c_t → tanh → ×(cổng xuất o_t) → h_t.
- h_{t-1} và x_t là đầu vào, feed vào 3 cổng.

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    h_t = o_t ⊙ tanh(c_t)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from _common import C_BLUE, C_RED, save_svg

ORANGE = "#e8732a"
GRAY = "#777"

y_c = 3.4    # cell-state band (trên)
y_g = 1.9    # hàng cổng
y_h = 0.65   # hidden band (dưới)
x_tap = 6.5  # vị trí rẽ nhánh tính h_t


def main():
    fig, ax = plt.subplots(figsize=(9.4, 4.9))

    cell_box = FancyBboxPatch((0.2, 0.15), 7.9, 4.0,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor="#f7fbfc", edgecolor=C_BLUE, lw=1.5)
    ax.add_patch(cell_box)
    ax.text(4.15, 4.3, "LSTM cell tại bước $t$", ha="center", fontsize=11,
            fontweight="bold", color=C_BLUE)

    def op(x, y, sym):
        ax.add_patch(Circle((x, y), 0.17, facecolor="white",
                            edgecolor="#444", lw=1.1, zorder=6))
        ax.text(x, y, sym, ha="center", va="center", fontsize=12.5,
                fontweight="bold", zorder=7)

    def gate(x, label, vi, color):
        ax.add_patch(FancyBboxPatch((x - 0.4, y_g - 0.25), 0.8, 0.5,
                     boxstyle="round,pad=0.04", facecolor=color,
                     edgecolor="#444", lw=1.0, zorder=4))
        ax.text(x, y_g, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=5)
        ax.text(x, y_g - 0.42, vi, ha="center", va="top", fontsize=8.3,
                color="#555", style="italic")

    def varrow(x, y0, y1, color=GRAY, lw=1.2, ms=10):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=ms, color=color, lw=lw, zorder=3))

    # ---- Cell-state band: c_{t-1} → ×f → +i → c_t (đi thẳng) ----
    ax.add_patch(FancyArrowPatch((-0.3, y_c), (8.7, y_c), arrowstyle="-|>",
                 mutation_scale=15, color=ORANGE, lw=2.6, zorder=2))
    ax.text(-0.45, y_c, "$c_{t-1}$", ha="right", va="center", fontsize=11,
            color=ORANGE, fontweight="bold")
    ax.text(8.85, y_c, "$c_t$", ha="left", va="center", fontsize=11,
            color=ORANGE, fontweight="bold")
    op(1.5, y_c, "×")
    op(3.0, y_c, "+")

    # ---- Gates ----
    gate(1.5, "$f_t$", "Cổng quên", "#fde2c4")
    varrow(1.5, y_g + 0.25, y_c - 0.18)
    gate(3.0, "$i_t$", "Cổng nhập", "#bfe2eb")
    varrow(3.0, y_g + 0.25, y_c - 0.18)
    gate(5.0, "$o_t$", "Cổng xuất", "#d4e8c0")

    # ---- Output branch: c_t → tanh → ×o → h_t ----
    varrow(x_tap, y_c - 0.18, 2.78)                       # tap xuống
    ax.add_patch(FancyBboxPatch((x_tap - 0.45, 2.32), 0.9, 0.46,
                 boxstyle="round,pad=0.03", facecolor="#ececec",
                 edgecolor="#444", lw=1.0, zorder=4))
    ax.text(x_tap, 2.55, "tanh", ha="center", va="center", fontsize=9.5,
            zorder=5)
    varrow(x_tap, 2.32, y_g + 0.18)                       # tanh → ×o
    op(x_tap, y_g, "×")
    ax.add_patch(FancyArrowPatch((5.4, y_g), (x_tap - 0.18, y_g),
                 arrowstyle="-|>", mutation_scale=10, color=GRAY, lw=1.2,
                 zorder=3))                               # o_t → ×o
    # ×o → xuống → h_t band → ra phải
    ax.add_patch(FancyArrowPatch((x_tap, y_g - 0.18), (x_tap, y_h),
                 arrowstyle="-", color=C_RED, lw=2.0, zorder=2))
    ax.add_patch(FancyArrowPatch((x_tap, y_h), (8.7, y_h), arrowstyle="-|>",
                 mutation_scale=15, color=C_RED, lw=2.0, zorder=2))
    ax.text(8.85, y_h, "$h_t$", ha="left", va="center", fontsize=11,
            color=C_RED, fontweight="bold")

    # ---- Inputs: h_{t-1}, x_t → feed các cổng ----
    ax.text(-0.45, y_h, "$h_{t-1}$", ha="right", va="center", fontsize=11,
            color=C_RED, fontweight="bold")
    ax.add_patch(FancyArrowPatch((-0.3, y_h), (0.55, y_h), arrowstyle="-",
                 color=C_RED, lw=2.0, zorder=2))
    ax.text(0.55, -0.12, "$x_t$", ha="center", va="top", fontsize=12,
            fontweight="bold", color="#444")
    ax.add_patch(FancyArrowPatch((0.55, 0.02), (0.55, y_h), arrowstyle="-",
                 color=GRAY, lw=1.2, zorder=2))
    # bus xám + nhánh dashed lên từng cổng
    ax.plot([0.55, 5.0], [y_h, y_h], color=GRAY, lw=1.0, zorder=1)
    for gx in [1.5, 3.0, 5.0]:
        ax.plot([gx, gx], [y_h, y_g - 0.27], color="#999", lw=0.7,
                linestyle="--", zorder=1)

    ax.text(4.15, -0.62,
            "$c_t$ đi thẳng (chỉ ×quên, +nhập); còn $h_t = o_t \\cdot \\tanh(c_t)$",
            ha="center", va="top", fontsize=9.5, color="#555", style="italic")

    ax.set_xlim(-0.95, 9.35)
    ax.set_ylim(-1.0, 4.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "lstm-cell.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
