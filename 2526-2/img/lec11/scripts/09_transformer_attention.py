"""Transformer self-attention — minh hoạ "mỗi từ nhìn mọi từ khác".

2 phần:
- Hàng tokens: "The cat sat on the mat"
- Attention matrix nhỏ phía dưới (6x6 heatmap) — màu càng đậm = chú ý
  càng nhiều. Highlight 1 hàng tiêu biểu (vd. cat → the/cat/sat).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def main():
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)

    # Synthetic attention matrix — emphasize "cat" attending strongly to "cat", "sat", a bit to "the", "on"
    np.random.seed(0)
    attn = np.array([
        [0.50, 0.10, 0.05, 0.05, 0.25, 0.05],  # The
        [0.20, 0.45, 0.20, 0.05, 0.05, 0.05],  # cat — heavier self + "sat"
        [0.05, 0.30, 0.40, 0.10, 0.05, 0.10],  # sat
        [0.05, 0.05, 0.10, 0.50, 0.05, 0.25],  # on
        [0.30, 0.05, 0.05, 0.10, 0.30, 0.20],  # the
        [0.05, 0.05, 0.10, 0.20, 0.20, 0.40],  # mat
    ])

    fig = plt.figure(figsize=(9.0, 4.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 2.4], hspace=0.35)

    # ---- Top: tokens + arrows from "cat" to all others ----
    ax_t = fig.add_subplot(gs[0])
    box_w = 1.0
    y_tok = 0.5
    positions = []
    for i, tok in enumerate(tokens):
        x = 1.5 + i * 1.2
        positions.append((x, y_tok))
        color = "#fde2c4" if i == 1 else "#bfe2eb"  # cat highlighted
        edge = "#e8732a" if i == 1 else C_BLUE
        ax_t.add_patch(Rectangle((x - box_w / 2, y_tok - 0.3), box_w, 0.6,
                                 facecolor=color, edgecolor=edge, lw=1.4))
        ax_t.text(x, y_tok, tok, ha="center", va="center", fontsize=11,
                  fontweight=("bold" if i == 1 else "normal"))

    # Arrows from "cat" (i=1) to other tokens — curved
    src = positions[1]
    for j, target in enumerate(positions):
        if j == 1:
            continue
        weight = attn[1, j]
        if weight < 0.06:
            continue
        rad = 0.4 if j > 1 else -0.4
        arr = FancyArrowPatch((src[0], src[1] + 0.32),
                              (target[0], target[1] + 0.32),
                              arrowstyle="-|>", mutation_scale=8,
                              color=C_RED, lw=0.6 + weight * 6, alpha=0.7,
                              connectionstyle=f"arc3,rad={rad}")
        ax_t.add_patch(arr)

    ax_t.text(0.5, y_tok + 1.4, "Từ 'cat' (cam) nhìn các từ khác — độ đậm = trọng số chú ý",
              fontsize=10, color="#444", style="italic", ha="left")

    ax_t.set_xlim(0, 9)
    ax_t.set_ylim(-0.2, 1.6)
    ax_t.set_aspect("equal")
    ax_t.axis("off")

    # ---- Bottom: attention heatmap ----
    ax_h = fig.add_subplot(gs[1])
    im = ax_h.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=0.55)
    ax_h.set_xticks(range(n))
    ax_h.set_xticklabels(tokens, fontsize=10)
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels(tokens, fontsize=10)
    ax_h.set_xlabel("Key (từ được nhìn)", fontsize=10)
    ax_h.set_ylabel("Query (từ đang nhìn)", fontsize=10)
    ax_h.set_title("Ma trận attention — mỗi ô = $\\mathrm{softmax}(Q K^\\top / \\sqrt{d})$",
                   fontsize=10)
    # Highlight cat row
    ax_h.add_patch(Rectangle((-0.5, 0.5), 6, 1, fill=False,
                             edgecolor=C_RED, lw=2.0))
    # Cell values
    for i in range(n):
        for j in range(n):
            v = attn[i, j]
            ax_h.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if v > 0.3 else "#333")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "transformer-attention.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
