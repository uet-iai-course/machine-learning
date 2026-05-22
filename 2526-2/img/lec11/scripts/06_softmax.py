"""Softmax — biến logits z thành phân phối xác suất.

2 panel side-by-side:
  trái: bar chart logits z_i (có thể âm, dương)
  phải: bar chart softmax(z) = e^{z_i} / Σ e^{z_j} — tổng = 1
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def main():
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # Logits demonstrating one strong class
    logits = np.array([0.5, -1.0, 2.0, 0.3, -0.5, 4.5, 1.0, -0.2, 0.8, -1.5])
    probs = softmax(logits)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    # Logits panel
    ax = axes[0]
    bars = ax.bar(classes, logits, color=C_BLUE, edgecolor="#0d4855",
                  linewidth=0.8, alpha=0.85)
    bars[5].set_color(C_RED)  # Highlight winner
    ax.axhline(0, color=C_GRAY, lw=0.6)
    ax.set_title("Logits $z_i$ (đầu ra trước softmax)")
    ax.set_ylabel("$z_i$")
    ax.set_xlabel("Lớp")
    ax.set_ylim(-2.5, 5.5)
    ax.grid(True, alpha=0.2, ls="--", axis="y")
    for i, v in enumerate(logits):
        ax.text(i, v + (0.15 if v >= 0 else -0.4), f"{v:.1f}",
                ha="center", fontsize=8, color="#333")

    # Probabilities panel
    ax = axes[1]
    bars = ax.bar(classes, probs, color=C_BLUE, edgecolor="#0d4855",
                  linewidth=0.8, alpha=0.85)
    bars[5].set_color(C_RED)
    ax.set_title("Sau softmax — xác suất $p_i$ (tổng = 1)")
    ax.set_ylabel("$p_i$")
    ax.set_xlabel("Lớp")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, ls="--", axis="y")
    for i, v in enumerate(probs):
        if v > 0.02:
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#333")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "softmax.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
