"""Không gian ẩn z: AE (rời rạc, có lỗ hổng) vs VAE (liên tục, khớp N(0,I)).

2 panel cạnh nhau, cùng khung toạ độ:
- AE  : 4 "đảo" mã tách biệt → khoảng trống ở giữa = vùng chết (decode ra rác).
- VAE : KL kéo mã về N(0,I) (mu->0 về tâm, sigma->1 nở ra) → các blob chồng lấn
        lấp đầy một đĩa Gauss quanh gốc → lấy mẫu z bất kỳ đều decode ra ảnh hợp lệ.

Dùng: .conda/bin/python img/lec13/scripts/latent_space.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PALETTE, apply_style, save_svg  # noqa: E402

OUT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(11)

CLASS_COLORS = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
LIM = 6.6


def _cloud(center, sigma, n):
    return RNG.normal(center, sigma, (n, 2))


def _style_ax(ax, title):
    ax.set_title(title, fontsize=13.5, color=PALETTE["ink"], pad=12)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("$z_1$", fontsize=11, color=PALETTE["muted"], labelpad=2)
    ax.set_ylabel("$z_2$", fontsize=11, color=PALETTE["muted"], labelpad=2)


def main():
    apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.9))

    # ---------- LEFT: Autoencoder — đảo rời rạc, có lỗ hổng ----------
    ae_centers = [(-3.9, 3.3), (3.8, 3.7), (-3.7, -3.5), (4.0, -3.1)]
    for c, col in zip(ae_centers, CLASS_COLORS):
        pts = _cloud(c, 0.45, 75)
        axL.scatter(pts[:, 0], pts[:, 1], s=15, color=col, alpha=0.72,
                    edgecolors="none", zorder=2)
    # lỗ hổng ở giữa: dấu X đỏ + chú thích
    axL.scatter([0.1], [0.2], marker="X", s=200, color=PALETTE["red"],
                edgecolors="white", linewidths=1.2, zorder=5)
    axL.annotate("lấy mẫu trúng\nlỗ hổng → ảnh rác", xy=(0.1, 0.2),
                 xytext=(0.1, -1.5), ha="center", va="top", fontsize=10.5,
                 color=PALETTE["red"], fontweight="bold")
    _style_ax(axL, "Autoencoder: mã rời rạc, có lỗ hổng")

    # ---------- RIGHT: VAE — liên tục, khớp N(0,I) ----------
    vae_centers = [(-0.95, 0.8), (0.95, 0.85), (-0.85, -0.8), (0.9, -0.75)]
    for c, col in zip(vae_centers, CLASS_COLORS):
        pts = _cloud(c, 1.0, 150)
        axR.scatter(pts[:, 0], pts[:, 1], s=15, color=col, alpha=0.5,
                    edgecolors="none", zorder=2)
    # vòng tham chiếu N(0,I): 1σ, 2σ
    for r in (1.0, 2.0):
        axR.add_patch(Circle((0, 0), r, fill=False, edgecolor=PALETTE["ink"],
                             lw=1.5, ls="--", alpha=0.6, zorder=4))
    axR.text(0.0, 2.2, r"$\mathcal{N}(0,\,I)$", ha="center", va="bottom",
             fontsize=12.5, color=PALETTE["ink"], fontweight="bold", zorder=5)
    # lấy mẫu bất kỳ: ngôi sao xanh lá + chú thích (góc dưới-phải trống)
    axR.scatter([1.7], [-1.6], marker="*", s=340, color=PALETTE["green"],
                edgecolors="white", linewidths=1.0, zorder=6)
    axR.annotate("lấy mẫu z bất kỳ\n→ ảnh hợp lệ", xy=(1.7, -1.6),
                 xytext=(4.6, -3.9), ha="center", va="top", fontsize=10.5,
                 color=PALETTE["green"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.6))
    _style_ax(axR, "VAE: liên tục, khớp $\\mathcal{N}(0,I)$")

    fig.tight_layout()
    save_svg(fig, str(OUT / "latent-space.svg"))


if __name__ == "__main__":
    main()
