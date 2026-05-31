"""Shared helpers for lec13 figure generation (matplotlib SVG).

Copy palette + style từ lec11/scripts/_common.py để đồng bộ toàn deck.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Palette (đồng bộ với lecture-style.css) ----
PALETTE = {
    "blue": "#1E93AB",     # primary — cấu trúc, định nghĩa
    "orange": "#e8732a",   # cảnh báo, nhấn
    "green": "#4a9d3f",    # ví dụ tốt, lựa chọn đúng
    "red": "#E62727",      # lỗi, sai
    "purple": "#8e4ec6",   # phụ
    "ink": "#1a2332",      # chữ chính
    "muted": "#6b7280",    # chữ phụ, caption
    "grid": "#cfd6dc",     # lưới, viền nhạt
    "bg_blue": "#f4fafc",
    "bg_orange": "#fef5ee",
    "bg_green": "#f1f8ef",
    "paper": "#ffffff",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "svg.fonttype": "none",
        "axes.edgecolor": "#1a2332",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#1a2332",
        "text.color": "#1a2332",
        "xtick.color": "#1a2332",
        "ytick.color": "#1a2332",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_svg(fig, path: str) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {path}")
