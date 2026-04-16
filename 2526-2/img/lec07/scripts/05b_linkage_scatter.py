"""Scatter plots showing when each linkage method excels or fails.

Each panel has ✓/✗ annotation explaining why the result is good or bad.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_moons, make_blobs

np.random.seed(42)

# --- Two datasets ---
X_moons, _ = make_moons(n_samples=120, noise=0.06, random_state=7)
X_blobs, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.55, random_state=12)

datasets = [
    ("Dữ liệu hình lưỡi liềm", X_moons, 2),
    ("Dữ liệu cụm compact", X_blobs, 3),
]

methods = [
    ("complete", "Complete", "#4a90d9"),
    ("single",   "Single",   "#5aaa44"),
    ("average",  "Average",  "#e8a020"),
]

# Annotations: (row, col) → (good, text)
annotations = {
    (0, 0): (False, "✗ Cắt ngang lưỡi liềm"),
    (0, 1): (True,  "✓ Tách đúng theo hình dạng"),
    (0, 2): (False, "✗ Không theo được hình cong"),
    (1, 0): (True,  "✓ Cụm tròn, tách gọn"),
    (1, 1): (False, "✗ Hiệu ứng chuỗi"),
    (1, 2): (True,  "✓ Cụm tròn, tách gọn"),
}

fig, axes = plt.subplots(2, 3, figsize=(11, 7))

for row, (data_label, X, K) in enumerate(datasets):
    for col, (method, method_label, color) in enumerate(methods):
        ax = axes[row, col]
        Z = linkage(X, method=method)
        labels = fcluster(Z, t=K, criterion="maxclust")

        cmap = plt.cm.tab10
        for k in sorted(set(labels)):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=22, alpha=0.8,
                       color=cmap(k - 1), edgecolors="none")

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(method_label, fontsize=12, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(data_label, fontsize=9, fontweight="bold")

        # Border color
        good, text = annotations[(row, col)]
        border_color = "#2ecc71" if good else "#e74c3c"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        # Annotation text inside panel
        text_color = "#1a7a3a" if good else "#c0392b"
        ax.text(0.5, 0.02, text, transform=ax.transAxes,
                fontsize=8.5, ha="center", va="bottom",
                fontweight="bold", color=text_color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=border_color, alpha=0.85, linewidth=1.2))

fig.tight_layout(pad=0.8)
fig.savefig("../linkage-scatter.svg", format="svg", bbox_inches="tight")
print("Saved linkage-scatter.svg")
