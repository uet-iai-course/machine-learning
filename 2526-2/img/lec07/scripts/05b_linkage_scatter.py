"""Scatter plots showing when each linkage method excels or fails.

Row 1: concentric circles — Single handles non-convex shapes.
Row 2: compact blobs close together — Complete/Average resist chaining.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_circles, make_blobs

np.random.seed(42)

# --- Dataset 1: concentric circles (non-convex) ---
X_circles, _ = make_circles(n_samples=200, noise=0.05, factor=0.45, random_state=7)

# --- Dataset 2: compact blobs close together + bridge noise ---
centers = np.array([[-1.2, 0], [1.2, 0], [0, 2.0]])
X_blobs, _ = make_blobs(n_samples=150, centers=centers, cluster_std=0.45, random_state=12)
# Add a few bridge points between cluster 0 and 1 to trigger chaining
bridge = np.array([[0.0, 0.05], [-0.1, -0.1], [0.15, 0.0]])
X_blobs = np.vstack([X_blobs, bridge])

datasets = [
    ("Vòng tròn đồng tâm", X_circles, 2),
    ("Cụm compact gần nhau", X_blobs, 3),
]

methods = [
    ("complete", "Complete", "#4a90d9"),
    ("single",   "Single",   "#5aaa44"),
    ("average",  "Average",  "#e8a020"),
]

annotations = {
    (0, 0): (False, "✗ Chia theo nửa, không theo vòng"),
    (0, 1): (True,  "✓ Tách đúng vòng trong / ngoài"),
    (0, 2): (False, "✗ Chia theo nửa"),
    (1, 0): (True,  "✓ Tách gọn 3 cụm"),
    (1, 1): (False, "✗ Chuỗi nối 2 cụm qua điểm cầu"),
    (1, 2): (True,  "✓ Tách gọn 3 cụm"),
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
            ax.scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.8,
                       color=cmap(k - 1), edgecolors="none")

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(method_label, fontsize=12, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(data_label, fontsize=9, fontweight="bold")

        good, text = annotations[(row, col)]
        border_color = "#2ecc71" if good else "#e74c3c"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        text_color = "#1a7a3a" if good else "#c0392b"
        ax.text(0.5, 0.02, text, transform=ax.transAxes,
                fontsize=8.5, ha="center", va="bottom",
                fontweight="bold", color=text_color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=border_color, alpha=0.85, linewidth=1.2))

fig.tight_layout(pad=0.8)
fig.savefig("../linkage-scatter.svg", format="svg", bbox_inches="tight")
print("Saved linkage-scatter.svg")
