"""Scatter plots showing when each linkage method excels or fails."""
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

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

for row, (data_label, X, K) in enumerate(datasets):
    for col, (method, method_label, color) in enumerate(methods):
        ax = axes[row, col]
        Z = linkage(X, method=method)
        labels = fcluster(Z, t=K, criterion="maxclust")

        cmap = plt.cm.tab10
        for k in sorted(set(labels)):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=18, alpha=0.8,
                       color=cmap(k - 1), edgecolors="none")

        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(method_label, fontsize=11, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(data_label, fontsize=9, fontweight="bold")

        # Mark good/bad
        good = (row == 0 and method == "single") or \
               (row == 1 and method in ("complete", "average"))
        bad  = (row == 0 and method == "complete") or \
               (row == 1 and method == "single")
        if good:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ecc71")
                spine.set_linewidth(2.5)
        elif bad:
            for spine in ax.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(2.5)

fig.tight_layout(pad=0.6)
fig.savefig("../linkage-scatter.svg", format="svg", bbox_inches="tight")
fig.savefig("../linkage-scatter.png", format="png", dpi=150, bbox_inches="tight")
print("Saved linkage-scatter.svg + .png")
