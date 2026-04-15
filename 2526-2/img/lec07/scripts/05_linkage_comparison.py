"""Compare average, single, and complete linkage dendrograms."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.9, random_state=12)

methods = [("average",  "Average Linkage",  "#e8a020"),
           ("single",   "Single Linkage",   "#5aaa44"),
           ("complete", "Complete Linkage", "#4a90d9")]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (method, title, col) in zip(axes, methods):
    Z = linkage(X, method=method)
    dendrogram(Z, ax=ax, color_threshold=0,
               above_threshold_color=col,
               leaf_rotation=90, leaf_font_size=6,
               link_color_func=lambda k: col)
    ax.set_title(title, fontsize=11, fontweight="bold", color=col)
    ax.set_ylabel("Khoảng cách" if ax is axes[0] else "")
    ax.tick_params(axis="x", labelsize=6)

fig.tight_layout(pad=0.8)
fig.savefig("../linkage-comparison.svg", format="svg", bbox_inches="tight")
print("Saved linkage-comparison.svg")
