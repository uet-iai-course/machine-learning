"""Three linkage dendrograms on simulated microarray-like data (mimic ISLR Fig 12.11)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

rng = np.random.RandomState(17)
# Simulate 40 'genes' x 20 'samples' — two broad groups with substructure
n = 40
group1 = rng.randn(n // 2, 6) + np.array([2, -1, 0, 1, -2, 1])
group2 = rng.randn(n // 2, 6) + np.array([-2, 1, 0, -1, 2, -1])
X = np.vstack([group1, group2])

methods = [
    ("average",  "Average Linkage",  "#e8a020"),
    ("complete", "Complete Linkage", "#4a90d9"),
    ("single",   "Single Linkage",   "#5aaa44"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for ax, (method, title, col) in zip(axes, methods):
    Z = linkage(X, method=method)
    dendrogram(
        Z, ax=ax,
        color_threshold=0,
        above_threshold_color=col,
        link_color_func=lambda k: col,
        leaf_rotation=90,
        leaf_font_size=0,   # hide individual leaf labels (too many)
        show_leaf_counts=False,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Khoảng cách bất đồng" if ax is axes[0] else "")
    ax.set_xlabel("Quan sát (mẫu)")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=1.0)
fig.savefig("../linkage-tumor.svg", format="svg", bbox_inches="tight")
print("Saved linkage-tumor.svg")
