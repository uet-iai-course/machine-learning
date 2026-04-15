"""Dendrogram-only figure for motivating hierarchical clustering."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=25, centers=4, cluster_std=0.75, random_state=12)
Z = linkage(X, method="complete")

# Two cut levels to show that K is chosen after the tree is built.
cut_high = 9.0
cut_low = 4.8
k_high = len(np.unique(fcluster(Z, cut_high, criterion="distance")))
k_low = len(np.unique(fcluster(Z, cut_low, criterion="distance")))

fig, ax = plt.subplots(figsize=(5.0, 3.6))
dendrogram(
    Z,
    ax=ax,
    color_threshold=0,
    above_threshold_color="#294c60",
    leaf_rotation=90,
    leaf_font_size=7,
)

ax.axhline(cut_high, color="#e68a2e", linestyle=(0, (4, 3)), linewidth=1.4)
ax.axhline(cut_low, color="#4f8f5b", linestyle=(0, (4, 3)), linewidth=1.4)

xmax = ax.get_xlim()[1]
ax.text(
    xmax * 0.98,
    cut_high + 0.18,
    f"Cắt cao \u2192 K = {k_high}",
    color="#e68a2e",
    ha="right",
    va="bottom",
    fontsize=8,
)
ax.text(
    xmax * 0.98,
    cut_low + 0.18,
    f"Cắt thấp \u2192 K = {k_low}",
    color="#4f8f5b",
    ha="right",
    va="bottom",
    fontsize=8,
)

ax.set_ylabel("Độ bất đồng")
ax.set_xlabel("Quan sát")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.6)
fig.savefig("../hierarchical-no-k.svg", format="svg", bbox_inches="tight")
print("Saved hierarchical-no-k.svg")
