"""Dendrogram example with horizontal cut and cluster colors."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=25, centers=3, cluster_std=0.8, random_state=5)
Z = linkage(X, method="complete")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: raw scatter
colors = ["#4a90d9", "#e8732a", "#5aaa44"]
for c in range(3):
    axes[0].scatter(X[y == c, 0], X[y == c, 1], s=40, color=colors[c], alpha=0.85)
axes[0].set_title("Dữ liệu gốc (3 nhóm thực tế)", fontsize=10)
axes[0].set_xlabel("$X_1$"); axes[0].set_ylabel("$X_2$")

# Right: dendrogram
cut_height = 6.5
dn = dendrogram(Z, ax=axes[1], color_threshold=cut_height,
                above_threshold_color="#888",
                leaf_rotation=90, leaf_font_size=7,
                link_color_func=lambda k: "#4a90d9")
axes[1].axhline(cut_height, color="#e8732a", linestyle="--", linewidth=1.5, label=f"Cắt ở {cut_height:.1f}")
axes[1].set_title("Dendrogram (Complete Linkage)", fontsize=10)
axes[1].set_ylabel("Khoảng cách bất đồng")
axes[1].set_xlabel("Quan sát")
axes[1].legend(fontsize=8)

fig.tight_layout(pad=1.0)
fig.savefig("../dendrogram-example.svg", format="svg", bbox_inches="tight")
print("Saved dendrogram-example.svg")
