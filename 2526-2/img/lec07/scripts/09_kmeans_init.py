"""K-means initialization: 6 random starts with K=3, show W values."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=90, centers=3, cluster_std=1.1, random_state=42)
colors = ["#e8a020", "#5aaa44", "#c0392b"]

# seeds 0,1 → W≈188 (good); 5,7,22,91 → W≈1660 (bad local minima)
seeds = [0, 1, 5, 7, 22, 91]
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
axes = axes.flatten()

for ax, seed in zip(axes, seeds):
    km = KMeans(n_clusters=3, init="random", n_init=1, random_state=seed, max_iter=300)
    labels = km.fit_predict(X)
    W = km.inertia_
    for c in range(3):
        ax.scatter(X[labels == c, 0], X[labels == c, 1],
                   s=22, color=colors[c], alpha=0.85)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               s=120, color="white", edgecolors="black", zorder=5, linewidths=1.5, marker="D")
    ax.set_title(f"W = {W:.1f}", fontsize=12, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

fig.suptitle("6 lần khởi tạo ngẫu nhiên (K=3) — chọn W nhỏ nhất", fontsize=13, y=1.02)
fig.tight_layout(pad=0.3)
fig.savefig("../kmeans-initialization.svg", format="svg", bbox_inches="tight")
print("Saved kmeans-initialization.svg")
