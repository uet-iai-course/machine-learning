"""K-means clustering demo: K=2, 3, 4 on synthetic data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

rng = np.random.RandomState(42)
X, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.1, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
colors = ["#4a90d9", "#e8732a", "#5aaa44", "#c0392b"]

for ax, k in zip(axes, [2, 3, 4]):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    for c in range(k):
        ax.scatter(X[labels == c, 0], X[labels == c, 1],
                   s=22, color=colors[c], alpha=0.85)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               s=120, color="white", edgecolors="black", zorder=5, linewidths=1.5, marker="*")
    ax.set_title(f"K = {k}", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

fig.tight_layout(pad=0.5)
fig.savefig("../kmeans-k234.svg", format="svg", bbox_inches="tight")
print("Saved kmeans-k234.svg")
