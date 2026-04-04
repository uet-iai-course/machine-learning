"""Lloyd's algorithm: 4 key steps visualized."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng = np.random.RandomState(7)
X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.9, random_state=7)
colors = ["#4a90d9", "#e8732a", "#5aaa44"]

def assign(X, centers):
    dists = np.array([[np.linalg.norm(x - c) for c in centers] for x in X])
    return dists.argmin(axis=1)

def centroid(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

# Initial random centers
np.random.seed(3)
centers = X[np.random.choice(len(X), 3, replace=False)]

steps = []
# Step 1: random init (unassigned)
steps.append(("Bước 1: Khởi tạo ngẫu nhiên", None, centers.copy()))
# Step 2: first assignment
labels1 = assign(X, centers)
steps.append(("Bước 2: Gán cụm (lần 1)", labels1, centers.copy()))
# Step 3: update centroids
c2 = centroid(X, labels1, 3)
steps.append(("Bước 3: Cập nhật centroid", labels1, c2.copy()))
# Step 4: final assignment
labels2 = assign(X, c2)
steps.append(("Bước 4: Gán cụm (lần 2)", labels2, c2.copy()))

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.flatten()
for ax, (title, labels, cents) in zip(axes, steps):
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=35, color="#aaa", alpha=0.7)
    else:
        for c in range(3):
            ax.scatter(X[labels == c, 0], X[labels == c, 1],
                       s=35, color=colors[c], alpha=0.8)
    ax.scatter(cents[:, 0], cents[:, 1], s=200, color="white",
               edgecolors="black", zorder=5, linewidths=1.8, marker="D")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout(pad=0.4)
fig.savefig("../kmeans-lloyd-steps.svg", format="svg", bbox_inches="tight")
print("Saved kmeans-lloyd-steps.svg")
