"""Lloyd's algorithm: 6 key steps visualized with centroid movement arrows."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=60, centers=[[-2, 0], [2, 0], [0, 3]],
                  cluster_std=0.85, random_state=7)
colors = ["#4a90d9", "#e8732a", "#5aaa44"]

def assign(X, centers):
    dists = np.array([[np.linalg.norm(x - c) for c in centers] for x in X])
    return dists.argmin(axis=1)

def compute_centroid(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

# Bad initial centroids — far from true centers to show big movement
centers0 = np.array([[-3.5, 3.0], [0.0, -1.5], [3.5, 3.0]])

steps = []

# Step 1: init
steps.append(("Bước 1: Khởi tạo", None, centers0.copy(), None))

# Step 2: first assignment
labels1 = assign(X, centers0)
steps.append(("Bước 2: Gán cụm", labels1, centers0.copy(), None))

# Step 3: update centroids
centers1 = compute_centroid(X, labels1, 3)
steps.append(("Bước 3: Cập nhật centroid", labels1, centers1.copy(), centers0.copy()))

# Step 4: re-assign
labels2 = assign(X, centers1)
steps.append(("Bước 4: Gán cụm", labels2, centers1.copy(), None))

# Step 5: update centroids again
centers2 = compute_centroid(X, labels2, 3)
steps.append(("Bước 5: Cập nhật centroid", labels2, centers2.copy(), centers1.copy()))

# Step 6: converged
labels3 = assign(X, centers2)
steps.append(("Bước 6: Hội tụ ✓", labels3, centers2.copy(), None))

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
axes = axes.flatten()

for ax, (title, labels, cents, old_cents) in zip(axes, steps):
    # Points
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=30, color="#bbb", alpha=0.7)
    else:
        for c in range(3):
            mask = labels == c
            ax.scatter(X[mask, 0], X[mask, 1],
                       s=30, color=colors[c], alpha=0.75)

    # Old centroids (ghost) + arrows
    if old_cents is not None:
        # Ghost old positions
        ax.scatter(old_cents[:, 0], old_cents[:, 1], s=120, color="#ddd",
                   edgecolors="#999", zorder=4, linewidths=1.2, marker="D",
                   alpha=0.7)
        # Arrows from old to new
        for i in range(3):
            dx = cents[i, 0] - old_cents[i, 0]
            dy = cents[i, 1] - old_cents[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.05:
                ax.annotate("", xy=cents[i], xytext=old_cents[i],
                            arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                            lw=2.2, mutation_scale=16),
                            zorder=6)

    # Current centroids
    ax.scatter(cents[:, 0], cents[:, 1], s=160, color="white",
               edgecolors="black", zorder=7, linewidths=2.0, marker="D")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout(pad=0.6)
fig.savefig("../kmeans-lloyd-steps.svg", format="svg", bbox_inches="tight")
print("Saved kmeans-lloyd-steps.svg")
