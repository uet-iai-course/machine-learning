"""Simulated dataset with 3 classes for hierarchical clustering demo."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=45, centers=3, cluster_std=0.8, random_state=5)
colors = ["#5aaa44", "#e8a020", "#c0392b"]

fig, ax = plt.subplots(figsize=(5, 4))
for c in range(3):
    ax.scatter(X[y == c, 0], X[y == c, 1], s=40, color=colors[c], alpha=0.9, label=f"Nhóm {c+1}")
ax.set_xlabel("$X_1$", fontsize=11)
ax.set_ylabel("$X_2$", fontsize=11)
ax.set_title("Dữ liệu mô phỏng (nhãn ẩn với thuật toán)", fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig("../hierarchical-data.svg", format="svg", bbox_inches="tight")
print("Saved hierarchical-data.svg")
