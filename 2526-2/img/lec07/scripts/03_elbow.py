"""Elbow method: within-cluster variance vs K."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)

inertias = []
ks = range(1, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

inertias = np.array(inertias)
inertias_norm = inertias / inertias[0]

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(ks, inertias_norm, "o-", color="#4a90d9", linewidth=2, markersize=7)
ax.axvline(3, color="#e8732a", linestyle="--", linewidth=1.2, label="Điểm khuỷu (K=3)")
ax.set_xlabel("Số cụm K", fontsize=11)
ax.set_ylabel("Phương sai trong cụm (chuẩn hóa)", fontsize=10)
ax.set_title("Elbow Method — Chọn K tối ưu", fontsize=11)
ax.set_xticks(list(ks))
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("../elbow-plot.svg", format="svg", bbox_inches="tight")
print("Saved elbow-plot.svg")
