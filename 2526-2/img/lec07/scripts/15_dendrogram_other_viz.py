"""Other Visualizations of Dendrograms (ISLR 12.4.2 Fig 12.12).

Shows two dendrogram styles on the same data:
  Left  — Standard dendrogram, single color
  Right — Colored by cluster (K=3 cut), tick labels use SAME colors as branches
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

np.random.seed(42)
n = 15
c1 = np.random.randn(n, 2) * 0.5 + [-4, 0]
c2 = np.random.randn(n, 2) * 0.5 + [0, 3]
c3 = np.random.randn(n, 2) * 0.5 + [4, 0]
X = np.vstack([c1, c2, c3])
labels = [str(i + 1) for i in range(len(X))]
n_obs = len(X)

Z = linkage(X, method="complete")

# Cut height between 3rd and 2nd-to-last merge → K=3
cut_h = (Z[-2, 2] + Z[-3, 2]) / 2

# Cluster assignment per leaf
cluster_ids = fcluster(Z, t=3, criterion="maxclust")   # values 1,2,3

# Three colors for the three clusters
CMAP = {1: "#4a90d9", 2: "#e8732a", 3: "#5aaa44"}
GRAY = "#aaaaaa"

# ── Build link_color_func using same CMAP ────────────────────────────
# Each node (leaf 0..n_obs-1, merge n_obs..2*n_obs-2) gets a color.
node_color = {}
for i in range(n_obs):
    node_color[i] = CMAP[cluster_ids[i]]

for i, row in enumerate(Z):
    left, right, dist = int(row[0]), int(row[1]), row[2]
    node_id = n_obs + i
    cl = node_color.get(left, GRAY)
    cr = node_color.get(right, GRAY)
    if dist >= cut_h:
        node_color[node_id] = GRAY
    elif cl == cr:
        node_color[node_id] = cl
    else:
        node_color[node_id] = GRAY

def link_color_func(k):
    return node_color.get(k, GRAY)

# ── Figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ── Left: standard, single color ────────────────────────────────────
ax1 = axes[0]
dendrogram(Z, ax=ax1, labels=labels,
           color_threshold=0,
           above_threshold_color="#4a90d9",
           link_color_func=lambda k: "#4a90d9",
           leaf_font_size=7.5)
ax1.set_title("")
ax1.set_ylabel("Khoảng cách bất đồng", fontsize=9)
ax1.tick_params(axis='x', labelsize=7.5)
ax1.tick_params(axis='y', labelsize=8)

# ── Right: colored by cluster via link_color_func ────────────────────
ax2 = axes[1]
dn = dendrogram(Z, ax=ax2, labels=labels,
                link_color_func=link_color_func,
                leaf_font_size=7.5)
ax2.axhline(y=cut_h, color="#c0392b", lw=1.5, ls="--", label="Cắt K=3")
ax2.set_title("")
ax2.set_ylabel("Khoảng cách bất đồng", fontsize=9)
ax2.tick_params(axis='x', labelsize=7.5)
ax2.tick_params(axis='y', labelsize=8)
ax2.legend(fontsize=8.5, loc="upper left")

# Color tick labels using the SAME CMAP as the branches
for ticklabel in ax2.get_xticklabels():
    txt = ticklabel.get_text()
    orig_idx = labels.index(txt)
    ticklabel.set_color(CMAP[cluster_ids[orig_idx]])
    ticklabel.set_fontweight("bold")

ax2.text(0.5, -0.18,
         "Tất cả các lá đều ở tọa độ y = 0 (baseline chung)",
         fontsize=8, color="#555", ha="center", va="top",
         transform=ax2.transAxes, style="italic")

fig.tight_layout(pad=1.2)
fig.savefig("../dendrogram-other-viz.svg", format="svg", bbox_inches="tight")
print("Saved dendrogram-other-viz.svg")
