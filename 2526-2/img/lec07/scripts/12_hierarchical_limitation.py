"""Example where a dendrogram is not the most natural summary.

Left: scatter coloured by gender with ellipses showing the 2-cluster split we want.
Right: dendrogram with K=2 cut line showing it splits by nationality, not gender.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

rng = np.random.RandomState(7)

# height, weight per group — spread groups so nationality causes merging issues
groups = {
    "Nam-Nhật":  ("Nam",  "#4a90d9", "s",  170, 65),
    "Nam-Pháp":  ("Nam",  "#4a90d9", "^",  176, 75),
    "Nam-Mỹ":    ("Nam",  "#4a90d9", "o",  180, 82),
    "Nữ-Nhật":  ("Nữ",   "#e8732a", "s",  155, 48),
    "Nữ-Pháp":  ("Nữ",   "#e8732a", "^",  162, 57),
    "Nữ-Mỹ":    ("Nữ",   "#e8732a", "o",  165, 62),
}

n_per = 6
X, leaf_labels, gender_list, colors_list, markers_list = [], [], [], [], []
for name, (gender, col, marker, h, w) in groups.items():
    pts = rng.randn(n_per, 2) * np.array([2.0, 2.5]) + np.array([h, w])
    X.append(pts)
    country = name.split("-")[1]
    leaf_labels.extend([f"{country}" for _ in range(n_per)])
    gender_list.extend([gender] * n_per)
    colors_list.extend([col] * n_per)
    markers_list.extend([marker] * n_per)

X = np.vstack(X)
n_obs = len(X)

Z = linkage(X, method="complete")

# Color dendrogram links by gender propagation
GRAY = "#aaaaaa"
node_color = {i: colors_list[i] for i in range(n_obs)}
for i, row in enumerate(Z):
    left, right = int(row[0]), int(row[1])
    nid = n_obs + i
    cl = node_color.get(left, GRAY)
    cr = node_color.get(right, GRAY)
    node_color[nid] = cl if cl == cr else GRAY

def link_color_func(k):
    return node_color.get(k, GRAY)

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

# ── Left: scatter with gender ellipses ──
ax1 = axes[0]

for i in range(n_obs):
    ax1.scatter(X[i, 0], X[i, 1], c=colors_list[i], marker=markers_list[i],
                s=60, edgecolors="white", linewidths=0.5, zorder=3)

# Ellipses for "intended" gender clusters
for gender, col in [("Nam", "#4a90d9"), ("Nữ", "#e8732a")]:
    mask = np.array([g == gender for g in gender_list])
    pts = X[mask]
    cx, cy = pts.mean(axis=0)
    sx, sy = pts.std(axis=0)
    ell = Ellipse((cx, cy), width=sx * 5, height=sy * 5, angle=0,
                  facecolor="none", edgecolor=col, linewidth=2.0,
                  linestyle="--", alpha=0.7)
    ax1.add_patch(ell)
    ax1.text(cx, cy + sy * 2.8, gender, fontsize=11, fontweight="bold",
             color=col, ha="center", va="bottom")

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Mỹ"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=8, label="Nhật"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=8, label="Pháp"),
]
ax1.legend(handles=legend_elements, fontsize=8.5, loc="lower right", framealpha=0.85)
ax1.set_xlabel("Chiều cao (cm)", fontsize=10)
ax1.set_ylabel("Cân nặng (kg)", fontsize=10)
ax1.set_title("Cách chia ta muốn: 2 cụm theo giới tính", fontsize=11, fontweight="bold")
ax1.tick_params(labelsize=8)

# ── Right: dendrogram with K=2 cut ──
ax2 = axes[1]
dn = dendrogram(Z, ax=ax2, labels=leaf_labels,
                link_color_func=link_color_func,
                leaf_rotation=90, leaf_font_size=7)

# Color x-tick labels by gender
for ticklabel, leaf_idx in zip(ax2.get_xticklabels(), dn["leaves"]):
    ticklabel.set_color(colors_list[leaf_idx])
    ticklabel.set_fontweight("bold")

# K=2 cut line
# Find the height that gives 2 clusters
sorted_dists = sorted(Z[:, 2], reverse=True)
cut_height = (sorted_dists[0] + sorted_dists[1]) / 2
ax2.axhline(cut_height, color="#c0392b", linewidth=1.5, linestyle="--", alpha=0.8)
ax2.text(ax2.get_xlim()[1] * 0.98, cut_height + 1, "cắt K=2",
         fontsize=9, color="#c0392b", ha="right", fontweight="bold")

# Annotate what K=2 produces
clusters_k2 = fcluster(Z, t=2, criterion="maxclust")
# Check if the split is by gender or nationality
from collections import Counter
for cid in [1, 2]:
    mask = clusters_k2 == cid
    genders_in = [gender_list[i] for i in range(n_obs) if mask[i]]
    cnt = Counter(genders_in)
    countries_in = [leaf_labels[i] for i in range(n_obs) if mask[i]]
    cnt_c = Counter(countries_in)

ax2.set_ylabel("Khoảng cách", fontsize=10)
ax2.set_title("Dendrogram gộp theo quốc tịch,\nkhông theo giới tính!", fontsize=11,
              fontweight="bold", color="#c0392b")
ax2.tick_params(axis='y', labelsize=8)

fig.tight_layout(pad=1.2)
fig.savefig("../hierarchical-limitation.svg", format="svg", bbox_inches="tight")
print("Saved hierarchical-limitation.svg")
