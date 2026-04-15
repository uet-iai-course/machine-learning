"""Example where a dendrogram is not the most natural summary.

Left: scatter plot coloured by gender -> clear 2-cluster structure.
Right: dendrogram with leaf labels showing nationality and colours showing gender
       -> the tree tends to merge by nationality first.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

rng = np.random.RandomState(7)

# height, weight means per group
groups = {
    "Nam-Mỹ":    ("Nam",  "#4a90d9", "o",  178, 80),
    "Nam-Nhật":  ("Nam",  "#4a90d9", "s",  170, 65),
    "Nam-Pháp":  ("Nam",  "#4a90d9", "^",  175, 74),
    "Nữ-Mỹ":    ("Nữ",   "#e8732a", "o",  163, 60),
    "Nữ-Nhật":  ("Nữ",   "#e8732a", "s",  156, 49),
    "Nữ-Pháp":  ("Nữ",   "#e8732a", "^",  161, 56),
}

n_per = 5
X, leaf_labels, colors_list, markers_list = [], [], [], []
for name, (gender, col, marker, h, w) in groups.items():
    pts = rng.randn(n_per, 2) * np.array([2.5, 3]) + np.array([h, w])
    X.append(pts)
    country = name.split("-")[1]
    leaf_labels.extend([country] * n_per)
    colors_list.extend([col] * n_per)
    markers_list.extend([marker] * n_per)

X = np.vstack(X)

Z = linkage(X, method="complete")

# ── Colors for dendrogram leaves ──────────────────────────────────────
n_obs = len(X)
GRAY = "#aaaaaa"

# Each leaf node color = gender color
node_color = {i: colors_list[i] for i in range(n_obs)}

# Propagate upward: if both children have same color → same color, else gray
for i, row in enumerate(Z):
    left, right = int(row[0]), int(row[1])
    nid = n_obs + i
    cl = node_color.get(left, GRAY)
    cr = node_color.get(right, GRAY)
    node_color[nid] = cl if cl == cr else GRAY

def link_color_func(k):
    return node_color.get(k, GRAY)

# ── Figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ── Left: scatter coloured by gender ────────────────────────────────
ax1 = axes[0]
nat_labels = {"o": "Mỹ", "s": "Nhật", "^": "Pháp"}

plotted_gender = set()
plotted_nat = set()
for i in range(n_obs):
    gender = "Nam" if colors_list[i] == "#4a90d9" else "Nữ"
    marker = markers_list[i]
    col = colors_list[i]

    gender_lbl = gender if gender not in plotted_gender else None
    nat_lbl = nat_labels[marker] if marker not in plotted_nat else None
    plotted_gender.add(gender)
    plotted_nat.add(marker)

    ax1.scatter(X[i, 0], X[i, 1], c=col, marker=marker,
                s=70, edgecolors="white", linewidths=0.6, zorder=3)

# Manual legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a90d9", markersize=9, label="Nam"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e8732a", markersize=9, label="Nữ"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Mỹ"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=8, label="Nhật"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=8, label="Pháp"),
]
ax1.legend(handles=legend_elements, fontsize=8.5, loc="upper left",
           ncol=2, framealpha=0.85)
ax1.set_xlabel("Chiều cao (cm)", fontsize=9)
ax1.set_ylabel("Cân nặng (kg)", fontsize=9)
ax1.set_title("Một cách chia tự nhiên: 2 cụm theo giới tính", fontsize=10)
ax1.tick_params(labelsize=8)

# ── Right: dendrogram, leaves coloured by gender ─────────────────────
ax2 = axes[1]
dn = dendrogram(Z, ax=ax2,
                labels=leaf_labels,
                link_color_func=link_color_func,
                leaf_rotation=90, leaf_font_size=7)

# Colour x-tick labels by gender
for ticklabel, leaf_idx in zip(ax2.get_xticklabels(), dn["leaves"]):
    ticklabel.set_color(colors_list[leaf_idx])
    ticklabel.set_fontweight("bold")

ax2.set_ylabel("Khoảng cách", fontsize=9)
ax2.set_title("Dendrogram lại gộp theo quốc tịch trước", fontsize=10)
ax2.tick_params(axis='y', labelsize=8)

fig.tight_layout(pad=1.2)
fig.savefig("../hierarchical-limitation.svg", format="svg", bbox_inches="tight")
print("Saved hierarchical-limitation.svg")
