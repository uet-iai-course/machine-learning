"""Dendrogram axis warning: x-axis proximity does NOT mean similarity.

Two leaves adjacent on the x-axis (boundary between clusters)
can merge at the TOP — i.e., be very dissimilar.
Conversely, two leaves far apart on x-axis can merge LOW (very similar).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(7)
# Two compact, well-separated clusters
cA = np.random.randn(5, 2) * 0.25 + np.array([-3.5, 0])
cB = np.random.randn(5, 2) * 0.25 + np.array([ 3.5, 0])
X = np.vstack([cA, cB])
labels = [f"A{i+1}" for i in range(5)] + [f"B{i+1}" for i in range(5)]

Z = linkage(X, method="complete")
dn_info = dendrogram(Z, labels=labels, no_plot=True)
leaf_order = dn_info["ivl"]
print("Leaf order:", leaf_order)

# Find the boundary pair: adjacent leaves from different clusters — merge at top
# Look for an adjacent (i, i+1) pair where one is A and the other is B
boundary_pair = None
for i in range(len(leaf_order) - 1):
    if (leaf_order[i].startswith("A") and leaf_order[i+1].startswith("B")) or \
       (leaf_order[i].startswith("B") and leaf_order[i+1].startswith("A")):
        boundary_pair = (i, i+1)
        break
assert boundary_pair is not None, "No cross-cluster adjacent pair found"
last_a_idx, first_b_idx = boundary_pair
lbl_near1 = leaf_order[last_a_idx]
lbl_near2 = leaf_order[first_b_idx]
print(f"Adjacent boundary pair: {lbl_near1} (pos {last_a_idx}) | {lbl_near2} (pos {first_b_idx})")

# Map label → original X index to get 2D coordinates
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
idx1 = label_to_idx[lbl_near1]
idx2 = label_to_idx[lbl_near2]
p1 = X[idx1]
p2 = X[idx2]

# ── Figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Left: scatter ──────────────────────────────────────────
ax = axes[0]
ax.scatter(cA[:, 0], cA[:, 1], s=60, color="#4a90d9", label="Cụm A", zorder=3)
ax.scatter(cB[:, 0], cB[:, 1], s=60, color="#e8732a", label="Cụm B", zorder=3)

# Highlight the adjacent-on-dendrogram pair with large markers
ax.scatter(*p1, s=180, color="#4a90d9", edgecolors="#c0392b", linewidths=2.5,
           zorder=5, marker="o")
ax.scatter(*p2, s=180, color="#e8732a", edgecolors="#c0392b", linewidths=2.5,
           zorder=5, marker="o")
ax.text(p1[0] - 0.15, p1[1] + 0.35, lbl_near1, fontsize=10, color="#c0392b",
        fontweight="bold", ha="center")
ax.text(p2[0] + 0.15, p2[1] + 0.35, lbl_near2, fontsize=10, color="#c0392b",
        fontweight="bold", ha="center")

# Double-headed arrow showing they are FAR apart
mid = (p1 + p2) / 2
ax.annotate("", xy=p2, xytext=p1,
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.0))
ax.text(mid[0], mid[1] - 0.45, "xa nhau!", fontsize=9.5, color="#c0392b",
        ha="center", fontweight="bold")

ax.legend(fontsize=9, loc="upper right")
ax.set_xlabel("$X_1$", fontsize=10)
ax.set_ylabel("$X_2$", fontsize=10)
ax.set_title("Dữ liệu gốc", fontsize=11)
ax.tick_params(labelsize=8)

# ── Right: dendrogram ──────────────────────────────────────
ax2 = axes[1]
dn = dendrogram(Z, ax=ax2, labels=labels,
                color_threshold=0,
                above_threshold_color="#888",
                link_color_func=lambda k: "#888",
                leaf_font_size=8)

ax2.set_ylabel("Khoảng cách bất đồng", fontsize=9)
ax2.set_title(f"Dendrogram — {lbl_near1} và {lbl_near2} kề nhau trên trục X\nnhưng ghép cụm ở ĐỈNH (rất khác nhau)!", fontsize=9.5)

# Color the boundary pair labels red
for ticklabel in ax2.get_xticklabels():
    if ticklabel.get_text() in (lbl_near1, lbl_near2):
        ticklabel.set_color("#c0392b")
        ticklabel.set_fontweight("bold")
        ticklabel.set_fontsize(9)

# Leaf x-positions (dendrogram uses 5, 15, 25, ... = 5 + 10*i)
leaf_pos = {lbl: 5 + 10 * i for i, lbl in enumerate(leaf_order)}
xn1 = leaf_pos[lbl_near1]
xn2 = leaf_pos[lbl_near2]

# Arrow at bottom showing x-axis proximity
yb = -0.18
ax2.annotate("", xy=(xn2, yb), xytext=(xn1, yb),
             xycoords=("data", "axes fraction"),
             textcoords=("data", "axes fraction"),
             arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.0),
             annotation_clip=False)
ax2.text((xn1 + xn2) / 2, yb - 0.09, "kề nhau trên trục X!",
         fontsize=9, color="#c0392b", ha="center", va="top",
         transform=ax2.get_xaxis_transform())

# Key message box
ax2.text(0.5, 0.97,
         "Khoảng cách trên trục X\nKHÔNG phản ánh mức độ tương đồng!",
         fontsize=8.5, color="#7d3c98", ha="center", va="top",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", fc="#f5eef8", ec="#7d3c98", lw=1.2))

fig.tight_layout(pad=1.0)
fig.savefig("../dendrogram-axis.svg", format="svg", bbox_inches="tight")
print("Saved dendrogram-axis.svg")
print("Leaf order:", leaf_order)
