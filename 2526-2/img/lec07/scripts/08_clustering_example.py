"""Clustering example: two clusters + outlier, color-coded."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rng = np.random.RandomState(3)
c1 = rng.randn(30, 2) * np.array([0.45, 0.5]) + np.array([-2.8, 0.3])
c2_raw = rng.randn(80, 2) * np.array([0.65, 0.55]) + np.array([1.8, 0.2])
# Keep only points inside the ellipse (semi-axes 1.8, 1.4)
mask = ((c2_raw[:, 0] - 1.8)**2 / 1.8**2 + (c2_raw[:, 1] - 0.2)**2 / 1.4**2) <= 1.0
c2 = c2_raw[mask][:45]
outlier = np.array([[-0.4, -2.1]])

fig, ax = plt.subplots(figsize=(7, 4.5))

# Cluster regions (filled ellipses)
ell1_bg = mpatches.Ellipse((-2.8, 0.3), 2.6, 2.6,
                            facecolor="#4a90d9", alpha=0.12, edgecolor="none")
ell2_bg = mpatches.Ellipse((1.8, 0.2), 3.6, 2.8,
                            facecolor="#e8732a", alpha=0.12, edgecolor="none")
ax.add_patch(ell1_bg)
ax.add_patch(ell2_bg)

# Cluster borders
ell1 = mpatches.Ellipse((-2.8, 0.3), 2.6, 2.6,
                         fill=False, edgecolor="#4a90d9", linewidth=2)
ell2 = mpatches.Ellipse((1.8, 0.2), 3.6, 2.8,
                         fill=False, edgecolor="#e8732a", linewidth=2)
ell3 = mpatches.Ellipse((-0.4, -2.1), 0.7, 0.7,
                         fill=False, edgecolor="#888", linewidth=1.5, linestyle="--")
ax.add_patch(ell1)
ax.add_patch(ell2)
ax.add_patch(ell3)

# Points
ax.scatter(c1[:, 0], c1[:, 1], s=30, color="#4a90d9", zorder=3)
ax.scatter(c2[:, 0], c2[:, 1], s=30, color="#e8732a", zorder=3)
ax.scatter(outlier[:, 0], outlier[:, 1], s=50, color="#888", zorder=3, marker="x", linewidths=2)

# Cluster labels above ellipses (outside)
# ell1 top: y = 0.3 + 1.3 = 1.6 → label at y=1.95
# ell2 top: y = 0.2 + 1.4 = 1.6 → label at y=1.95
ax.text(-2.8, 1.95, "Cụm 1", fontsize=14, fontweight="bold",
        color="#4a90d9", ha="center", va="bottom", zorder=4)
ax.text(1.8, 1.95, "Cụm 2", fontsize=14, fontweight="bold",
        color="#e8732a", ha="center", va="bottom", zorder=4)

# Arrow touches right edge of ell1 (-1.5, 0.25) → left edge of ell2 (0.0, 0.25)
ax.annotate("", xy=(0.0, 0.25), xytext=(-1.5, 0.25),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.8))
ax.text(-0.75, 0.62, "Giữa các cụm:\nkhoảng cách lớn",
        fontsize=11, color="#555", ha="center")

# Annotation: outlier
ax.annotate("Điểm nhiễu?",
            xy=(-0.4, -2.1), xytext=(1.5, -2.5),
            fontsize=12, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))

ax.set_xlim(-4.8, 4.2)
ax.set_ylim(-3.2, 2.6)
ax.axis("off")
fig.tight_layout(pad=0.3)
fig.savefig("../clustering-example.svg", format="svg", bbox_inches="tight")
print("Saved clustering-example.svg")
