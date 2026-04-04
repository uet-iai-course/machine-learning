"""Circular / radial dendrogram visualization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

np.random.seed(42)
n = 7  # points per cluster
c1 = np.random.randn(n, 2) * 0.5 + [-4, 0]
c2 = np.random.randn(n, 2) * 0.5 + [0, 3.5]
c3 = np.random.randn(n, 2) * 0.5 + [4, 0]
X = np.vstack([c1, c2, c3])
n_obs = len(X)
labels = [f"N{i+1}" for i in range(n_obs)]

Z = linkage(X, method="complete")
cut_h = (Z[-2, 2] + Z[-3, 2]) / 2
cluster_ids = fcluster(Z, t=3, criterion="maxclust")

CMAP = {1: "#4a90d9", 2: "#e8732a", 3: "#5aaa44"}
GRAY = "#aaaaaa"

# Build node_color dict (leaf + merge nodes)
node_color = {}
for i in range(n_obs):
    node_color[i] = CMAP[cluster_ids[i]]
for i, row in enumerate(Z):
    left, right, dist = int(row[0]), int(row[1]), row[2]
    nid = n_obs + i
    cl = node_color.get(left, GRAY)
    cr = node_color.get(right, GRAY)
    node_color[nid] = cl if (dist < cut_h and cl == cr) else GRAY

# Get dendrogram layout (no draw)
dn = dendrogram(Z, labels=labels, no_plot=True)
icoord = np.array(dn['icoord'])   # shape (n_merges, 4)
dcoord = np.array(dn['dcoord'])   # shape (n_merges, 4)
ivl = dn['ivl']                   # leaf labels in display order

n_leaves = len(ivl)
max_x = n_leaves * 10.0
max_y = max(Z[:, 2])

R_LEAF = 0.88   # leaves at this radius
R_ROOT = 0.08   # root at center

def to_polar(x, y):
    """Map dendrogram (x,y) → (angle, radius). Leaves at outer edge."""
    angle = (x / max_x) * 2 * np.pi - np.pi / 2   # start from top
    r = R_LEAF - (y / max_y) * (R_LEAF - R_ROOT)
    return angle, r

# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(projection='polar'))
ax.set_ylim(0, 1.2)
ax.axis('off')

# Draw each merge link (U-shape → 2 radial stems + 1 arc)
for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
    nid = n_obs + i
    color = node_color.get(nid, GRAY)
    lw = 1.6

    a_l = (xs[0] / max_x) * 2 * np.pi - np.pi / 2
    a_r = (xs[3] / max_x) * 2 * np.pi - np.pi / 2
    r_bot_l = R_LEAF - (ys[0] / max_y) * (R_LEAF - R_ROOT)
    r_bot_r = R_LEAF - (ys[3] / max_y) * (R_LEAF - R_ROOT)
    r_top   = R_LEAF - (ys[1] / max_y) * (R_LEAF - R_ROOT)

    # Left radial stem
    ax.plot([a_l, a_l], [r_bot_l, r_top], color=color, lw=lw,
            solid_capstyle='round', zorder=2)
    # Right radial stem
    ax.plot([a_r, a_r], [r_bot_r, r_top], color=color, lw=lw,
            solid_capstyle='round', zorder=2)
    # Arc at merge height (ensure short-way direction)
    a_start, a_end = sorted([a_l, a_r])
    arcs = np.linspace(a_start, a_end, 80)
    ax.plot(arcs, np.full_like(arcs, r_top), color=color, lw=lw, zorder=2)

# Draw leaf dots and labels
for pos, lbl in enumerate(ivl):
    orig_idx = labels.index(lbl)
    color = CMAP[cluster_ids[orig_idx]]
    angle = ((pos * 10 + 5) / max_x) * 2 * np.pi - np.pi / 2

    # Dot
    ax.scatter(angle, R_LEAF, s=100, color=color, zorder=5,
               edgecolors='white', linewidths=1.2)

    # Label
    r_lbl = R_LEAF + 0.14
    # Rotate label to read outward
    rot_deg = np.degrees(angle)
    if rot_deg > 90:
        rot_deg -= 180
    elif rot_deg < -90:
        rot_deg += 180
    ax.text(angle, r_lbl, lbl,
            ha='center', va='center',
            fontsize=8.5, color=color, fontweight='bold',
            rotation=rot_deg, rotation_mode='anchor')

fig.tight_layout()
fig.savefig("../dendrogram-radial.svg", format="svg", bbox_inches="tight")
print("Saved dendrogram-radial.svg")
