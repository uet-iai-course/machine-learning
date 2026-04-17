"""Generate Figure 1, 2, 3 for Homework 08 (Problem 2).

Figures reproduce the partition / tree / decision-boundary diagrams
from Saarland WS 2023/24 Homework Sheet #6 (Problem 2).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Predictor space partition (student must draw the tree)
# ═══════════════════════════════════════════════════════════════════
#
# 8 regions on [0, 3.5] × [-1.5, 1.5] with values {0, 7, 3, 11, 31, -12, -8, 11}
# corresponding to the recursive partition:
#
#   X1 < 1?
#     Yes: X2 < 0? [Yes: 11, No: 0]
#     No:  X2 < 1?
#            Yes: X1 < 2?
#                   Yes: 31
#                   No:  X2 < -1?
#                          Yes: X1 < 3? [Yes: -8, No: 11]
#                          No:  -12
#            No:  X1 < 3? [Yes: 7, No: 3]
fig1, ax1 = plt.subplots(figsize=(5.8, 5.0))

regions = [
    # (x, y, w, h, value)
    (0.0,  0.0, 1.0, 1.5, "0"),     # X1<1, X2>=0
    (0.0, -1.5, 1.0, 1.5, "11"),    # X1<1, X2<0
    (1.0,  1.0, 2.0, 0.5, "7"),     # 1<=X1<3, X2>=1
    (3.0,  1.0, 0.5, 0.5, "3"),     # X1>=3, X2>=1
    (1.0, -1.5, 1.0, 2.5, "31"),    # 1<=X1<2, X2<1
    (2.0, -1.0, 1.5, 2.0, "-12"),   # X1>=2, -1<=X2<1
    (2.0, -1.5, 1.0, 0.5, "-8"),    # 2<=X1<3, X2<-1
    (3.0, -1.5, 0.5, 0.5, "11"),    # X1>=3, X2<-1
]
for x, y, w, h, label in regions:
    ax1.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                     edgecolor="#333", linewidth=1.4))
    ax1.text(x + w / 2, y + h / 2, label, fontsize=12,
             ha="center", va="center", fontweight="bold", color="#c0392b")

ax1.set_xlim(-0.08, 3.58)
ax1.set_ylim(-1.6, 1.6)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_yticks([-1, 0, 1])
ax1.set_xlabel(r"$X_1$", fontsize=12)
ax1.set_ylabel(r"$X_2$", fontsize=12)
ax1.tick_params(labelsize=10)
ax1.set_aspect("equal")

fig1.tight_layout()
fig1.savefig("fig1.svg", format="svg", bbox_inches="tight")
print("Saved fig1.svg")


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Decision tree (student must draw the partition)
# ═══════════════════════════════════════════════════════════════════
#
# Tree reproduced verbatim from PDF (8 leaves: 2, 4, -8, 0, -2, 42, 9, -3):
#
#   X2 < 3?
#     Yes: X1 < -1?
#            Yes: X2 < 1? [Yes: 2, No: 4]
#            No:  X2 < 2?
#                   Yes: X1 < 1?
#                          Yes: -8
#                          No:  X2 < 1? [Yes: 0, No: -2]
#                   No:  42
#     No:  X1 < 0? [Yes: 9, No: -3]
fig2, ax2 = plt.subplots(figsize=(9.5, 5.2))
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 6.5)
ax2.axis("off")

q_kw = dict(boxstyle="round,pad=0.26", facecolor="#fff8e1", edgecolor="#d4a017", linewidth=1.4)
leaf_kw = dict(boxstyle="round,pad=0.26", facecolor="#e8f4fd", edgecolor="#4a90d9", linewidth=1.4)
edge_kw = dict(color="#333", linewidth=1.2)
yn_kw = dict(fontsize=9, ha="center", va="center", color="#666")

# Leaf x-coordinates (in-order traversal of 8 leaves)
# 2(L1), 4(L2), -8(L3), 0(L4), -2(L5), 42(L6), 9(L7), -3(L8)
lx = {"2": 1.0, "4": 2.5, "-8": 4.5, "0": 6.0, "-2": 7.5, "42": 9.5, "9": 12.5, "-3": 14.5}

# Internal node positions computed bottom-up
n_X2lt1_L = ((lx["2"] + lx["4"]) / 2, 3.2)          # left X2<1 above 2,4
n_X2lt1_R = ((lx["0"] + lx["-2"]) / 2, 1.8)         # right X2<1 above 0,-2
n_X1lt1   = ((lx["-8"] + n_X2lt1_R[0]) / 2, 3.0)    # X1<1 above -8 and right X2<1
n_X2lt2   = ((n_X1lt1[0] + lx["42"]) / 2, 4.2)      # X2<2 above X1<1 and 42
n_X1ltm1  = ((n_X2lt1_L[0] + n_X2lt2[0]) / 2, 5.1)  # X1<-1 above left X2<1 and X2<2
n_X1lt0   = ((lx["9"] + lx["-3"]) / 2, 5.1)         # X1<0 above 9,-3
n_root    = ((n_X1ltm1[0] + n_X1lt0[0]) / 2, 6.0)   # X2<3 root

# Leaf y-coordinates (at their actual depth)
leaf_y = {"2": 2.1, "4": 2.1, "-8": 1.8, "0": 0.6, "-2": 0.6, "42": 3.0,
          "9": 4.1, "-3": 4.1}

def edge(a, b):
    (xa, ya), (xb, yb) = a, b
    ax2.plot([xa, xb], [ya - 0.22, yb + 0.22], **edge_kw)

def mid(a, b, dy=0, dx=0):
    return ((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy)

# Positions for leaves
leaf_pos = {k: (lx[k], leaf_y[k]) for k in lx}

# Edges
edges_list = [
    (n_root, n_X1ltm1), (n_root, n_X1lt0),
    (n_X1ltm1, n_X2lt1_L), (n_X1ltm1, n_X2lt2),
    (n_X2lt1_L, leaf_pos["2"]), (n_X2lt1_L, leaf_pos["4"]),
    (n_X2lt2, n_X1lt1), (n_X2lt2, leaf_pos["42"]),
    (n_X1lt1, leaf_pos["-8"]), (n_X1lt1, n_X2lt1_R),
    (n_X2lt1_R, leaf_pos["0"]), (n_X2lt1_R, leaf_pos["-2"]),
    (n_X1lt0, leaf_pos["9"]), (n_X1lt0, leaf_pos["-3"]),
]
for a, b in edges_list:
    edge(a, b)

# Internal nodes
for pos, label in [(n_root, r"$X_2 < 3$?"), (n_X1ltm1, r"$X_1 < -1$?"),
                    (n_X1lt0, r"$X_1 < 0$?"), (n_X2lt1_L, r"$X_2 < 1$?"),
                    (n_X2lt2, r"$X_2 < 2$?"), (n_X1lt1, r"$X_1 < 1$?"),
                    (n_X2lt1_R, r"$X_2 < 1$?")]:
    ax2.text(*pos, label, fontsize=10, ha="center", va="center",
             fontweight="bold", bbox=q_kw)

# Leaves
for k, pos in leaf_pos.items():
    ax2.text(*pos, k, fontsize=11, ha="center", va="center",
             fontweight="bold", color="#c0392b", bbox=leaf_kw)

# Yes/No labels
def yn(parent, child, label, dx=0, dy=0):
    p = mid(parent, child, dy=dy, dx=dx)
    ax2.text(*p, label, **yn_kw)

yn(n_root, n_X1ltm1, "Có", dx=-0.3)
yn(n_root, n_X1lt0, "Không", dx=0.4)
yn(n_X1ltm1, n_X2lt1_L, "Có", dx=-0.25)
yn(n_X1ltm1, n_X2lt2, "Không", dx=0.35)
yn(n_X2lt1_L, leaf_pos["2"], "Có", dx=-0.2)
yn(n_X2lt1_L, leaf_pos["4"], "Không", dx=0.3)
yn(n_X2lt2, n_X1lt1, "Có", dx=-0.3)
yn(n_X2lt2, leaf_pos["42"], "Không", dx=0.35)
yn(n_X1lt1, leaf_pos["-8"], "Có", dx=-0.2)
yn(n_X1lt1, n_X2lt1_R, "Không", dx=0.3)
yn(n_X2lt1_R, leaf_pos["0"], "Có", dx=-0.2)
yn(n_X2lt1_R, leaf_pos["-2"], "Không", dx=0.3)
yn(n_X1lt0, leaf_pos["9"], "Có", dx=-0.2)
yn(n_X1lt0, leaf_pos["-3"], "Không", dx=0.3)

fig2.tight_layout(pad=0.3)
fig2.savefig("fig2.svg", format="svg", bbox_inches="tight")
print("Saved fig2.svg")


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — 4 decision boundaries (student classifies each)
# ═══════════════════════════════════════════════════════════════════
#
# (a) Off-center diagonal: blue lower-left triangle, green upper-right polygon
# (b) Off-center vertical split (left-leaning): thin blue left strip, green right
# (c) Two parallel diagonals (negative slope): green band between, blue outside
# (d) Blue left strip + right area divided horizontally (blue/green stripes)
BLUE = "#4a7bd9"
GREEN = "#7ac974"

fig3, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# (a) Off-center diagonal: line from (0, 0.85) to (1, 0)
a = axes[0]
a.add_patch(patches.Polygon([(0, 0), (1, 0), (0, 0.85)], facecolor=BLUE,
                             edgecolor="#222", linewidth=1))
a.add_patch(patches.Polygon([(0, 0.85), (1, 0), (1, 1), (0, 1)], facecolor=GREEN,
                             edgecolor="#222", linewidth=1))
a.set_title("(a)", fontsize=12, fontweight="bold", pad=6)

# (b) Off-center vertical split at x ≈ 0.22 (left-leaning)
b = axes[1]
b.add_patch(patches.Rectangle((0, 0), 0.22, 1, facecolor=BLUE,
                               edgecolor="#222", linewidth=1))
b.add_patch(patches.Rectangle((0.22, 0), 0.78, 1, facecolor=GREEN,
                               edgecolor="#222", linewidth=1))
b.set_title("(b)", fontsize=12, fontweight="bold", pad=6)

# (c) Two parallel diagonals (negative slope), green band between
c = axes[2]
# Fill with blue first
c.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=BLUE,
                               edgecolor="#222", linewidth=1))
# Green band: upper line (0, 1.00) → (1, 0.45); lower line (0, 0.55) → (1, 0.00)
band = patches.Polygon([(0, 0.55), (1, 0.00), (1, 0.45), (0, 1.00)],
                        facecolor=GREEN, edgecolor="#222", linewidth=1)
c.add_patch(band)
c.set_title("(c)", fontsize=12, fontweight="bold", pad=6)

# (d) Blue left strip + right area divided horizontally
d = axes[3]
# Blue left strip
d.add_patch(patches.Rectangle((0, 0), 0.18, 1, facecolor=BLUE,
                               edgecolor="#222", linewidth=1))
# Right area horizontal stripes
stripes = [(0.00, 0.30, GREEN), (0.30, 0.40, BLUE),
           (0.40, 0.85, GREEN), (0.85, 1.00, BLUE)]
for y0, y1, color in stripes:
    d.add_patch(patches.Rectangle((0.18, y0), 0.82, y1 - y0, facecolor=color,
                                   edgecolor="#222", linewidth=1))
d.set_title("(d)", fontsize=12, fontweight="bold", pad=6)

fig3.tight_layout(pad=0.6)
fig3.savefig("fig3.svg", format="svg", bbox_inches="tight")
print("Saved fig3.svg")
