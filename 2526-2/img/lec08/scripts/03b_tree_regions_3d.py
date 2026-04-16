"""From Tree to Regions and Back: tree diagram + 2D partition + 3D step surface."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

# ── Thresholds ──
t1, t2, t3, t4 = 0.4, 0.35, 0.7, 0.6

# ── Regions ──
regions = [
    (0,  t1, 0,  t2, 2.0, r"$R_1$", "#a8d5a2"),
    (0,  t1, t2, 1,  4.0, r"$R_2$", "#7ec8e3"),
    (t1, t3, 0,  t4, 1.0, r"$R_3$", "#f7dc6f"),
    (t3, 1,  0,  1,  3.0, r"$R_4$", "#f1948a"),
    (t1, t3, t4, 1,  5.0, r"$R_5$", "#c39bd3"),
]

def region_value(x1, x2):
    if x1 < t1 and x2 < t2:   return 2.0
    elif x1 < t1 and x2 >= t2: return 4.0
    elif x1 < t3 and x2 < t4:  return 1.0
    elif x1 >= t3:              return 3.0
    else:                       return 5.0

fig = plt.figure(figsize=(14, 4.6))

# ══════════════ Panel 1: Tree diagram ══════════════
ax1 = fig.add_axes([0.01, 0.05, 0.28, 0.90])
ax1.set_xlim(0, 10); ax1.set_ylim(-0.2, 7); ax1.axis("off")

node_kw = dict(fontsize=10, ha="center", va="center", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9f9f9",
                         edgecolor="#444", linewidth=1.2))
leaf_colors = {
    r"$R_1$": "#a8d5a2", r"$R_2$": "#7ec8e3", r"$R_3$": "#f7dc6f",
    r"$R_4$": "#f1948a", r"$R_5$": "#c39bd3",
}
def leaf_kw(label):
    return dict(fontsize=10, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=leaf_colors[label],
                          edgecolor="#444", linewidth=1.2))

for _, (x, y, lab) in {"root":(5,6.2,r"$X_1 \leq t_1$"), "L":(2,4.2,r"$X_2 \leq t_2$"),
                        "R":(8,4.2,r"$X_1 \leq t_3$"), "RL":(6.5,2.2,r"$X_2 \leq t_4$")}.items():
    ax1.text(x, y, lab, **node_kw)
for _, (x, y, lab) in {"LL":(0.8,2.2,r"$R_1$"), "LR":(3.2,2.2,r"$R_2$"),
                        "RR":(9.5,2.2,r"$R_4$"), "RLL":(5.3,0.3,r"$R_3$"),
                        "RLR":(7.7,0.3,r"$R_5$")}.items():
    ax1.text(x, y, lab, **leaf_kw(lab))

for x1,y1,x2,y2 in [(5,5.8,2,4.65),(5,5.8,8,4.65),(2,3.8,0.8,2.65),(2,3.8,3.2,2.65),
                     (8,3.8,6.5,2.65),(8,3.8,9.5,2.65),(6.5,1.8,5.3,0.75),(6.5,1.8,7.7,0.75)]:
    ax1.plot([x1,x2],[y1,y2], color="#444", linewidth=1.3)

# ══════════════ Panel 2: 2D partition ══════════════
ax2 = fig.add_axes([0.315, 0.10, 0.27, 0.82])
ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_aspect("equal")

for x1lo,x1hi,x2lo,x2hi,_,lab,col in regions:
    ax2.add_patch(Rectangle((x1lo,x2lo), x1hi-x1lo, x2hi-x2lo,
                             facecolor=col, alpha=0.55, edgecolor="none"))
    ax2.text((x1lo+x1hi)/2, (x2lo+x2hi)/2, lab,
             fontsize=12, ha="center", va="center", fontweight="bold")

for seg in [([t1,t1],[0,1]),([t3,t3],[0,1]),([0,t1],[t2,t2]),([t1,t3],[t4,t4])]:
    ax2.plot(*seg, color="#333", linewidth=1.8)
ax2.set_xticks([t1,t3]); ax2.set_xticklabels([r"$t_1$",r"$t_3$"], fontsize=10)
ax2.set_yticks([t2,t4]); ax2.set_yticklabels([r"$t_2$",r"$t_4$"], fontsize=10)
ax2.set_xlabel(r"$X_1$", fontsize=11); ax2.set_ylabel(r"$X_2$", fontsize=11)
ax2.spines[:].set_linewidth(1.2)

# ══════════════ Panel 3: 3D step surface ══════════════
ax3 = fig.add_axes([0.63, 0.02, 0.37, 0.96], projection="3d")

# Build grid with tight pairs at each boundary → sharp vertical steps
eps = 1e-3
bx = [0, t1-eps, t1+eps, t3-eps, t3+eps, 1]
by = [0, t2-eps, t2+eps, t4-eps, t4+eps, 1]

# Add ~8 interior points per interval for smooth flat tops
def interp(vals, n=8):
    pts = []
    for a, b in zip(vals[:-1], vals[1:]):
        pts.extend(np.linspace(a, b, n, endpoint=False))
    pts.append(vals[-1])
    return np.array(sorted(set(pts)))

X1, X2 = np.meshgrid(interp(bx), interp(by))
Z = np.vectorize(region_value)(X1, X2)

# Build per-face colors: flat faces → region color, vertical faces → gray
from matplotlib.colors import to_rgba
val_to_color = {1.0: "#f7dc6f", 2.0: "#a8d5a2", 3.0: "#f1948a",
                4.0: "#7ec8e3", 5.0: "#c39bd3"}
wall_color = to_rgba("#888888", alpha=0.7)

nr, nc = Z.shape
facecolors = np.empty((nr-1, nc-1, 4))
for i in range(nr-1):
    for j in range(nc-1):
        corners = [Z[i,j], Z[i+1,j], Z[i,j+1], Z[i+1,j+1]]
        if max(corners) - min(corners) > 0.5:  # vertical wall
            facecolors[i, j] = wall_color
        else:
            facecolors[i, j] = to_rgba(val_to_color[corners[0]], alpha=0.92)

ax3.plot_surface(X1, X2, Z, facecolors=facecolors,
                 rstride=1, cstride=1,
                 edgecolor="#999", linewidth=0.05, shade=True)

# Region labels floating above top
for x1lo,x1hi,x2lo,x2hi,h,lab,_ in regions:
    ax3.text((x1lo+x1hi)/2, (x2lo+x2hi)/2, h+0.25, lab,
             fontsize=9, ha="center", va="center", fontweight="bold", color="#333")

ax3.set_xlabel(r"$X_1$", fontsize=10, labelpad=5)
ax3.set_ylabel(r"$X_2$", fontsize=10, labelpad=5)
ax3.set_zlabel(r"$\hat{y}$", fontsize=10, labelpad=2)
ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.set_zlim(0,5.8)
ax3.set_xticks([0,0.5,1]); ax3.set_yticks([0,0.5,1]); ax3.set_zticks([1,2,3,4,5])
ax3.tick_params(labelsize=7, pad=1)
ax3.view_init(elev=32, azim=-55)

ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#ccc")
ax3.yaxis.pane.set_edgecolor("#ccc")
ax3.zaxis.pane.set_edgecolor("#ccc")
ax3.grid(True, alpha=0.25)

fig.savefig("../tree-regions-3d.svg", format="svg", bbox_inches="tight")
fig.savefig("../tree-regions-3d.png", format="png", dpi=180, bbox_inches="tight")
print("Saved tree-regions-3d.svg + .png")
