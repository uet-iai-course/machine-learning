"""Draw two 2D region partitions: arbitrary vs guillotine cuts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# ---- Left: arbitrary regions ----
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$x_1$", fontsize=10)
ax.set_ylabel(r"$x_2$", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

# Irregular regions
rects = [
    (0.0, 0.0, 0.4, 0.5),   # R_J (bottom-left)
    (0.0, 0.5, 0.3, 0.5),   # R_2 (top-left)
    (0.3, 0.5, 0.3, 0.5),   # part of top
    (0.4, 0.0, 0.6, 0.35),  # bottom-right
    (0.6, 0.35, 0.4, 0.65), # R_1 (top-right)
]
labels_left = [
    (0.2, 0.25, r"$R_J$"),
    (0.15, 0.75, r"$R_2$"),
    (0.45, 0.75, ""),
    (0.7, 0.17, ""),
    (0.8, 0.7, r"$R_1$"),
]
for (x, y, w, h) in rects:
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="#333", linewidth=1.2))
for x, y, label in labels_left:
    if label:
        ax.text(x, y, label, fontsize=10, ha="center", va="center")

# ---- Right: guillotine cuts (5 regions) ----
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$x_1$", fontsize=10)
ax.set_ylabel(r"$x_2$", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

# Guillotine partition: t1=0.4, t3=0.7 on x1; t2=0.35, t4=0.6 on x2
# Split 1: x1 = 0.4
ax.plot([0.4, 0.4], [0, 1], color="#333", linewidth=1.2)
# Split 2: x2 = 0.35 (left of 0.4)
ax.plot([0, 0.4], [0.35, 0.35], color="#333", linewidth=1.2)
# Split 3: x1 = 0.7 (right of 0.4)
ax.plot([0.7, 0.7], [0, 1], color="#333", linewidth=1.2)
# Split 4: x2 = 0.6 (between 0.4 and 0.7)
ax.plot([0.4, 0.7], [0.6, 0.6], color="#333", linewidth=1.2)

# Labels
ax.text(0.2, 0.67, r"$R_2$", fontsize=10, ha="center", va="center")
ax.text(0.2, 0.17, r"$R_1$", fontsize=10, ha="center", va="center")
ax.text(0.55, 0.3, r"$R_3$", fontsize=10, ha="center", va="center")
ax.text(0.55, 0.8, r"$R_5$", fontsize=10, ha="center", va="center")
ax.text(0.85, 0.5, r"$R_4$", fontsize=10, ha="center", va="center")

# Tick labels for thresholds
ax.set_xticks([0.4, 0.7])
ax.set_xticklabels([r"$t_1$", r"$t_3$"], fontsize=9)
ax.set_yticks([0.35, 0.6])
ax.set_yticklabels([r"$t_2$", r"$t_4$"], fontsize=9)

fig.tight_layout(pad=0.8)
fig.savefig("../regions-partition.svg", format="svg", bbox_inches="tight")
print("Saved regions-partition.svg")
