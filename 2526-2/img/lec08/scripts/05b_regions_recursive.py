"""Draw a recursive partition with 5 regions R1-R5 (guillotine cuts)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

# Cuts: x1=0.4 first, then x2=0.35 (left), x1=0.7 (right), x2=0.6 (mid)
lw = 1.5
ax.plot([0.4, 0.4], [0, 1], color="#333", linewidth=lw)      # cut 1
ax.plot([0, 0.4], [0.35, 0.35], color="#333", linewidth=lw)   # cut 2
ax.plot([0.7, 0.7], [0, 1], color="#333", linewidth=lw)       # cut 3
ax.plot([0.4, 0.7], [0.6, 0.6], color="#333", linewidth=lw)   # cut 4

# Labels
ax.text(0.2, 0.17, r"$R_1$", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(0.2, 0.67, r"$R_2$", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(0.55, 0.3, r"$R_3$", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(0.55, 0.8, r"$R_5$", fontsize=12, ha="center", va="center", fontweight="bold")
ax.text(0.85, 0.5, r"$R_4$", fontsize=12, ha="center", va="center", fontweight="bold")

# Threshold labels
ax.set_xticks([0.4, 0.7])
ax.set_xticklabels([r"$t_1$", r"$t_3$"], fontsize=10)
ax.set_yticks([0.35, 0.6])
ax.set_yticklabels([r"$t_2$", r"$t_4$"], fontsize=10)
ax.set_xlabel(r"$X_1$", fontsize=11)
ax.set_ylabel(r"$X_2$", fontsize=11)

fig.tight_layout(pad=0.5)
fig.savefig("../regions-recursive.svg", format="svg", bbox_inches="tight")
print("Saved regions-recursive.svg")
