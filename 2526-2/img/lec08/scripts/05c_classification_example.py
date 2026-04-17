"""Simulate 2-class grid to illustrate misclassification error insensitivity."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

fig, ax = plt.subplots(figsize=(5, 4))

# 200 points in [0,10] x [0,6], 2 classes
n = 200
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 6, n)

# Class assignment: left half mostly class 0, right half mostly class 1
# but with 40/200 = 20% minority overall
classes = np.zeros(n, dtype=int)
# Right side (x1 > 5): 60 class 1, 40 class 0
right = x1 >= 5
classes[right] = 1
# Flip some to create 40/60 split in each half
right_idx = np.where(right)[0]
left_idx = np.where(~right)[0]
# In left half: 60 class 0, 40 class 1
np.random.shuffle(left_idx)
classes[left_idx[:40]] = 1
# In right half: 40 class 0, 60 class 1
np.random.shuffle(right_idx)
classes[right_idx[:40]] = 0

colors = ["#2ecc71", "#e74c3c"]
for c in [0, 1]:
    mask = classes == c
    ax.scatter(x1[mask], x2[mask], s=15, color=colors[c], alpha=0.7, edgecolors="none")

# Draw the best split
ax.axvline(5, color="#333", linewidth=1.5, linestyle="--")

# Labels
n_left = np.sum(x1 < 5)
n_right = np.sum(x1 >= 5)
c1_left = np.sum((x1 < 5) & (classes == 1))
c1_right = np.sum((x1 >= 5) & (classes == 1))

ax.text(2.5, 5.5, f"{c1_left}", fontsize=11, ha="center", va="center",
        color="#e74c3c", fontweight="bold")
ax.text(2.5, 4.8, f"{n_left - c1_left}", fontsize=11, ha="center", va="center",
        color="#2ecc71", fontweight="bold")
ax.text(7.5, 5.5, f"{c1_right}", fontsize=11, ha="center", va="center",
        color="#e74c3c", fontweight="bold")
ax.text(7.5, 4.8, f"{n_right - c1_right}", fontsize=11, ha="center", va="center",
        color="#2ecc71", fontweight="bold")

ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

fig.tight_layout(pad=0.5)
fig.savefig("../classification-example.svg", format="svg", bbox_inches="tight")
print("Saved classification-example.svg")
