"""Slide 3 — Motivation: many separating lines, which is best?"""
import numpy as np
import matplotlib.pyplot as plt
from _common import make_separable_2d, scatter_two_class, C_HYP

X, y = make_separable_2d(n_per=15, sep=2.8, noise=0.55, seed=7)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
scatter_two_class(ax, X, y, s=60)

# Three different separating lines (all separate this dataset)
xs = np.linspace(-3.5, 3.5, 50)
lines = [
    (-1.0,  0.0, "-"),   # y = -x
    (-0.4,  0.3, "--"),  # y = -0.4x + 0.3
    (-2.0, -0.5, ":"),   # y = -2x - 0.5
]
for slope, intercept, ls in lines:
    ax.plot(xs, slope * xs + intercept, color=C_HYP, linestyle=ls, linewidth=1.8, alpha=0.85)

ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../motivation-many-lines.svg", format="svg", bbox_inches="tight")
print("Saved motivation-many-lines.svg")
