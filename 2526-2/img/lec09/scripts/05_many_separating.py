"""Slide 8 — Several separating hyperplanes on the same dataset."""
import numpy as np
import matplotlib.pyplot as plt
from _common import make_separable_2d, scatter_two_class, C_HYP

X, y = make_separable_2d(n_per=18, sep=2.6, noise=0.55, seed=11)

fig, ax = plt.subplots(figsize=(5.4, 4.5))
scatter_two_class(ax, X, y, s=55)

xs = np.linspace(-3.5, 3.5, 50)
# Four different separating lines
lines = [
    (-1.1, -0.1),
    (-0.6,  0.3),
    (-1.8, -0.4),
    (-0.3,  0.5),
]
styles = ["-", "--", ":", "-."]
for (slope, intercept), ls in zip(lines, styles):
    ax.plot(xs, slope * xs + intercept, color=C_HYP, linestyle=ls, linewidth=1.7, alpha=0.85)

ax.text(1.5, 2.7, r"$\ldots$", fontsize=16, color=C_HYP)
ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../many-separating.svg", format="svg", bbox_inches="tight")
print("Saved many-separating.svg")
