"""Slide 6 — Hyperplane with normal vector and the two half-spaces."""
import numpy as np
import matplotlib.pyplot as plt
from _common import C_HYP, C_POS, C_NEG, C_ARROW

fig, ax = plt.subplots(figsize=(5.6, 4.5))

# Hyperplane: beta0 + 1*x1 + 1.5*x2 = 0  ==>  x2 = -(x1 + beta0)/1.5
beta0, b1, b2 = 0.0, 1.0, 1.5
xs = np.linspace(-3, 3, 50)
line_y = -(beta0 + b1 * xs) / b2

# Half-space shading
ax.fill_between(xs, line_y, 3, color=C_POS, alpha=0.10)
ax.fill_between(xs, line_y, -3, color=C_NEG, alpha=0.10)
ax.plot(xs, line_y, color=C_HYP, linewidth=2)

# Normal vector (unit) rooted at origin of hyperplane (0, 0 gives beta0/|beta|=0 on plane)
norm = np.sqrt(b1 ** 2 + b2 ** 2)
u = np.array([b1, b2]) / norm
# Place the arrow starting at a point on the hyperplane, point to +1 half-space
p0 = np.array([0.0, 0.0])
ax.annotate("", xy=(p0 + 1.5 * u), xytext=p0,
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2.2))
ax.text((p0 + 1.7 * u)[0] + 0.05, (p0 + 1.7 * u)[1] + 0.05,
        r"$(\beta_1, \beta_2)$", color=C_ARROW, fontsize=12, fontweight="bold")

# Half-space labels
ax.text(1.6, 2.2, r"$\beta_0 + \beta \cdot x > 0$", color=C_POS, fontsize=11, fontweight="bold")
ax.text(-2.8, -2.2, r"$\beta_0 + \beta \cdot x < 0$", color=C_NEG, fontsize=11, fontweight="bold")
ax.text(1.9, -1.4, r"$\beta_0 + \beta \cdot x = 0$", color=C_HYP, fontsize=10,
        rotation=np.degrees(np.arctan2(-b1, b2)))

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../hyperplane-normal.svg", format="svg", bbox_inches="tight")
print("Saved hyperplane-normal.svg")
