"""Slide 5 — Hyperplane: 2D line and 3D plane side by side."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import C_HYP, C_POS, C_NEG

fig = plt.figure(figsize=(9, 4))

# ── Left: 2D line ─────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1)
xs = np.linspace(-3, 3, 50)
ax1.plot(xs, -0.6 * xs + 0.4, color=C_HYP, linewidth=2)
# Shade half-spaces
ax1.fill_between(xs, -0.6 * xs + 0.4, 3, color=C_POS, alpha=0.12)
ax1.fill_between(xs, -0.6 * xs + 0.4, -3, color=C_NEG, alpha=0.12)
ax1.text(1.8, 2.2, r"$f(x) > 0$", color=C_POS, fontsize=12, fontweight="bold")
ax1.text(-2.5, -2.2, r"$f(x) < 0$", color=C_NEG, fontsize=12, fontweight="bold")
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel(r"$X_1$")
ax1.set_ylabel(r"$X_2$")
ax1.set_title(r"$p=2$: đường thẳng", fontsize=11)
ax1.set_aspect("equal")

# ── Right: 3D plane ───────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
x1 = np.linspace(-2, 2, 15)
x2 = np.linspace(-2, 2, 15)
X1, X2 = np.meshgrid(x1, x2)
X3 = -0.4 * X1 - 0.5 * X2 + 0.2
ax2.plot_surface(X1, X2, X3, alpha=0.4, color=C_HYP, edgecolor="none")
ax2.plot_wireframe(X1, X2, X3, color=C_HYP, linewidth=0.4, alpha=0.5,
                    rcount=6, ccount=6)
ax2.set_xlabel(r"$X_1$")
ax2.set_ylabel(r"$X_2$")
ax2.set_zlabel(r"$X_3$")
ax2.set_title(r"$p=3$: mặt phẳng", fontsize=11)
ax2.view_init(elev=22, azim=-55)

fig.tight_layout(pad=0.8)
fig.savefig("../hyperplane-2d-3d.svg", format="svg", bbox_inches="tight")
print("Saved hyperplane-2d-3d.svg")
