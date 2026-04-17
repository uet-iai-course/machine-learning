"""Slide 23 — Lifting 2D → 3D via z = x1^2 + x2^2 makes the concentric data separable."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import C_POS, C_NEG, C_HYP

rng = np.random.RandomState(1)
n = 45
r1 = rng.rand(n) * 1.2
theta1 = rng.rand(n) * 2 * np.pi
X_pos = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
r2 = 2.0 + rng.rand(n) * 0.7
theta2 = rng.rand(n) * 2 * np.pi
X_neg = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)

fig = plt.figure(figsize=(10, 4.2))

# Left: original 2D
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_pos[:, 0], X_pos[:, 1], c=C_POS, s=44, edgecolors="white", linewidths=0.5)
ax1.scatter(X_neg[:, 0], X_neg[:, 1], c=C_NEG, s=44, edgecolors="white", linewidths=0.5)
ax1.set_xlim(-3.3, 3.3)
ax1.set_ylim(-3.3, 3.3)
ax1.set_xlabel(r"$X_1$")
ax1.set_ylabel(r"$X_2$")
ax1.set_title("Không gian gốc — không tách tuyến tính", fontsize=11)
ax1.set_aspect("equal")

# Right: 3D with z = x1^2 + x2^2
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
z_pos = X_pos[:, 0] ** 2 + X_pos[:, 1] ** 2
z_neg = X_neg[:, 0] ** 2 + X_neg[:, 1] ** 2
ax2.scatter(X_pos[:, 0], X_pos[:, 1], z_pos, c=C_POS, s=36, edgecolors="white", linewidths=0.4)
ax2.scatter(X_neg[:, 0], X_neg[:, 1], z_neg, c=C_NEG, s=36, edgecolors="white", linewidths=0.4)

# Separating plane z = 2.2
x1g = np.linspace(-3, 3, 10)
x2g = np.linspace(-3, 3, 10)
X1g, X2g = np.meshgrid(x1g, x2g)
Zg = np.full_like(X1g, 2.2)
ax2.plot_surface(X1g, X2g, Zg, alpha=0.18, color=C_HYP, edgecolor="none")

ax2.set_xlabel(r"$X_1$")
ax2.set_ylabel(r"$X_2$")
ax2.set_zlabel(r"$X_1^2 + X_2^2$")
ax2.set_title("Không gian mở rộng — tách được bằng mặt phẳng", fontsize=11)
ax2.view_init(elev=18, azim=-55)

fig.tight_layout(pad=0.6)
fig.savefig("../feature-map-2d-to-3d.svg", format="svg", bbox_inches="tight")
print("Saved feature-map-2d-to-3d.svg")
