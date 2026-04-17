"""Slide 7 — Classify by sign of f(x); |f(x)| = confidence."""
import numpy as np
import matplotlib.pyplot as plt
from _common import make_separable_2d, C_HYP, C_POS, C_NEG

X, y = make_separable_2d(n_per=14, sep=2.6, noise=0.5, seed=3)

fig, ax = plt.subplots(figsize=(5.6, 4.5))

# Choose a good separating line f(x) = 0.7 x1 + x2 - 0.2
b1, b2, b0 = 0.7, 1.0, -0.2
xs = np.linspace(-3.2, 3.2, 100)
line_y = -(b0 + b1 * xs) / b2
ax.plot(xs, line_y, color=C_HYP, linewidth=2)

# Color points by sign; size by |f(x)|
f_vals = b0 + b1 * X[:, 0] + b2 * X[:, 1]
sizes = 30 + 80 * np.abs(f_vals) / np.max(np.abs(f_vals))
colors = [C_POS if f > 0 else C_NEG for f in f_vals]
ax.scatter(X[:, 0], X[:, 1], c=colors, s=sizes, alpha=0.85,
           edgecolors="white", linewidths=0.6)

# Annotate one far point and one near point
idx_far = np.argmax(np.abs(f_vals))
idx_near = np.argmin(np.abs(f_vals))
for idx, label in [(idx_far, "xa \u2192 tự tin"), (idx_near, "gần \u2192 kém tin")]:
    ax.annotate(label, xy=X[idx], xytext=(X[idx, 0] + 0.6, X[idx, 1] + 0.8),
                fontsize=9.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))

ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.0, 3.2)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../classify-by-sign.svg", format="svg", bbox_inches="tight")
print("Saved classify-by-sign.svg")
