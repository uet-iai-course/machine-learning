"""Slide 20 — Only points on/inside margin are SVs; distant points don't matter."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import scatter_two_class, C_HYP, C_MARGIN, C_ARROW

rng = np.random.RandomState(4)
n = 24
X1 = rng.randn(n, 2) * 0.9 + [1.4, 1.2]
X2 = rng.randn(n, 2) * 0.9 + [-1.2, -1.0]
X = np.vstack([X1, X2])
y = np.concatenate([np.ones(n), -np.ones(n)])

clf = SVC(kernel="linear", C=0.8).fit(X, y)
w, b = clf.coef_[0], clf.intercept_[0]
xs = np.linspace(-3.6, 3.6, 80)

fig, ax = plt.subplots(figsize=(5.8, 4.5))
ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                color=C_MARGIN, alpha=0.18)
ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=2)
for off in (1, -1):
    ax.plot(xs, -(w[0] * xs + b - off) / w[1], color=C_MARGIN, linestyle="--", linewidth=1.2)

# Color SVs (support vectors) and non-SVs differently
sv_mask = np.zeros(len(X), dtype=bool)
sv_mask[clf.support_] = True
# Non-SV: smaller, faded
ax.scatter(X[~sv_mask & (y > 0), 0], X[~sv_mask & (y > 0), 1], c="#2c6ea3",
           s=35, alpha=0.3, edgecolors="white", linewidths=0.5)
ax.scatter(X[~sv_mask & (y < 0), 0], X[~sv_mask & (y < 0), 1], c="#c0392b",
           s=35, alpha=0.3, edgecolors="white", linewidths=0.5)
# SV: large, vibrant, circled
ax.scatter(X[sv_mask & (y > 0), 0], X[sv_mask & (y > 0), 1], c="#2c6ea3",
           s=80, alpha=0.95, edgecolors="white", linewidths=0.5)
ax.scatter(X[sv_mask & (y < 0), 0], X[sv_mask & (y < 0), 1], c="#c0392b",
           s=80, alpha=0.95, edgecolors="white", linewidths=0.5)
ax.scatter(X[sv_mask, 0], X[sv_mask, 1], s=220, facecolors="none",
           edgecolors=C_ARROW, linewidths=2)

ax.text(2.2, 2.8, "SV đậm,\nnon-SV nhạt", fontsize=9.5, color="#444",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.4, 3.4)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../sv-property.svg", format="svg", bbox_inches="tight")
print("Saved sv-property.svg")
