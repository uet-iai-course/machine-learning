"""Slide 16 — Soft-margin: a few points allowed inside / on wrong side of margin."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import scatter_two_class, C_HYP, C_MARGIN, C_ARROW

rng = np.random.RandomState(3)
n = 22
X1 = rng.randn(n, 2) * 0.8 + [1.3, 1.2]
X2 = rng.randn(n, 2) * 0.8 + [-1.1, -1.0]
# Add two violators
X1 = np.vstack([X1, [[-0.3, 0.2], [0.2, -0.6]]])
X2 = np.vstack([X2, [[0.8, -0.5]]])
X = np.vstack([X1, X2])
y = np.concatenate([np.ones(len(X1)), -np.ones(len(X2))])

clf = SVC(kernel="linear", C=1.0).fit(X, y)
w, b = clf.coef_[0], clf.intercept_[0]

xs = np.linspace(-3.5, 3.5, 100)
line_y = -(w[0] * xs + b) / w[1]
line_y_up = -(w[0] * xs + b - 1) / w[1]
line_y_dn = -(w[0] * xs + b + 1) / w[1]

fig, ax = plt.subplots(figsize=(5.6, 4.5))
ax.fill_between(xs, line_y_dn, line_y_up, color=C_MARGIN, alpha=0.18)
ax.plot(xs, line_y, color=C_HYP, linewidth=2)
ax.plot(xs, line_y_up, color=C_MARGIN, linewidth=1.3, linestyle="--")
ax.plot(xs, line_y_dn, color=C_MARGIN, linewidth=1.3, linestyle="--")

scatter_two_class(ax, X, y, s=45)

# Mark the violators (points with y*f < 1)
f_vals = w[0] * X[:, 0] + w[1] * X[:, 1] + b
viol = y * f_vals < 1
ax.scatter(X[viol, 0], X[viol, 1], s=220, facecolors="none",
           edgecolors=C_ARROW, linewidths=2.2)

ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.4, 3.4)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../soft-margin-concept.svg", format="svg", bbox_inches="tight")
print("Saved soft-margin-concept.svg")
