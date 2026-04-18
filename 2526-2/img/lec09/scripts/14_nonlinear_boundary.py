"""Slide 22 — Nonlinear boundary: concentric classes; linear SVC fails."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import scatter_two_class, C_HYP, C_MARGIN

rng = np.random.RandomState(1)
n = 60
# Inner class
r1 = rng.rand(n) * 1.2
theta1 = rng.rand(n) * 2 * np.pi
X_pos = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
# Outer class
r2 = 2.0 + rng.rand(n) * 0.8
theta2 = rng.rand(n) * 2 * np.pi
X_neg = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n), -np.ones(n)])

clf = SVC(kernel="linear", C=1.0).fit(X, y)
w, b = clf.coef_[0], clf.intercept_[0]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
scatter_two_class(ax, X, y, s=42)

xs = np.linspace(-3.5, 3.5, 80)
# Linear SVC's best-try line
if abs(w[1]) > 1e-6:
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=2)
else:
    ax.axvline(-b / w[0], color=C_HYP, linewidth=2)

ax.text(0, 3.1, "Linear SVC không tách được", fontsize=11, color="#c0392b",
        ha="center", fontweight="bold")

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.7)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../nonlinear-boundary.svg", format="svg", bbox_inches="tight")
print("Saved nonlinear-boundary.svg")
