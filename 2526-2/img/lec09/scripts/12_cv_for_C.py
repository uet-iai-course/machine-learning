"""Slide 19 — CV error vs log10(C)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from _common import C_ARROW

rng = np.random.RandomState(2)
n = 60
X1 = rng.randn(n, 2) * 0.95 + [0.9, 0.9]
X2 = rng.randn(n, 2) * 0.95 + [-0.9, -0.9]
X = np.vstack([X1, X2])
y = np.concatenate([np.ones(n), -np.ones(n)])

Cs = np.logspace(-3, 3, 25)
errors = []
for C in Cs:
    scores = cross_val_score(SVC(kernel="linear", C=C), X, y, cv=5)
    errors.append(1 - scores.mean())

errors = np.array(errors)
best_idx = int(np.argmin(errors))

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(Cs, errors, color="#2c6ea3", linewidth=2)
ax.scatter([Cs[best_idx]], [errors[best_idx]], color=C_ARROW, s=90,
           zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate(f"C tối ưu", xy=(Cs[best_idx], errors[best_idx]),
            xytext=(Cs[best_idx] * 5, errors[best_idx] + 0.04),
            fontsize=10, color=C_ARROW, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.3))

ax.set_xscale("log")
ax.set_xlabel(r"$C$ (log scale)")
ax.set_ylabel("CV error")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("../cv-for-C.svg", format="svg", bbox_inches="tight")
print("Saved cv-for-C.svg")
