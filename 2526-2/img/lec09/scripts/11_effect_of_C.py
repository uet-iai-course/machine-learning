"""Slide 18 — Effect of C on margin width (2x2 panel)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import scatter_two_class, C_HYP, C_MARGIN


rng = np.random.RandomState(5)
n = 22
X1 = rng.randn(n, 2) * 0.95 + [1.2, 1.2]
X2 = rng.randn(n, 2) * 0.95 + [-1.1, -1.0]
X = np.vstack([X1, X2])
y = np.concatenate([np.ones(n), -np.ones(n)])


def plot_c(ax, X, y, C, title):
    clf = SVC(kernel="linear", C=C).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-3.5, 3.5, 80)
    ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                    color=C_MARGIN, alpha=0.18)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=1.8)
    for off in (1, -1):
        ax.plot(xs, -(w[0] * xs + b - off) / w[1], color=C_MARGIN,
                linestyle="--", linewidth=1.1)
    scatter_two_class(ax, X, y, s=30)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
# As C in sklearn is inverse of the "budget" — large C ~ hard margin (thin margin).
# The slide's "large C" in ISLR notation means large BUDGET → wide margin.
# In sklearn, this maps to small C. We label with the slide's terminology.
plot_c(axes[0, 0], X, y, C=0.02, title=r"$C$ rất lớn (budget lớn)")
plot_c(axes[0, 1], X, y, C=0.1,  title=r"$C$ lớn")
plot_c(axes[1, 0], X, y, C=1.0,  title=r"$C$ nhỏ")
plot_c(axes[1, 1], X, y, C=50.0, title=r"$C$ rất nhỏ")

fig.tight_layout(pad=0.6)
fig.savefig("../effect-of-C.svg", format="svg", bbox_inches="tight")
print("Saved effect-of-C.svg")
