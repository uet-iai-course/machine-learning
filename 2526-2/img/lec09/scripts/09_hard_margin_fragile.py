"""Slide 15 — Adding one outlier dramatically changes the hard-margin hyperplane."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import make_separable_2d, scatter_two_class, C_HYP, C_MARGIN, C_ARROW


def fit_hard(X, y):
    return SVC(kernel="linear", C=1e6).fit(X, y)


def plot_svc(ax, X, y, clf, title):
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-3.6, 3.6, 60)
    ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                    color=C_MARGIN, alpha=0.18)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=2)
    for offset in (1, -1):
        ax.plot(xs, -(w[0] * xs + b - offset) / w[1], color=C_MARGIN, linestyle="--", linewidth=1.2)
    scatter_two_class(ax, X, y, s=45)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.4, 3.4)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


X, y = make_separable_2d(n_per=16, sep=3.2, noise=0.55, seed=4)

# Outlier: a positive-class point deep in the negative region but still barely separable
X_out = np.vstack([X, [[0.8, -0.3]]])
y_out = np.concatenate([y, [1]])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
plot_svc(axes[0], X, y, fit_hard(X, y), "Trước khi thêm điểm ngoại lai")
plot_svc(axes[1], X_out, y_out, fit_hard(X_out, y_out), "Sau khi thêm 1 điểm ngoại lai")

# Highlight the outlier
axes[1].scatter([0.8], [-0.3], s=230, facecolors="none", edgecolors=C_ARROW, linewidths=2.2)
axes[1].annotate("điểm ngoại lai", xy=(0.8, -0.3), xytext=(1.5, -2.7),
                 fontsize=11, color=C_ARROW, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4))

fig.tight_layout(pad=0.8)
fig.savefig("../hard-margin-fragile.svg", format="svg", bbox_inches="tight")
print("Saved hard-margin-fragile.svg")
