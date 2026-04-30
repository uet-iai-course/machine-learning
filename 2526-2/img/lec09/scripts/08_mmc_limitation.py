"""Slide 13 — Limitations: outlier shrinks margin; non-separable dataset."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import make_separable_2d, scatter_two_class, C_HYP, C_MARGIN, C_ARROW


def plot_hard(ax, X, y):
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-3.5, 3.5, 50)
    ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                    color=C_MARGIN, alpha=0.18)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=2)
    for offset in (1, -1):
        ax.plot(xs, -(w[0] * xs + b - offset) / w[1], color=C_MARGIN, linestyle="--", linewidth=1.2)


# ── Left: separable but with outlier that shrinks margin ──
X, y = make_separable_2d(n_per=15, sep=3.0, noise=0.5, seed=8)
X_out = np.vstack([X, [[0.2, 0.5]]])
y_out = np.concatenate([y, [1]])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

plot_hard(axes[0], X_out, y_out)
scatter_two_class(axes[0], X_out, y_out, s=45)
# Highlight the outlier
axes[0].scatter([0.2], [0.5], s=220, facecolors="none", edgecolors=C_ARROW, linewidths=2.2)
axes[0].annotate("điểm ngoại lai", xy=(0.2, 0.5), xytext=(1.4, 2.4),
                 fontsize=11, color=C_ARROW, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4))
axes[0].set_xlim(-3.6, 3.6)
axes[0].set_ylim(-3.2, 3.2)
axes[0].set_xlabel(r"$X_1$")
axes[0].set_ylabel(r"$X_2$")
axes[0].set_aspect("equal")
axes[0].set_title("Điểm ngoại lai làm lề co lại", fontsize=11)

# ── Right: non-separable dataset (overlapping classes) ──
rng = np.random.RandomState(9)
n = 30
X1 = rng.randn(n, 2) * 1.0 + [1.2, 1.0]
X2 = rng.randn(n, 2) * 1.0 + [-0.8, -0.8]
X_ns = np.vstack([X1, X2])
y_ns = np.concatenate([np.ones(n), -np.ones(n)])

scatter_two_class(axes[1], X_ns, y_ns, s=45)
axes[1].set_xlim(-3.6, 3.6)
axes[1].set_ylim(-3.2, 3.2)
axes[1].set_xlabel(r"$X_1$")
axes[1].set_ylabel(r"$X_2$")
axes[1].set_aspect("equal")
axes[1].set_title("Không đường thẳng nào tách đúng toàn bộ", fontsize=11)

fig.tight_layout(pad=0.8)
fig.savefig("../mmc-limitation.svg", format="svg", bbox_inches="tight")
print("Saved mmc-limitation.svg")
