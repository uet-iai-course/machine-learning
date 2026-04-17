"""Slide 12 — Moving a non-SV vs moving a SV: only SV affects hyperplane."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import make_separable_2d, scatter_two_class, C_HYP, C_MARGIN, C_ARROW


def fit_and_plot(ax, X, y, title):
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-3.5, 3.5, 50)
    ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                    color=C_MARGIN, alpha=0.18)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=2)
    for offset, ls in [(1, "--"), (-1, "--")]:
        ax.plot(xs, -(w[0] * xs + b - offset) / w[1], color=C_MARGIN, linestyle=ls, linewidth=1.2)
    scatter_two_class(ax, X, y, s=42)
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="none", edgecolors=C_ARROW, linewidths=2)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


X, y = make_separable_2d(n_per=14, sep=2.8, noise=0.55, seed=2)
clf0 = SVC(kernel="linear", C=1e6).fit(X, y)
sv_mask = np.zeros(len(X), dtype=bool)
sv_mask[clf0.support_] = True

# Pick a non-SV positive point to move far away (hyperplane shouldn't change)
non_sv_pos_idx = np.where((y > 0) & ~sv_mask)[0][0]
# Pick a SV to nudge (hyperplane should change)
sv_pos_idx = np.where((y > 0) & sv_mask)[0][0]

X_nonsv_move = X.copy()
X_nonsv_move[non_sv_pos_idx] += np.array([1.5, 1.5])

X_sv_move = X.copy()
X_sv_move[sv_pos_idx] += np.array([-0.6, -0.6])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
fit_and_plot(axes[0], X, y, "Gốc")
fit_and_plot(axes[1], X_nonsv_move, y, "Di chuyển một điểm không phải SV")
fit_and_plot(axes[2], X_sv_move, y, "Di chuyển một SV")

# Arrow on moved point
axes[1].annotate("", xy=X_nonsv_move[non_sv_pos_idx], xytext=X[non_sv_pos_idx],
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.8))
axes[2].annotate("", xy=X_sv_move[sv_pos_idx], xytext=X[sv_pos_idx],
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.8))

fig.tight_layout(pad=0.8)
fig.savefig("../mmc-example.svg", format="svg", bbox_inches="tight")
print("Saved mmc-example.svg")
