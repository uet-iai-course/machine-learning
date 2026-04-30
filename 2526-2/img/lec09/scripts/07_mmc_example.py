"""Slide 12 — Moving a non-SV vs moving a SV: only SV affects hyperplane.

Panel 3 overlays the original hyperplane (faded dashed) so the viewer can read
the shift directly. Panel 2 deliberately shows just one hyperplane: nothing
changes, so a second line would be visual noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import make_separable_2d, scatter_two_class, C_HYP, C_MARGIN, C_ARROW


def hyperplane_lines(clf, xs):
    w, b = clf.coef_[0], clf.intercept_[0]
    y_mid = -(w[0] * xs + b) / w[1]
    y_up = -(w[0] * xs + b - 1) / w[1]
    y_lo = -(w[0] * xs + b + 1) / w[1]
    return y_mid, y_up, y_lo


def fit_and_plot(ax, X, y, title, ref_clf=None):
    """ref_clf: if given, draw it as a faded dashed reference line."""
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    xs = np.linspace(-3.5, 3.5, 80)
    y_mid, y_up, y_lo = hyperplane_lines(clf, xs)
    ax.fill_between(xs, y_lo, y_up, color=C_MARGIN, alpha=0.18)
    if ref_clf is not None:
        y_mid0, _, _ = hyperplane_lines(ref_clf, xs)
        ax.plot(xs, y_mid0, color="#888", linewidth=1.4, linestyle="--",
                alpha=0.85)
    ax.plot(xs, y_mid, color=C_HYP, linewidth=2.2)
    for offset in (1, -1):
        ax.plot(xs, -((clf.coef_[0][0] * xs + clf.intercept_[0] - offset)
                      / clf.coef_[0][1]),
                color=C_MARGIN, linestyle="--", linewidth=1.0)
    scatter_two_class(ax, X, y, s=42)
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="none",
               edgecolors=C_ARROW, linewidths=2)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    return clf


X, y = make_separable_2d(n_per=14, sep=2.8, noise=0.55, seed=2)
clf0 = SVC(kernel="linear", C=1e6).fit(X, y)
sv_mask = np.zeros(len(X), dtype=bool)
sv_mask[clf0.support_] = True

non_sv_pos_idx = np.where((y > 0) & ~sv_mask)[0][0]
sv_pos_idx = np.where((y > 0) & sv_mask)[0][0]

non_sv_delta = np.array([1.4, 1.4])
sv_delta = np.array([-0.7, -0.7])

X_nonsv_move = X.copy()
X_nonsv_move[non_sv_pos_idx] += non_sv_delta

X_sv_move = X.copy()
X_sv_move[sv_pos_idx] += sv_delta

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
fit_and_plot(axes[0], X, y, "Gốc")
# Panel 2: no ghost — hyperplane unchanged, one solid line is enough.
fit_and_plot(axes[1], X_nonsv_move, y,
             "Di chuyển một điểm không phải vector hỗ trợ")
# Panel 3: ghost of original to make the shift visible.
fit_and_plot(axes[2], X_sv_move, y,
             "Di chuyển một vector hỗ trợ",
             ref_clf=clf0)

axes[1].annotate("", xy=X_nonsv_move[non_sv_pos_idx],
                 xytext=X[non_sv_pos_idx],
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.8))
axes[2].annotate("", xy=X_sv_move[sv_pos_idx],
                 xytext=X[sv_pos_idx],
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.8))

fig.tight_layout(pad=0.8)
fig.savefig("../mmc-example.svg", format="svg", bbox_inches="tight")
print("Saved mmc-example.svg")
