"""Slide 10 — Hyperplane, margin band, and support vectors circled."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import make_separable_2d, scatter_two_class, C_HYP, C_MARGIN, C_ARROW

X, y = make_separable_2d(n_per=18, sep=2.8, noise=0.55, seed=5)

# Fit hard-margin-ish SVC (very large C)
clf = SVC(kernel="linear", C=1e6)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]
norm_w = np.linalg.norm(w)
margin = 1.0 / norm_w

# Decision boundary: w.x + b = 0  =>  x2 = -(w0*x1 + b) / w1
xs = np.linspace(-3.5, 3.5, 50)
line_y = -(w[0] * xs + b) / w[1]
# Margin lines: w.x + b = ±1
line_y_up = -(w[0] * xs + b - 1) / w[1]
line_y_dn = -(w[0] * xs + b + 1) / w[1]

fig, ax = plt.subplots(figsize=(5.6, 4.5))

# Shade margin band
ax.fill_between(xs, line_y_dn, line_y_up, color=C_MARGIN, alpha=0.18)
ax.plot(xs, line_y, color=C_HYP, linewidth=2)
ax.plot(xs, line_y_up, color=C_MARGIN, linewidth=1.3, linestyle="--")
ax.plot(xs, line_y_dn, color=C_MARGIN, linewidth=1.3, linestyle="--")

scatter_two_class(ax, X, y, s=50)

# Circle support vectors
sv = clf.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none", edgecolors=C_ARROW,
           linewidths=2.2, label="vectơ hỗ trợ")

# Annotate margin
# Draw a perpendicular from boundary to margin line to show distance M
mid_x = 0.5
mid_y = -(w[0] * mid_x + b) / w[1]
u = w / norm_w  # unit normal
ax.annotate("", xy=(mid_x + margin * u[0], mid_y + margin * u[1]),
            xytext=(mid_x, mid_y),
            arrowprops=dict(arrowstyle="<->", color=C_ARROW, lw=1.6))
ax.text(mid_x + 0.7 * u[0] + 0.15, mid_y + 0.7 * u[1] - 0.1, r"$M$",
        color=C_ARROW, fontsize=13, fontweight="bold")

ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("../margin-and-sv.svg", format="svg", bbox_inches="tight")
print("Saved margin-and-sv.svg")
