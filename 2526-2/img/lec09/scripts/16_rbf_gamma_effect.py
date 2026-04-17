"""Slide 27 — RBF kernel: gamma controls locality."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import C_POS, C_NEG

rng = np.random.RandomState(2)
n = 45
# Two "islands" per class to showcase locality
X_pos = np.vstack([
    rng.randn(n, 2) * 0.5 + [1.2, 1.4],
    rng.randn(n // 2, 2) * 0.5 + [-1.8, -0.5],
])
X_neg = np.vstack([
    rng.randn(n, 2) * 0.5 + [-0.5, 1.5],
    rng.randn(n // 2, 2) * 0.5 + [1.5, -1.3],
])
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(len(X_pos)), -np.ones(len(X_neg))])


def plot_rbf(ax, X, y, gamma, title):
    clf = SVC(kernel="rbf", gamma=gamma, C=10.0).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=[C_NEG, C_POS], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[0], colors=["#333"], linewidths=1.6)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c=C_POS, s=30, edgecolors="white", linewidths=0.4)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c=C_NEG, s=30, edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
plot_rbf(axes[0], X, y, gamma=0.05, title=r"$\gamma$ nhỏ — mượt, ít cục bộ")
plot_rbf(axes[1], X, y, gamma=0.5,  title=r"$\gamma$ vừa")
plot_rbf(axes[2], X, y, gamma=5.0,  title=r"$\gamma$ lớn — rất cục bộ (overfit)")

fig.tight_layout(pad=0.6)
fig.savefig("../rbf-gamma-effect.svg", format="svg", bbox_inches="tight")
print("Saved rbf-gamma-effect.svg")
