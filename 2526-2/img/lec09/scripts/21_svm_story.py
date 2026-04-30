"""Slide 35 — Three-panel story: hard → soft → kernel."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import scatter_two_class, C_HYP, C_MARGIN, C_POS, C_NEG

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))


def plot_linear(ax, X, y, C, title):
    clf = SVC(kernel="linear", C=C).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(-3.5, 3.5, 80)
    ax.fill_between(xs, -(w[0] * xs + b + 1) / w[1], -(w[0] * xs + b - 1) / w[1],
                    color=C_MARGIN, alpha=0.18)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=C_HYP, linewidth=1.8)
    for off in (1, -1):
        ax.plot(xs, -(w[0] * xs + b - off) / w[1], color=C_MARGIN, linestyle="--", linewidth=1.0)
    scatter_two_class(ax, X, y, s=32)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")


def plot_rbf(ax, X, y, title):
    clf = SVC(kernel="rbf", gamma=0.5, C=1.0).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-3.3, 3.3, 200), np.linspace(-3.3, 3.3, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=[C_NEG, C_POS], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[0], colors=["#333"], linewidths=1.6)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c=C_POS, s=30, edgecolors="white", linewidths=0.4)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c=C_NEG, s=30, edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")


# Panel 1: hard margin on well-separated data
rng = np.random.RandomState(5)
X1 = rng.randn(18, 2) * 0.45 + [1.3, 1.2]
X2 = rng.randn(18, 2) * 0.45 + [-1.3, -1.2]
X_hard = np.vstack([X1, X2])
y_hard = np.concatenate([np.ones(18), -np.ones(18)])
plot_linear(axes[0], X_hard, y_hard, 1e6, "Lề cứng")

# Panel 2: soft margin on slightly overlapping data
rng2 = np.random.RandomState(7)
X1 = rng2.randn(22, 2) * 0.95 + [1.0, 1.0]
X2 = rng2.randn(22, 2) * 0.95 + [-0.9, -0.9]
X_soft = np.vstack([X1, X2])
y_soft = np.concatenate([np.ones(22), -np.ones(22)])
plot_linear(axes[1], X_soft, y_soft, 1.0, "Lề mềm")

# Panel 3: kernel on concentric data
rng3 = np.random.RandomState(1)
n = 35
r1 = rng3.rand(n) * 1.1
theta1 = rng3.rand(n) * 2 * np.pi
X_pos = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
r2 = 2.0 + rng3.rand(n) * 0.7
theta2 = rng3.rand(n) * 2 * np.pi
X_neg = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
X_ker = np.vstack([X_pos, X_neg])
y_ker = np.concatenate([np.ones(n), -np.ones(n)])
plot_rbf(axes[2], X_ker, y_ker, "Kernel (RBF)")

fig.tight_layout(pad=0.5)
fig.savefig("../svm-story.svg", format="svg", bbox_inches="tight")
print("Saved svm-story.svg")
