"""Slide 28 — Compare linear / polynomial / RBF kernels on the same concentric dataset."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from _common import C_POS, C_NEG

rng = np.random.RandomState(1)
n = 60
r1 = rng.rand(n) * 1.2
theta1 = rng.rand(n) * 2 * np.pi
X_pos = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
r2 = 2.0 + rng.rand(n) * 0.8
theta2 = rng.rand(n) * 2 * np.pi
X_neg = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n), -np.ones(n)])


def plot(ax, clf, title):
    clf.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-3.2, 3.2, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=[C_NEG, C_POS], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[0], colors=["#333"], linewidths=1.6)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c=C_POS, s=30, edgecolors="white", linewidths=0.4)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c=C_NEG, s=30, edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
plot(axes[0], SVC(kernel="linear", C=1.0), "Kernel tuyến tính")
plot(axes[1], SVC(kernel="poly", degree=3, C=1.0, coef0=1), "Kernel đa thức (bậc 3)")
plot(axes[2], SVC(kernel="rbf", gamma=0.5, C=1.0), "Kernel RBF")

fig.tight_layout(pad=0.6)
fig.savefig("../kernel-comparison.svg", format="svg", bbox_inches="tight")
print("Saved kernel-comparison.svg")
