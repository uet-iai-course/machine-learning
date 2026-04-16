"""Bagging vs Boosting decision boundaries on simulated 2D data.

p=5 features, 2 classes, true boundary x1 + x2 = 1.
Left: Bagged stumps. Right: Boosted stumps.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

rng = np.random.RandomState(42)

# Generate data: 5 features, boundary at x1+x2=1, only x1,x2 matter
N = 200
X = rng.uniform(0, 2, size=(N, 5))
y = (X[:, 0] + X[:, 1] > 2).astype(int)  # diagonal boundary

# Train
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50, random_state=42)
bag.fit(X, y)
bag_err = 1 - bag.score(X, y)

boost = GradientBoostingClassifier(
    n_estimators=50, max_depth=1, learning_rate=0.2, random_state=42)
boost.fit(X, y)
boost_err = 1 - boost.score(X, y)

# Decision boundary grid (only x1, x2 vary; others at mean)
xx1, xx2 = np.meshgrid(np.linspace(0, 2, 300), np.linspace(0, 2, 300))
X_grid = np.column_stack([
    xx1.ravel(), xx2.ravel(),
    np.full(xx1.size, 1.0), np.full(xx1.size, 1.0), np.full(xx1.size, 1.0),
])

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)

for ax, model, title, err in [
    (axes[0], bag, "Bagging (stumps)", bag_err),
    (axes[1], boost, "Boosting (stumps)", boost_err),
]:
    Z = model.predict(X_grid).reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, levels=[-0.5, 0.5, 1.5],
                colors=["#fdebd0", "#d5f5e3"], alpha=0.5)
    ax.contour(xx1, xx2, Z, levels=[0.5], colors=["#7f8c8d"], linewidths=1.5)

    # True boundary
    ax.plot([0, 2], [2, 0], color="#aaa", linewidth=1.2, linestyle="--", label=r"$x_1+x_2=2$")

    # Scatter
    c0 = y == 0
    ax.scatter(X[c0, 0], X[c0, 1], c="#e67e22", s=18, edgecolors="#c0392b",
               linewidths=0.4, alpha=0.7, label="Lớp 0", zorder=3)
    ax.scatter(X[~c0, 0], X[~c0, 1], c="#27ae60", s=18, edgecolors="#1e8449",
               linewidths=0.4, alpha=0.7, label="Lớp 1", zorder=3)

    ax.set_title(f"{title}\ntest error: {err:.3f}", fontsize=10)
    ax.set_xlabel(r"$x_1$", fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

axes[0].set_ylabel(r"$x_2$", fontsize=10)
axes[0].legend(fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig("../bagging-vs-boosting-sim.svg", format="svg", bbox_inches="tight")
print(f"Saved bagging-vs-boosting-sim.svg (bag_err={bag_err:.3f}, boost_err={boost_err:.3f})")
