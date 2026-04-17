"""Hitters-like: train regression tree, plot MSE vs tree size (train/CV/test)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 263
# Simulate 9 features
years = np.random.exponential(5, n).clip(1, 24)
hits = np.random.normal(100, 50, n).clip(1, 238)
runs = np.random.normal(50, 25, n).clip(0, 130)
rbis = np.random.normal(45, 25, n).clip(0, 120)
walks = np.random.normal(35, 20, n).clip(0, 100)
putouts = np.random.normal(200, 150, n).clip(0, 700)
assists = np.random.normal(100, 100, n).clip(0, 400)
errors = np.random.normal(8, 5, n).clip(0, 30)
atbat = np.random.normal(350, 120, n).clip(50, 700)

X = np.column_stack([years, hits, runs, rbis, walks, putouts, assists, errors, atbat])
y = 4.5 + 0.08*years + 0.005*hits + 0.003*runs + 0.004*rbis + np.random.normal(0, 0.5, n)

idx = np.random.permutation(n)
n_train = 132
X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

leaf_sizes = list(range(2, 18))
train_mse, cv_mse, test_mse = [], [], []

for n_leaves in leaf_sizes:
    dt = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=42)
    dt.fit(X_train, y_train)
    train_mse.append(np.mean((dt.predict(X_train) - y_train)**2))
    test_mse.append(np.mean((dt.predict(X_test) - y_test)**2))
    scores = cross_val_score(dt, X_train, y_train, cv=6, scoring="neg_mean_squared_error")
    cv_mse.append(-scores.mean())

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(leaf_sizes, train_mse, "o-", color="#2c3e50", linewidth=1.5, markersize=4, label="Training")
ax.plot(leaf_sizes, cv_mse, "s-", color="#e67e22", linewidth=1.5, markersize=4, label="Cross-Validation")
ax.plot(leaf_sizes, test_mse, "^-", color="#27ae60", linewidth=1.5, markersize=4, label="Test")

ax.set_xlabel("Số lá (Tree Size)", fontsize=10)
ax.set_ylabel("Mean Squared Error", fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

fig.tight_layout(pad=0.5)
fig.savefig("../hitters-pruning.svg", format="svg", bbox_inches="tight")
print("Saved hitters-pruning.svg")
