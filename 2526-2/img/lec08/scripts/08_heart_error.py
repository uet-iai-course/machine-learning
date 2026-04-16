"""Heart dataset: error vs tree size plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load Heart
heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)
y = (heart.target == "present").astype(int) if heart.target.dtype == object or hasattr(heart.target, "cat") else (heart.target.astype(int) > 0).astype(int)

# Split
np.random.seed(42)
idx = np.random.permutation(len(X))
n_train = 207
X_train, y_train = X.iloc[idx[:n_train]], y.iloc[idx[:n_train]]
X_test, y_test = X.iloc[idx[n_train:]], y.iloc[idx[n_train:]]

leaf_sizes = list(range(2, 20))
train_err = []
cv_err = []
test_err = []

for n_leaves in leaf_sizes:
    dt = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=42)
    dt.fit(X_train, y_train)

    train_err.append(1 - dt.score(X_train, y_train))
    test_err.append(1 - dt.score(X_test, y_test))

    scores = cross_val_score(dt, X_train, y_train, cv=6, scoring="accuracy")
    cv_err.append(1 - scores.mean())

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(leaf_sizes, train_err, "o-", color="#2c3e50", linewidth=1.5, markersize=4, label="Training")
ax.plot(leaf_sizes, cv_err, "s-", color="#e67e22", linewidth=1.5, markersize=4, label="Cross-Validation")
ax.plot(leaf_sizes, test_err, "^-", color="#27ae60", linewidth=1.5, markersize=4, label="Test")

# Mark best CV
best_idx = np.argmin(cv_err)
ax.axvline(leaf_sizes[best_idx], color="#3498db", linewidth=1.5, linestyle="--", alpha=0.7)

ax.set_xlabel("Số lá (Tree Size)", fontsize=10)
ax.set_ylabel("Error Rate", fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

fig.tight_layout(pad=0.5)
fig.savefig("../heart-error.svg", format="svg", bbox_inches="tight")
print("Saved heart-error.svg")
