"""Heart dataset: train classification tree and plot pruned tree."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

# Load Heart disease dataset
heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data
y = (heart.target == "present").astype(int) if heart.target.dtype == object or hasattr(heart.target, "cat") else (heart.target.astype(int) > 0).astype(int)

# Encode categoricals if any
X = X.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)

# Find best tree size by CV
best_score = -np.inf
best_leaves = 6
for n_leaves in range(2, 20):
    dt = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=42)
    scores = cross_val_score(dt, X, y, cv=6, scoring="accuracy")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_leaves = n_leaves

# Train final tree
dt = DecisionTreeClassifier(max_leaf_nodes=best_leaves, random_state=42)
dt.fit(X, y)

fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(dt, ax=ax, filled=True, rounded=True, fontsize=10,
          feature_names=X.columns.tolist(),
          class_names=["No", "Yes"],
          impurity=False, proportion=True)

fig.tight_layout(pad=0.3)
fig.savefig("../heart-tree.svg", format="svg", bbox_inches="tight")
print(f"Saved heart-tree.svg (best_leaves={best_leaves}, CV accuracy={best_score:.3f})")
