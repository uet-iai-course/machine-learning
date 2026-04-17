"""Heart dataset: train and plot UNPRUNED classification tree."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load Heart disease dataset
heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data
y = (heart.target == "present").astype(int)
X = X.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)

# Fully grown (unpruned) tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(dt, ax=ax, filled=True, rounded=True, fontsize=7,
          feature_names=X.columns.tolist(),
          class_names=["No", "Yes"],
          impurity=False, proportion=True)

fig.tight_layout(pad=0.3)
fig.savefig("../heart-unpruned.svg", format="svg", bbox_inches="tight")
print(f"Saved heart-unpruned.svg (depth={dt.get_depth()}, leaves={dt.get_n_leaves()})")
