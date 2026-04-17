"""RF: effect of m (max_features) on test error — simulated cancer-like dataset.

Train RF once with 500 trees, then evaluate cumulative error as trees are added.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# Simulate high-dimensional multi-class data with many correlated features
# Few informative + many redundant → RF benefits from small m (decorrelation)
p = 1000
X, y = make_classification(
    n_samples=800, n_features=p, n_informative=20,
    n_redundant=200, n_classes=8, n_clusters_per_class=1,
    class_sep=2.0, flip_y=0.02, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

max_trees = 500
sqrt_p = int(np.sqrt(p))
configs = [
    (r"$m = p$ (bagging)", p, "#cc8833"),
    (r"$m = p/2$", p // 2, "#5588cc"),
    (r"$m = \sqrt{p}$", sqrt_p, "#22aa88"),
]

eval_at = list(range(1, 21)) + list(range(25, max_trees + 1, 10))

fig, ax = plt.subplots(figsize=(5.5, 4.2))

first_err = None
for label, max_feat, color in configs:
    rf = RandomForestClassifier(
        n_estimators=max_trees, max_features=max_feat,
        random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    errors = []
    for k in eval_at:
        preds = np.array([t.predict(X_test) for t in rf.estimators_[:k]])
        majority = mode(preds, axis=0).mode.ravel()
        errors.append(np.mean(majority != y_test))

    ax.plot(eval_at, errors, color=color, linewidth=1.5, label=label)
    print(f"  {label}: final={errors[-1]:.3f}, single={errors[0]:.3f}")
    if first_err is None:
        first_err = errors[0]

# Annotate single-tree error
ax.annotate("lỗi 1 cây", xy=(1, first_err),
            xytext=(70, first_err + 0.02),
            fontsize=8, color="#cc3333",
            arrowprops=dict(arrowstyle="->", color="#cc3333", lw=1.2))

ax.set_xlabel("Số cây (Number of Trees)", fontsize=10)
ax.set_ylabel("Test Classification Error", fontsize=10)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("../choosing-m.svg", format="svg", bbox_inches="tight")
print("Saved choosing-m.svg")
