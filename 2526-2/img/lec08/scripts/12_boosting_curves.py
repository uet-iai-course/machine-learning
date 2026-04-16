"""Boosting vs RF test error curves on simulated high-dimensional data.

Compare Boosting (depth=1, depth=2) vs Random Forest (m=sqrt(p)).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# Same simulated cancer-like data as 10b_choosing_m.py
p = 1000
X, y = make_classification(
    n_samples=800, n_features=p, n_informative=20,
    n_redundant=200, n_classes=8, n_clusters_per_class=1,
    class_sep=2.0, flip_y=0.02, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

max_trees = 500

# Single tree error
single = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
single_err = 1 - single.score(X_test, y_test)

# --- Boosting depth=1 (stumps) ---
gb1 = GradientBoostingClassifier(
    n_estimators=max_trees, max_depth=1, learning_rate=0.1,
    random_state=42)
gb1.fit(X_train, y_train)
boost1_err = [np.mean(p != y_test) for p in gb1.staged_predict(X_test)]
print(f"Boosting d=1: final={boost1_err[-1]:.3f}")

# --- Boosting depth=2 ---
gb2 = GradientBoostingClassifier(
    n_estimators=max_trees, max_depth=2, learning_rate=0.1,
    random_state=42)
gb2.fit(X_train, y_train)
boost2_err = [np.mean(p != y_test) for p in gb2.staged_predict(X_test)]
print(f"Boosting d=2: final={boost2_err[-1]:.3f}")

# --- Random Forest m=sqrt(p) ---
rf = RandomForestClassifier(
    n_estimators=max_trees, max_features=int(np.sqrt(p)),
    random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

eval_at = list(range(1, 21)) + list(range(25, max_trees + 1, 10))
rf_err = []
for k in eval_at:
    preds = np.array([t.predict(X_test) for t in rf.estimators_[:k]])
    majority = mode(preds, axis=0).mode.ravel()
    rf_err.append(np.mean(majority != y_test))
print(f"RF m=sqrt(p): final={rf_err[-1]:.3f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 4.5))

trees_boost = list(range(1, max_trees + 1))
ax.plot(trees_boost, boost1_err, color="#cc8833", linewidth=1.5, label="Boosting: depth=1")
ax.plot(trees_boost, boost2_err, color="#5588cc", linewidth=1.5, label="Boosting: depth=2")
ax.plot(eval_at, rf_err, color="#22aa88", linewidth=1.5, label=r"Random Forest: $m=\sqrt{p}$")
ax.axhline(single_err, color="#1a237e", linewidth=2, linestyle="--", alpha=0.7, label="Cây đơn")

ax.annotate("lỗi 1 cây", xy=(max_trees * 0.7, single_err),
            xytext=(max_trees * 0.7, single_err + 0.03),
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

ax.set_xlabel("Số cây (Number of Trees)", fontsize=10)
ax.set_ylabel("Test Classification Error", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("../boosting-curves.svg", format="svg", bbox_inches="tight")
print("Saved boosting-curves.svg")
