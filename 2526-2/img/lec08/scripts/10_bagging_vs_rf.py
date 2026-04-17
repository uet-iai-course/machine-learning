"""Bagging vs Random Forest error curves on Heart dataset."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load Heart
heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c).values
y = ((heart.target == "present").astype(int) if heart.target.dtype == object or hasattr(heart.target, "cat") else (heart.target.astype(int) > 0).astype(int)).values

np.random.seed(42)
idx = np.random.permutation(len(X))
n_train = 207
X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

n_trees_list = list(range(1, 301, 5))

# Bagging = RF with max_features = n_features
bag_test = []
rf_test = []
bag_oob = []
rf_oob = []

for n_trees in n_trees_list:
    # Bagging
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_trees, random_state=42, oob_score=True)
    bag.fit(X_train, y_train)
    bag_test.append(1 - bag.score(X_test, y_test))
    bag_oob.append(1 - bag.oob_score_)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)
    rf_test.append(1 - rf.score(X_test, y_test))
    rf_oob.append(1 - rf.oob_score_)

# Single tree error
single = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
single_err = 1 - single.score(X_test, y_test)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(n_trees_list, bag_test, color="#2c3e50", linewidth=1.2, label="Test: Bagging")
ax.plot(n_trees_list, rf_test, color="#e67e22", linewidth=1.2, label="Test: Random Forest")
ax.plot(n_trees_list, bag_oob, color="#2c3e50", linewidth=1, linestyle="--", alpha=0.5, label="OOB: Bagging")
ax.plot(n_trees_list, rf_oob, color="#e67e22", linewidth=1, linestyle="--", alpha=0.5, label="OOB: Random Forest")
ax.axhline(single_err, color="#c0392b", linewidth=1.5, linestyle=":", label="Cây đơn")

ax.set_xlabel("Số cây", fontsize=11)
ax.set_ylabel("Error Rate", fontsize=11)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout(pad=0.5)
fig.savefig("../bagging-vs-rf.svg", format="svg", bbox_inches="tight")
print("Saved bagging-vs-rf.svg")
