"""Slide 31 — Heart: linear SVM vs RBF SVM with different gamma (train + test ROC)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c).values
y = (heart.target == "present").astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=207, random_state=42, stratify=y)

configs = [
    ("Linear", dict(kernel="linear", C=1.0),           "#c0392b"),
    (r"RBF $\gamma=10^{-3}$", dict(kernel="rbf", gamma=1e-3, C=1.0), "#2c6ea3"),
    (r"RBF $\gamma=10^{-2}$", dict(kernel="rbf", gamma=1e-2, C=1.0), "#5aaa44"),
    (r"RBF $\gamma=10^{-1}$", dict(kernel="rbf", gamma=1e-1, C=1.0), "#d4a017"),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for label, kw, color in configs:
    pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC(**kw))])
    pipe.fit(X_tr, y_tr)
    for ax, X_, y_ in [(axes[0], X_tr, y_tr), (axes[1], X_te, y_te)]:
        s = pipe.decision_function(X_)
        fpr, tpr, _ = roc_curve(y_, s)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{label} (AUC={auc(fpr, tpr):.3f})")

for ax, title in [(axes[0], "Tập huấn luyện"), (axes[1], "Tập kiểm tra")]:
    ax.plot([0, 1], [0, 1], color="#aaa", linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("../heart-kernels.svg", format="svg", bbox_inches="tight")
print("Saved heart-kernels.svg")
