"""Slide 30 — Heart dataset: ROC curves of linear SVM vs LDA (train + test)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c).values
y = (heart.target == "present").astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=207, random_state=42, stratify=y)

svm_pipe = Pipeline([("scale", StandardScaler()), ("clf", LinearSVC(C=1.0, max_iter=10000))])
lda = LinearDiscriminantAnalysis()

svm_pipe.fit(X_tr, y_tr)
lda.fit(X_tr, y_tr)

def roc_pair(model, X_tr, y_tr, X_te, y_te, is_svm=False):
    if is_svm:
        s_tr = model.decision_function(X_tr)
        s_te = model.decision_function(X_te)
    else:
        s_tr = model.decision_function(X_tr)
        s_te = model.decision_function(X_te)
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, s_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, s_te)
    return (fpr_tr, tpr_tr, auc(fpr_tr, tpr_tr)), (fpr_te, tpr_te, auc(fpr_te, tpr_te))

svm_tr, svm_te = roc_pair(svm_pipe, X_tr, y_tr, X_te, y_te, is_svm=True)
lda_tr, lda_te = roc_pair(lda, X_tr, y_tr, X_te, y_te)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (svm, lda_r), title in [
    (axes[0], (svm_tr, lda_tr), "Tập huấn luyện"),
    (axes[1], (svm_te, lda_te), "Tập kiểm tra"),
]:
    ax.plot(svm[0], svm[1], color="#c0392b", linewidth=2,
            label=f"SVM tuyến tính (AUC={svm[2]:.3f})")
    ax.plot(lda_r[0], lda_r[1], color="#2c6ea3", linewidth=2,
            label=f"LDA (AUC={lda_r[2]:.3f})")
    ax.plot([0, 1], [0, 1], color="#aaa", linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("../heart-svm-vs-lda.svg", format="svg", bbox_inches="tight")
print("Saved heart-svm-vs-lda.svg")
