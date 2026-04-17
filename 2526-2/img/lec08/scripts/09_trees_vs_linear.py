"""Trees vs Linear Regression: 4 panels comparing linear and tree models on two data types."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 200

# --- Dataset 1: linearly separable ---
X_lin = np.random.randn(n, 2)
y_lin = (X_lin[:, 0] + X_lin[:, 1] > 0).astype(int)

# --- Dataset 2: regionally separable (XOR-like) ---
X_reg = np.random.randn(n, 2)
y_reg = ((X_reg[:, 0] > 0) == (X_reg[:, 1] > 0)).astype(int)

datasets = [
    ("Dữ liệu tuyến tính", X_lin, y_lin),
    ("Dữ liệu theo vùng", X_reg, y_reg),
]
models = [
    ("Mô hình tuyến tính", LogisticRegression),
    ("Mô hình cây", DecisionTreeClassifier),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

colors = ["#f4d03f", "#82e0aa"]  # yellow, green

panel_idx = 0
for col_d, (data_label, X, y) in enumerate(datasets):
    for col_m, (model_label, ModelClass) in enumerate(models):
        ax = axes[panel_idx]
        panel_idx += 1
        kw = {"max_depth": 4} if ModelClass == DecisionTreeClassifier else {}
        clf = ModelClass(**kw).fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=colors, alpha=0.4)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#333"], linewidths=1.5)

        for c in [0, 1]:
            mask = y == c
            ax.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.7,
                       color=colors[c], edgecolors="#333", linewidths=0.3)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$X_1$", fontsize=9)
        ax.set_ylabel(r"$X_2$", fontsize=9)

        ax.set_title(f"{data_label}\n{model_label}", fontsize=9, fontweight="bold")

fig.tight_layout(pad=0.6)
fig.savefig("../trees-vs-linear.svg", format="svg", bbox_inches="tight")
print("Saved trees-vs-linear.svg")
