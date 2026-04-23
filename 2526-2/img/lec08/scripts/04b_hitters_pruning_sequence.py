"""Three trees pruned from the same full tree at increasing alpha."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(42)
n = 263
# Simulate 9 Hitters-like features (match 04_hitters_pruning.py exactly)
years = np.random.exponential(5, n).clip(1, 24)
hits = np.random.normal(100, 50, n).clip(1, 238)
runs = np.random.normal(50, 25, n).clip(0, 130)
rbis = np.random.normal(45, 25, n).clip(0, 120)
walks = np.random.normal(35, 20, n).clip(0, 100)
putouts = np.random.normal(200, 150, n).clip(0, 700)
assists = np.random.normal(100, 100, n).clip(0, 400)
errors = np.random.normal(8, 5, n).clip(0, 30)
atbat = np.random.normal(350, 120, n).clip(50, 700)

feature_names = ["Years", "Hits", "Runs", "RBIs", "Walks",
                 "PutOuts", "Assists", "Errors", "AtBat"]
X = np.column_stack([years, hits, runs, rbis, walks, putouts, assists, errors, atbat])
y = 4.5 + 0.08*years + 0.005*hits + 0.003*runs + 0.004*rbis + np.random.normal(0, 0.5, n)

idx = np.random.permutation(n)
X_train, y_train = X[idx[:132]], y[idx[:132]]

# "Full" tree capped at depth 4 so we can still read the nodes in slide view
full_tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X_train, y_train)

# Cost-complexity pruning path (computed on the capped tree so alphas align with it)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas


def fit_at(alpha):
    return DecisionTreeRegressor(
        max_depth=4, ccp_alpha=alpha, random_state=42
    ).fit(X_train, y_train)


# Pick 3 alphas: 0 (full), optimal (~3 leaves), large (stump)
alpha_full = alphas[0]
alpha_mid = next(
    (a for a in alphas if fit_at(a).get_n_leaves() <= 3),
    alphas[len(alphas) // 2],
)
alpha_stump = alphas[-1] * 2

chosen = [
    (alpha_full,  "α = 0",                            "Cây đầy đủ"),
    (alpha_mid,   rf"α $\approx$ {alpha_mid:.2g}",     "CV chọn (~3 lá)"),
    (alpha_stump, "α rất lớn",                        "Stump (1 lá)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (a, alpha_label, caption) in zip(axes, chosen):
    t = fit_at(a)
    plot_tree(
        t, ax=ax, filled=True, rounded=True, fontsize=8,
        feature_names=feature_names, impurity=False, proportion=False,
        precision=2, label="none",
    )
    n_leaves = t.get_n_leaves()
    ax.set_title(f"{alpha_label}  •  {n_leaves} lá\n{caption}",
                 fontsize=11, fontweight="bold")

fig.tight_layout(pad=0.6)
fig.savefig("../hitters-pruning-sequence.svg", format="svg", bbox_inches="tight")
print(f"Saved hitters-pruning-sequence.svg "
      f"(alphas = {[f'{a:.3g}' for a, _, _ in chosen]})")
