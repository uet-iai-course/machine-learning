"""
Recreate img/lec04a/03.png — multiple 50-50 validation splits on Auto dataset.
Each line = one random split. Shows variability of the validation MSE estimate.

Output: ../auto-validation-variability.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# ── Data ───────────────────────────────────────────────────────────────────────
auto = sm.datasets.get_rdataset('Auto', 'ISLR').data.dropna()
X = auto['horsepower'].values.astype(float)
y = auto['mpg'].values.astype(float)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

degs = np.arange(1, 11)
n    = len(X)

for seed in range(10):
    np.random.seed(seed)
    idx  = np.random.permutation(n)
    X_tr, y_tr = X[idx[:n // 2]], y[idx[:n // 2]]
    X_te, y_te = X[idx[n // 2:]], y[idx[n // 2:]]

    mses = []
    for d in degs:
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d),
                             LinearRegression())
        pipe.fit(X_tr.reshape(-1, 1), y_tr)
        mses.append(mean_squared_error(y_te, pipe.predict(X_te.reshape(-1, 1))))

    ax.plot(degs, mses, linewidth=1.6, alpha=0.85)

ax.set_xlabel('Degree of Polynomial', fontsize=10)
ax.set_ylabel('Mean Squared Error', fontsize=10)
ax.set_xticks([1, 2, 4, 6, 8, 10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig('../auto-validation-variability.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../auto-validation-variability.svg')
