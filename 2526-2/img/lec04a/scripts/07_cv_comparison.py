"""
Recreate img/lec04a/07.png — comparison of validation set, LOOCV, and 10-fold CV
on Auto dataset (mpg ~ poly(horsepower, d)), degrees 1-10.

Output: ../auto-cv-comparison.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score

# ── Data ───────────────────────────────────────────────────────────────────────
auto = sm.datasets.get_rdataset('Auto', 'ISLR').data.dropna()
X = auto['horsepower'].values.astype(float).reshape(-1, 1)
y = auto['mpg'].values.astype(float)
n = len(X)
degs = np.arange(1, 11)

# ── Validation set (one 50/50 split, seed=1) ──────────────────────────────────
np.random.seed(1)
idx = np.random.permutation(n)
X_tr, y_tr = X[idx[:n//2]], y[idx[:n//2]]
X_te, y_te = X[idx[n//2:]], y[idx[n//2:]]

mse_val = []
for d in degs:
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
    pipe.fit(X_tr, y_tr)
    mse_val.append(np.mean((y_te - pipe.predict(X_te))**2))

# ── LOOCV ─────────────────────────────────────────────────────────────────────
loo = LeaveOneOut()
mse_loocv = []
for d in degs:
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
    scores = cross_val_score(pipe, X, y, cv=loo, scoring='neg_mean_squared_error')
    mse_loocv.append(-scores.mean())

# ── 10-fold CV (10 different random splits) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

for seed in range(10):
    mse_kfold = []
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for d in degs:
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
        scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_squared_error')
        mse_kfold.append(-scores.mean())
    ax.plot(degs, mse_kfold, color='#5599cc', linewidth=1.2, alpha=0.5,
            label='10-fold CV' if seed == 0 else '')

# ── Overlay LOOCV and validation set ──────────────────────────────────────────
ax.plot(degs, mse_loocv, color='#222222', linewidth=2,
        label='LOOCV')
ax.plot(degs, mse_val, color='#cc4444', linewidth=1.8, linestyle='--',
        label='Validation set (50/50)')

ax.set_xlabel('Degree of Polynomial', fontsize=10)
ax.set_ylabel('Mean Squared Error', fontsize=10)
ax.set_xticks([1, 2, 4, 6, 8, 10])
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig('../auto-cv-comparison.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../auto-cv-comparison.svg')
