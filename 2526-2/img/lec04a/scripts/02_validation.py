"""
Recreate img/lec04a/02.png using ISLR Auto dataset.

Left panel : mpg vs horsepower scatter + polynomial fits (degree 1, 2, 5)
Right panel: validation MSE vs polynomial degree (1-10), one random 50-50 split
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

# ── Left panel: scatter + polynomial fits ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

ax1.scatter(X, y, color='#aaaaaa', s=14, alpha=0.6, zorder=1)

hp_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colors  = ['#e8a030', '#e05050', '#2a9d8f']
labels  = ['Linear', 'Degree 2', 'Degree 5']
degrees = [1, 2, 5]

for d, col, lbl in zip(degrees, colors, labels):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d), LinearRegression())
    pipe.fit(X.reshape(-1, 1), y)
    ax1.plot(hp_grid, pipe.predict(hp_grid.reshape(-1, 1)),
             color=col, linewidth=2, label=lbl, zorder=2)

ax1.set_xlabel('Horsepower', fontsize=10)
ax1.set_ylabel('Miles per gallon', fontsize=10)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=8)

# formula annotation
ax1.text(0.32, 0.92,
         r'$\mathrm{mpg} = \beta_0 + \beta_1 \times \mathrm{hp} + \beta_2 \times \mathrm{hp}^2 + \varepsilon$',
         transform=ax1.transAxes, fontsize=8.5, color='#c04040',
         ha='center', va='top')

# ── Right panel: validation MSE vs degree ─────────────────────────────────────
np.random.seed(1)
n   = len(X)
idx = np.random.permutation(n)
train_idx = idx[:n // 2]
test_idx  = idx[n // 2:]

X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx],  y[test_idx]

max_deg = 10
mse_val = []
for d in range(1, max_deg + 1):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d), LinearRegression())
    pipe.fit(X_tr.reshape(-1, 1), y_tr)
    mse_val.append(mean_squared_error(y_te, pipe.predict(X_te.reshape(-1, 1))))

degs = np.arange(1, max_deg + 1)
ax2.plot(degs, mse_val, color='#e05050', linewidth=1.8, marker='o',
         markersize=5, markerfacecolor='#e05050')
ax2.set_xlabel('Degree of Polynomial', fontsize=10)
ax2.set_ylabel('Mean Squared Error', fontsize=10)
ax2.set_xticks(degs)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=8)


plt.tight_layout()
plt.savefig('../auto-polynomial-validation.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../auto-polynomial-validation.svg')
