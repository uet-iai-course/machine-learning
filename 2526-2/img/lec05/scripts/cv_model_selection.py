"""
Page 16: 10-fold CV MSE vs degrees of freedom for natural spline and cubic spline on Wage data.
df=1 → constant, df=2 → linear, df≥3 → spline.
Output: ../cv-model-selection.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values.astype(float)
wage = wage_data['wage'].values.astype(float)
kf   = KFold(n_splits=10, shuffle=True, random_state=42)

def cv_ols(X_full):
    mses = []
    for tr, te in kf.split(age):
        m = sm.OLS(wage[tr], X_full[tr]).fit()
        mses.append(mean_squared_error(wage[te], m.predict(X_full[te])))
    return np.mean(mses)

def design_ns(df_val):
    if df_val == 1: return np.ones((len(age), 1))
    if df_val == 2: return np.column_stack([np.ones(len(age)), age])
    return np.array(dmatrix(f"cr(age, df={df_val})", {'age': age}, return_type='matrix'))

def design_cs(df_val):
    if df_val == 1: return np.ones((len(age), 1))
    if df_val == 2: return np.column_stack([np.ones(len(age)), age])
    if df_val == 3: return np.column_stack([np.ones(len(age)), age, age**2, age**3])
    return np.array(dmatrix(f"bs(age, df={df_val}, include_intercept=True)",
                            {'age': age}, return_type='matrix'))

dofs   = list(range(1, 11))
ns_mse = [cv_ols(design_ns(d)) for d in dofs]
cs_mse = [cv_ols(design_cs(d)) for d in dofs]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
for ax, mse_vals, col, xlabel in [
    (axes[0], ns_mse, '#cc4444', 'Degrees of Freedom of Natural Spline'),
    (axes[1], cs_mse, '#4444cc', 'Degrees of Freedom of Cubic Spline'),
]:
    ax.plot(dofs, mse_vals, 'o--', color=col, linewidth=1.2, markersize=5,
            markerfacecolor=col, markeredgewidth=0)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel('Mean Squared Error', fontsize=9.5)
    ax.set_xticks(dofs)
    ax.tick_params(labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(True)

plt.tight_layout()
plt.savefig('../cv-model-selection.svg', format='svg', bbox_inches='tight')
print('Saved: ../cv-model-selection.svg')
