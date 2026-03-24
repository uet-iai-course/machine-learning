"""
Pages 14 & 15: Natural cubic spline on ISLR Wage data.
Knots at quartiles of Age (Q1=33.75, Q2=42, Q3=51).

Outputs:
  ../natural-spline-regression.svg  (page 14)
  ../natural-spline-logistic.svg    (page 15)
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patsy import dmatrix

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values.astype(float)
wage = wage_data['wage'].values.astype(float)
high = (wage > 250).astype(int)
age_grid = np.linspace(age.min(), age.max(), 500)

knots   = np.percentile(age, [25, 50, 75])   # [33.75, 42., 51.]
formula = f"cr(age, knots=({knots[0]},{knots[1]},{knots[2]}))"
X_mat   = np.array(dmatrix(formula, {'age': age},      return_type='matrix'))
Xg_mat  = np.array(dmatrix(formula, {'age': age_grid}, return_type='matrix'))

# ── Page 14: Regression ───────────────────────────────────────────────────────
m_reg  = sm.OLS(wage, X_mat).fit()
pred   = m_reg.get_prediction(Xg_mat).summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(age, wage, s=5, color='#cccccc', alpha=0.5, zorder=1)
ax.fill_between(age_grid, pred['mean_ci_lower'], pred['mean_ci_upper'],
                color='#cc2222', alpha=0.12, zorder=2)
ax.plot(age_grid, pred['mean'],          color='#cc2222', linewidth=2,   zorder=3)
ax.plot(age_grid, pred['mean_ci_lower'], color='#cc2222', linewidth=1,
        linestyle='--', zorder=3)
ax.plot(age_grid, pred['mean_ci_upper'], color='#cc2222', linewidth=1,
        linestyle='--', zorder=3)
for k in knots:
    ax.axvline(k, color='#aaaaaa', linestyle='--', linewidth=0.9, zorder=2)
ax.text(0.98, 0.04, 'natural cubic spline\n3 knots at quartiles',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7.5, color='#555', fontstyle='italic')
ax.set_xlabel('Age', fontsize=11); ax.set_ylabel('Wage', fontsize=11)
ax.set_xlim(18, 80); ax.set_ylim(0, 330)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../natural-spline-regression.svg', format='svg', bbox_inches='tight')
plt.close(); print('Saved: ../natural-spline-regression.svg')

# ── Page 15: Logistic ─────────────────────────────────────────────────────────
m_log  = sm.Logit(high, X_mat).fit(disp=0)
pred_l = m_log.get_prediction(Xg_mat).summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.fill_between(age_grid, pred_l['ci_lower'], pred_l['ci_upper'],
                color='#cc2222', alpha=0.12)
ax.plot(age_grid, pred_l['predicted'], color='#cc2222', linewidth=2)
ax.plot(age_grid, pred_l['ci_lower'],  color='#cc2222', linewidth=1, linestyle='--')
ax.plot(age_grid, pred_l['ci_upper'],  color='#cc2222', linewidth=1, linestyle='--')
for k in knots:
    ax.axvline(k, color='#aaaaaa', linestyle='--', linewidth=0.9, zorder=2)
ax.plot(age[high==1], np.full(high.sum(),        0.198), '|',
        color='#333', alpha=0.6,  markersize=4, markeredgewidth=0.8)
ax.plot(age[high==0], np.full((high==0).sum(), 0.002), '|',
        color='#333', alpha=0.15, markersize=4, markeredgewidth=0.8)
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Pr(Wage > 250 | Age)', fontsize=11)
ax.set_xlim(18, 80); ax.set_ylim(-0.01, 0.21)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../natural-spline-logistic.svg', format='svg', bbox_inches='tight')
plt.close(); print('Saved: ../natural-spline-logistic.svg')
