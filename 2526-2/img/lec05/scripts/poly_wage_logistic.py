"""
Page 4: Degree-4 logistic regression for Pr(Wage > 250 | Age).
Output: ../poly-wage-logistic.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values
wage = wage_data['wage'].values
high = (wage > 250).astype(int)

def poly_design(x, degree=4):
    return np.column_stack([x**d for d in range(1, degree + 1)])

X     = sm.add_constant(poly_design(age))
model = sm.Logit(high, X).fit(disp=0)

age_grid = np.linspace(18, 80, 500)
X_grid   = sm.add_constant(poly_design(age_grid))
pred     = model.get_prediction(X_grid).summary_frame(alpha=0.05)
p_fit    = pred['predicted'].values
p_lo     = pred['ci_lower'].values
p_hi     = pred['ci_upper'].values

fig, ax = plt.subplots(figsize=(6, 4))

ax.fill_between(age_grid, p_lo, p_hi, color='#2244aa', alpha=0.15)
ax.plot(age_grid, p_fit, color='#1a2e6e', linewidth=2)
ax.plot(age_grid, p_lo,  color='#1a2e6e', linewidth=1, linestyle='--')
ax.plot(age_grid, p_hi,  color='#1a2e6e', linewidth=1, linestyle='--')

# Rug plots
ax.plot(age[high == 1], np.full(high.sum(),    0.198), '|', color='#333', alpha=0.6, markersize=4)
ax.plot(age[high == 0], np.full((high==0).sum(), 0.002), '|', color='#333', alpha=0.1, markersize=4)

ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Pr(Wage > 250 | Age)', fontsize=11)
ax.set_xlim(18, 80)
ax.set_ylim(-0.01, 0.21)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../poly-wage-logistic.svg', format='svg', bbox_inches='tight')
print('Saved: ../poly-wage-logistic.svg')
