"""
Page 3: Polynomial regression (degree 4) on ISLR Wage data.
Output: ../poly-wage-regression.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values
wage = wage_data['wage'].values

# Degree-4 polynomial design matrix
def poly_design(x, degree=4):
    return np.column_stack([x**d for d in range(1, degree + 1)])

X = sm.add_constant(poly_design(age))
model = sm.OLS(wage, X).fit()

age_grid = np.linspace(18, 80, 500)
X_grid   = sm.add_constant(poly_design(age_grid))
pred     = model.get_prediction(X_grid).summary_frame(alpha=0.05)
y_fit    = pred['mean'].values
y_lo     = pred['mean_ci_lower'].values
y_hi     = pred['mean_ci_upper'].values

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.scatter(age, wage, s=8, marker='s', facecolors='none', edgecolors='#aaaaaa',
           linewidths=0.5, alpha=0.6, zorder=1)
ax.fill_between(age_grid, y_lo, y_hi, color='#2244aa', alpha=0.15, zorder=2)
ax.plot(age_grid, y_fit, color='#1a2e6e', linewidth=2, zorder=3)
ax.plot(age_grid, y_lo,  color='#1a2e6e', linewidth=1, linestyle='--', zorder=3)
ax.plot(age_grid, y_hi,  color='#1a2e6e', linewidth=1, linestyle='--', zorder=3)
ax.axhline(250, color='#cc2222', linewidth=1.2, linestyle='--', zorder=4)

ax.text(79, 258, 'High earners', color='#cc2222', fontsize=8, ha='right', va='bottom')
ax.text(79, 242, 'Low earners',  color='#cc2222', fontsize=8, ha='right', va='top')

ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Wage', fontsize=11)
ax.set_xlim(18, 80)
ax.set_ylim(0, 330)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles = [
    plt.Line2D([0], [0], color='#1a2e6e', linewidth=2, label='Degree-4 fit'),
    plt.Line2D([0], [0], color='#1a2e6e', linewidth=1, linestyle='--', label='95% CI'),
]
ax.legend(handles=handles, fontsize=8, loc='upper left', frameon=False)

plt.tight_layout()
plt.savefig('../poly-wage-regression.svg', format='svg', bbox_inches='tight')
print('Saved: ../poly-wage-regression.svg')
