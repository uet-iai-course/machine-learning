"""
Page 6: Step function regression on ISLR Wage data (cutpoints [33.5, 49.0, 64.5]).
Output: ../step-wage-regression.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values
wage = wage_data['wage'].values

cuts = [33.5, 49.0, 64.5]

def make_step_design(x, cutpoints):
    cp   = [-np.inf] + cutpoints + [np.inf]
    cols = [np.ones(len(x))]
    for k in range(1, len(cp) - 1):
        cols.append(((x >= cp[k]) & (x < cp[k+1])).astype(float))
    return np.column_stack(cols)

X     = make_step_design(age, cuts)
model = sm.OLS(wage, X).fit()

age_grid = np.linspace(18, 80, 500)
pred     = model.get_prediction(make_step_design(age_grid, cuts)).summary_frame(alpha=0.05)
y_fit    = pred['mean'].values
y_lo     = pred['mean_ci_lower'].values
y_hi     = pred['mean_ci_upper'].values

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.scatter(age, wage, s=8, facecolors='none', edgecolors='#bbbbbb',
           linewidths=0.4, alpha=0.5, zorder=1)
ax.fill_between(age_grid, y_lo, y_hi, color='#2a7a3a', alpha=0.12, step='post', zorder=2)
ax.step(age_grid, y_fit, where='post', color='#1a5c2a', linewidth=2,   zorder=3)
ax.step(age_grid, y_lo,  where='post', color='#1a5c2a', linewidth=1,
        linestyle='--', zorder=3)
ax.step(age_grid, y_hi,  where='post', color='#1a5c2a', linewidth=1,
        linestyle='--', zorder=3)

for c in cuts:
    ax.axvline(c, color='#aaaaaa', linestyle=':', linewidth=1, zorder=2)

# Cutpoints annotation
origin = (67.0, 280)
ax.text(origin[0] + 0.5, origin[1], 'cutpoints',
        color='#cc7700', fontsize=9, va='center', ha='left')
for c in cuts:
    ax.annotate('', xy=(c, 230), xytext=origin,
                arrowprops=dict(arrowstyle='->', color='#cc7700', lw=1.2))

ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Wage', fontsize=11)
ax.set_xlim(18, 80)
ax.set_ylim(0, 330)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../step-wage-regression.svg', format='svg', bbox_inches='tight')
print('Saved: ../step-wage-regression.svg')
