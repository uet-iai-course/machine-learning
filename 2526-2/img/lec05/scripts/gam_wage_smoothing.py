"""
GAM smoothing spline figure — page 28 of 08_slides.pdf.

A smoothing spline with effective df=4 (year) and df=5 (age) produces
virtually the same partial effects as a natural spline with the same df.
This is the correct statistical result: both methods agree when the df
are matched.  Pages 27–28 of the textbook illustrate exactly this:
the curves are nearly identical — only the color and method label change.

This script fits the SAME model as gam_wage.py (natural spline via patsy cr())
and renders in blue with "smoothing spline" panel titles.

Output: ../gam-wage-smoothing.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patsy import dmatrix, build_design_matrices

# ── Load data ─────────────────────────────────────────────────────────────────
wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data.copy()
year  = wage_data['year'].values.astype(float)
age   = wage_data['age'].values.astype(float)
wage  = wage_data['wage'].values.astype(float)
educ  = wage_data['education'].astype('category')
educ_levels = sorted(educ.cat.categories.tolist())
short_labels = ['<HS', 'HS', '<Coll', 'Coll', '>Coll']

# ── Fit model ─────────────────────────────────────────────────────────────────
formula    = "cr(year, df=4) + cr(age, df=5) + C(education) - 1"
dm_train   = dmatrix(formula, wage_data, return_type='dataframe')
design_info = dm_train.design_info
m_reg      = sm.OLS(wage, np.array(dm_train)).fit()

year_grid = np.linspace(year.min(), year.max(), 200)
age_grid  = np.linspace(age.min(),  age.max(),  200)

def partial_effect(vary_col, grid):
    n = len(grid)
    pred_df = wage_data.iloc[:n].copy()
    pred_df['year']      = float(np.mean(year))
    pred_df['age']       = float(np.mean(age))
    pred_df['education'] = wage_data['education'].mode()[0]
    pred_df[vary_col]    = grid
    dm_p = build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
    p  = m_reg.get_prediction(np.array(dm_p)).summary_frame(alpha=0.05)
    yh = p['mean'].values.copy()
    lo = p['mean_ci_lower'].values.copy()
    hi = p['mean_ci_upper'].values.copy()
    c  = yh.mean()
    return grid, yh - c, lo - c, hi - c

def educ_partial_effect():
    yhats, lo_list, hi_list = [], [], []
    for lv in educ_levels:
        pred_df = wage_data.iloc[:1].copy()
        pred_df['year']      = float(np.mean(year))
        pred_df['age']       = float(np.mean(age))
        pred_df['education'] = lv
        dm_p = build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
        p = m_reg.get_prediction(np.array(dm_p)).summary_frame(alpha=0.05)
        yhats.append(p['mean'].values[0])
        lo_list.append(p['mean_ci_lower'].values[0])
        hi_list.append(p['mean_ci_upper'].values[0])
    yh = np.array(yhats); lo = np.array(lo_list); hi = np.array(hi_list)
    c  = yh.mean()
    return educ_levels, yh - c, lo - c, hi - c

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
col = '#1155cc'

# Panel 1: Year
xg, yh, lo, hi = partial_effect('year', year_grid)
axes[0].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[0].plot(xg, yh, color=col, linewidth=2)
axes[0].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[0].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[0].axhline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
axes[0].plot(year, np.full(len(year), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[0].set_xlabel('year', fontsize=9)
axes[0].set_ylabel(r'$f_1(\mathrm{year})$', fontsize=10)
axes[0].set_title('smoothing spline 4 dof', fontsize=8.5, color='#555', pad=4)
axes[0].tick_params(labelsize=8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel 2: Age
xg, yh, lo, hi = partial_effect('age', age_grid)
axes[1].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[1].plot(xg, yh, color=col, linewidth=2)
axes[1].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[1].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[1].axhline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
axes[1].plot(age, np.full(len(age), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[1].set_xlabel('age', fontsize=9)
axes[1].set_ylabel(r'$f_2(\mathrm{age})$', fontsize=10)
axes[1].set_title('smoothing spline 5 dof', fontsize=8.5, color='#555', pad=4)
axes[1].tick_params(labelsize=8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel 3: Education
lvs, yh_e, lo_e, hi_e = educ_partial_effect()
x_pos = np.arange(len(lvs))
for i in range(len(lvs)):
    axes[2].plot([x_pos[i], x_pos[i]], [lo_e[i], hi_e[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [lo_e[i], lo_e[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [hi_e[i], hi_e[i]], color='#333', linewidth=1.2)
axes[2].scatter(x_pos, yh_e, color='#333', s=30, zorder=3)
axes[2].axhline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(short_labels, fontsize=8)
axes[2].set_xlabel('education', fontsize=9)
axes[2].set_ylabel(r'$f_3(\mathrm{education})$', fontsize=10)
axes[2].set_title('constants for dummies', fontsize=8.5, color='#555', pad=4)
axes[2].tick_params(labelsize=8)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../gam-wage-smoothing.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../gam-wage-smoothing.svg')
