"""
GAM figures on ISLR Wage data.

Pages 27 & 32 from 08_slides.pdf:
  - Page 27: GAM regression, 3-panel (year, age, education) — red curves
  - Page 32: GAM logistic, 3-panel (year, age, education) — green curves

Model: wage ~ cr(year, df=4) + cr(age, df=5) + C(education)

Outputs:
  ../gam-wage-regression.svg
  ../gam-wage-logistic.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patsy import dmatrix, build_design_matrices

# ── Load data ─────────────────────────────────────────────────────────────────
wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
wage_data = wage_data.copy()

year  = wage_data['year'].values.astype(float)
age   = wage_data['age'].values.astype(float)
wage  = wage_data['wage'].values.astype(float)
educ  = wage_data['education'].astype('category')
high  = (wage > 250).astype(int)

# Education levels (ordered)
educ_levels = sorted(educ.cat.categories.tolist())

# ── Build design matrix (training) ────────────────────────────────────────────
formula = "cr(year, df=4) + cr(age, df=5) + C(education) - 1"
dm_train = dmatrix(formula, wage_data, return_type='dataframe')
design_info = dm_train.design_info

X_train = np.array(dm_train)

# ── Fit models ────────────────────────────────────────────────────────────────
m_reg = sm.OLS(wage, X_train).fit()
m_log = sm.Logit(high, X_train).fit(disp=0)

# ── Helper: partial effect for a continuous predictor ─────────────────────────
def partial_effect(model, vary_col, grid, wage_data, design_info,
                   fix_year=None, fix_age=None, fix_educ=None,
                   logistic=False):
    """
    Hold all other predictors fixed at mean/mode; vary `vary_col` over `grid`.
    Returns (grid, yhat, yhat_lo, yhat_hi).
    """
    n = len(grid)
    pred_df = wage_data.iloc[:n].copy()  # placeholder rows, will overwrite

    year_mean = float(np.mean(year))
    age_mean  = float(np.mean(age))
    educ_mode = wage_data['education'].mode()[0]

    pred_df['year']      = fix_year  if fix_year  is not None else year_mean
    pred_df['age']       = fix_age   if fix_age   is not None else age_mean
    pred_df['education'] = fix_educ  if fix_educ  is not None else educ_mode
    pred_df[vary_col]    = grid

    # Use training DesignInfo so knots are from training data
    dm_pred = build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
    X_pred  = np.array(dm_pred)

    if logistic:
        preds = model.get_prediction(X_pred).summary_frame(alpha=0.05)
        yh, lo, hi = preds['predicted'].values, preds['ci_lower'].values, preds['ci_upper'].values
    else:
        preds = model.get_prediction(X_pred).summary_frame(alpha=0.05)
        yh, lo, hi = preds['mean'].values, preds['mean_ci_lower'].values, preds['mean_ci_upper'].values
    # Center: subtract mean so partial effect is around 0
    center = yh.mean()
    return grid, yh - center, lo - center, hi - center


# ── Partial effect for education ───────────────────────────────────────────────
def educ_partial_effect(model, design_info, wage_data, logistic=False):
    """
    For each education level, predict with year=mean, age=mean, educ=level.
    Returns (levels, yhat, yhat_lo, yhat_hi) — centered around 0.
    """
    year_mean = float(np.mean(year))
    age_mean  = float(np.mean(age))

    yhats, lo_list, hi_list = [], [], []
    for lv in educ_levels:
        pred_df = wage_data.iloc[:1].copy()
        pred_df['year']      = year_mean
        pred_df['age']       = age_mean
        pred_df['education'] = lv

        dm_pred = build_design_matrices([design_info], pred_df, return_type='dataframe')[0]
        X_pred  = np.array(dm_pred)

        if logistic:
            p = model.get_prediction(X_pred).summary_frame(alpha=0.05)
            yhats.append(p['predicted'].values[0])
            lo_list.append(p['ci_lower'].values[0])
            hi_list.append(p['ci_upper'].values[0])
        else:
            p = model.get_prediction(X_pred).summary_frame(alpha=0.05)
            yhats.append(p['mean'].values[0])
            lo_list.append(p['mean_ci_lower'].values[0])
            hi_list.append(p['mean_ci_upper'].values[0])

    yh = np.array(yhats)
    lo = np.array(lo_list)
    hi = np.array(hi_list)
    center = yh.mean()
    return educ_levels, yh - center, lo - center, hi - center


year_grid = np.linspace(year.min(), year.max(), 200)
age_grid  = np.linspace(age.min(),  age.max(),  200)


# ── Page 27: Regression ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
col = '#cc2222'
short_labels = ['<HS', 'HS', '<Coll', 'Coll', '>Coll']

# Panel 1: Year
xg, yh, lo, hi = partial_effect(m_reg, 'year', year_grid, wage_data, design_info)
axes[0].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[0].plot(xg, yh, color=col, linewidth=2)
axes[0].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[0].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[0].axhline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
# rug
axes[0].plot(year, np.full(len(year), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[0].set_xlabel('year', fontsize=9)
axes[0].set_ylabel(r'$f_1(\mathrm{year})$', fontsize=10)
axes[0].set_title('natural spline 4 dof', fontsize=8.5, color='#555', pad=4)
axes[0].tick_params(labelsize=8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel 2: Age
xg, yh, lo, hi = partial_effect(m_reg, 'age', age_grid, wage_data, design_info)
axes[1].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[1].plot(xg, yh, color=col, linewidth=2)
axes[1].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[1].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[1].axhline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
# rug
axes[1].plot(age, np.full(len(age), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[1].set_xlabel('age', fontsize=9)
axes[1].set_ylabel(r'$f_2(\mathrm{age})$', fontsize=10)
axes[1].set_title('natural spline 5 dof', fontsize=8.5, color='#555', pad=4)
axes[1].tick_params(labelsize=8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel 3: Education — dot + CI error bars
lvs, yh, lo, hi = educ_partial_effect(m_reg, design_info, wage_data, logistic=False)
x_pos = np.arange(len(lvs))
for i in range(len(lvs)):
    axes[2].plot([x_pos[i], x_pos[i]], [lo[i], hi[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [lo[i], lo[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [hi[i], hi[i]], color='#333', linewidth=1.2)
axes[2].scatter(x_pos, yh, color='#333', s=30, zorder=3)
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
plt.savefig('../gam-wage-regression.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../gam-wage-regression.svg')


# ── Page 32: Logistic ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
col = '#228833'
short_labels = ['<HS', 'HS', '<Coll', 'Coll', '>Coll']

# Panel 1: Year
xg, yh, lo, hi = partial_effect(m_log, 'year', year_grid, wage_data, design_info, logistic=True)
axes[0].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[0].plot(xg, yh, color=col, linewidth=2)
axes[0].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[0].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[0].plot(year, np.full(len(year), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[0].set_xlabel('year', fontsize=9)
axes[0].set_ylabel(r'$f_1(\mathrm{year})$', fontsize=10)
axes[0].set_title('natural spline 4 dof', fontsize=8.5, color='#555', pad=4)
axes[0].tick_params(labelsize=8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel 2: Age
xg, yh, lo, hi = partial_effect(m_log, 'age', age_grid, wage_data, design_info, logistic=True)
axes[1].fill_between(xg, lo, hi, color=col, alpha=0.12)
axes[1].plot(xg, yh, color=col, linewidth=2)
axes[1].plot(xg, lo, color=col, linewidth=0.8, linestyle='--')
axes[1].plot(xg, hi, color=col, linewidth=0.8, linestyle='--')
axes[1].plot(age, np.full(len(age), lo.min() - (hi.max()-lo.min())*0.06),
             '|', color='#555', alpha=0.3, markersize=3, markeredgewidth=0.6)
axes[1].set_xlabel('age', fontsize=9)
axes[1].set_ylabel(r'$f_2(\mathrm{age})$', fontsize=10)
axes[1].set_title('natural spline 5 dof', fontsize=8.5, color='#555', pad=4)
axes[1].tick_params(labelsize=8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel 3: Education — dot + CI error bars
lvs, yh, lo, hi = educ_partial_effect(m_log, design_info, wage_data, logistic=True)
x_pos = np.arange(len(lvs))
for i in range(len(lvs)):
    axes[2].plot([x_pos[i], x_pos[i]], [lo[i], hi[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [lo[i], lo[i]], color='#333', linewidth=1.2)
    axes[2].plot([x_pos[i]-0.15, x_pos[i]+0.15], [hi[i], hi[i]], color='#333', linewidth=1.2)
axes[2].scatter(x_pos, yh, color='#333', s=30, zorder=3)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(short_labels, fontsize=8)
axes[2].set_xlabel('education', fontsize=9)
axes[2].set_ylabel(r'$f_3(\mathrm{education})$', fontsize=10)
axes[2].set_title('constants for dummies', fontsize=8.5, color='#555', pad=4)
axes[2].tick_params(labelsize=8)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../gam-wage-logistic.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../gam-wage-logistic.svg')
