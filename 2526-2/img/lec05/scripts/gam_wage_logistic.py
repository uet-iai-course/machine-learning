"""
GAM logistic figures — pages 32 and 33 of 08_slides.pdf.

Page 32 (gam-wage-logistic.svg):
  Logistic GAM on all Wage data.
  year = linear term, age = natural spline df=5, education = dummies.
  <HS has complete separation (no <HS earns >$250K) → huge CI bar.

Page 33 (gam-wage-logistic-nohs.svg):
  Same model refit after excluding <HS education.

Color: green (#228833). Partial effects on logit scale, centered at 0.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patsy import dmatrix, build_design_matrices

# ── Data ───────────────────────────────────────────────────────────────────────
raw = sm.datasets.get_rdataset('Wage', 'ISLR').data.copy()
raw['high_earner'] = (raw['wage'] > 250).astype(int)

COL = '#228833'
SHORT = {
    '1. < HS Grad': '<HS', '2. HS Grad': 'HS',
    '3. Some College': '<Coll', '4. College Grad': 'Coll',
    '5. Advanced Degree': '>Coll',
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def make_figure(data, outpath, annotate_hs=False):
    year = data['year'].values.astype(float)
    age  = data['age'].values.astype(float)
    educ_levels  = sorted(data['education'].unique().tolist())
    short_labels = [SHORT[l] for l in educ_levels]
    mode_educ    = data['education'].mode()[0]

    # Build RHS design matrix (intercept included by patsy default)
    formula     = "year + cr(age, df=5) + C(education)"
    dm_train    = dmatrix(formula, data, return_type='dataframe')
    design_info = dm_train.design_info

    # Fit logistic regression on logit scale
    result = sm.Logit(data['high_earner'].values,
                      np.array(dm_train)).fit(disp=0, maxiter=300)
    params = result.params
    cov    = result.cov_params()

    year_grid = np.linspace(year.min(), year.max(), 200)
    age_grid  = np.linspace(age.min(),  age.max(),  200)

    def lp_with_ci(pred_df):
        """Return (linear_predictor, lower_95, upper_95) for rows in pred_df."""
        dm_p = build_design_matrices([design_info], pred_df,
                                     return_type='dataframe')[0]
        X    = np.array(dm_p)
        lp   = X @ params
        var  = np.einsum('ij,jk,ik->i', X, cov, X)
        se   = np.sqrt(np.clip(var, 0, None))
        return lp, lp - 1.96 * se, lp + 1.96 * se

    def partial_effect(vary_col, grid):
        n      = len(grid)
        pred_df = pd.DataFrame({
            'year':      np.full(n, np.mean(year)),
            'age':       np.full(n, np.mean(age)),
            'education': mode_educ,
        })
        pred_df[vary_col] = grid
        lp, lo, hi = lp_with_ci(pred_df)
        c = lp.mean()
        return grid, lp - c, lo - c, hi - c

    def educ_partial_effect():
        yhats, los, his = [], [], []
        for lv in educ_levels:
            pred_df = pd.DataFrame({
                'year':      [np.mean(year)],
                'age':       [np.mean(age)],
                'education': [lv],
            })
            lp, lo, hi = lp_with_ci(pred_df)
            yhats.append(lp[0]); los.append(lo[0]); his.append(hi[0])
        yh = np.array(yhats); lo = np.array(los); hi = np.array(his)
        c  = yh.mean()
        return educ_levels, yh - c, lo - c, hi - c

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    # Panel 1 — Year (linear)
    xg, yh, lo, hi = partial_effect('year', year_grid)
    axes[0].fill_between(xg, lo, hi, color=COL, alpha=0.12)
    axes[0].plot(xg, yh, color=COL, lw=2)
    axes[0].plot(xg, lo, color=COL, lw=0.8, ls=':')
    axes[0].plot(xg, hi, color=COL, lw=0.8, ls=':')
    axes[0].axhline(0, color='#bbb', lw=0.6, ls=':')
    rug_y = lo.min() - (hi.max() - lo.min()) * 0.06
    axes[0].plot(year, np.full(len(year), rug_y),
                 '|', color='#555', alpha=0.3, ms=3, mew=0.6)
    axes[0].set_xlabel('year', fontsize=9)
    axes[0].set_ylabel(r'$f_1(\mathrm{year})$', fontsize=10)
    axes[0].set_title('linear function in year', fontsize=8.5, color='#555', pad=4)
    style_ax(axes[0])

    # Panel 2 — Age (natural spline 5 dof)
    xg, yh, lo, hi = partial_effect('age', age_grid)
    axes[1].fill_between(xg, lo, hi, color=COL, alpha=0.12)
    axes[1].plot(xg, yh, color=COL, lw=2)
    axes[1].plot(xg, lo, color=COL, lw=0.8, ls=':')
    axes[1].plot(xg, hi, color=COL, lw=0.8, ls=':')
    axes[1].axhline(0, color='#bbb', lw=0.6, ls=':')
    rug_y = lo.min() - (hi.max() - lo.min()) * 0.06
    axes[1].plot(age, np.full(len(age), rug_y),
                 '|', color='#555', alpha=0.3, ms=3, mew=0.6)
    axes[1].set_xlabel('age', fontsize=9)
    axes[1].set_ylabel(r'$f_2(\mathrm{age})$', fontsize=10)
    axes[1].set_title('smoothing spline 5 dof', fontsize=8.5, color='#555', pad=4)
    style_ax(axes[1])

    # Panel 3 — Education
    lvs, yh_e, lo_e, hi_e = educ_partial_effect()
    x_pos = np.arange(len(lvs))
    for i in range(len(lvs)):
        ls = '--' if (annotate_hs and i == 0) else '-'
        axes[2].plot([x_pos[i]] * 2, [lo_e[i], hi_e[i]],
                     color='#333', lw=1.2, ls=ls)
        for y_cap in [lo_e[i], hi_e[i]]:
            axes[2].plot([x_pos[i] - 0.15, x_pos[i] + 0.15], [y_cap, y_cap],
                         color='#333', lw=1.2)
    axes[2].scatter(x_pos, yh_e, color='#333', s=28, zorder=3)
    axes[2].axhline(0, color='#bbb', lw=0.6, ls=':')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(short_labels, fontsize=8)
    axes[2].set_xlabel('education', fontsize=9)
    axes[2].set_ylabel(r'$f_3(\mathrm{education})$', fontsize=10)
    axes[2].set_title('constants for dummies', fontsize=8.5, color='#555', pad=4)
    style_ax(axes[2])

    if annotate_hs and '1. < HS Grad' in educ_levels:
        hs_idx = educ_levels.index('1. < HS Grad')
        # Annotate pointing at the top of the <HS CI bar
        axes[2].annotate(
            'No <HS individual\nearns >$250K',
            xy=(x_pos[hs_idx], hi_e[hs_idx]),
            xytext=(x_pos[hs_idx] + 1.2, hi_e[hs_idx] * 0.6),
            fontsize=7.5, color='#cc2222',
            arrowprops=dict(arrowstyle='->', color='#cc2222', lw=0.9),
            ha='left',
        )

    plt.tight_layout()
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


# Page 32 — all data (with <HS)
make_figure(raw, '../gam-wage-logistic.svg', annotate_hs=True)

# Page 33 — exclude <HS
data_no_hs = raw[raw['education'] != '1. < HS Grad'].copy()
make_figure(data_no_hs, '../gam-wage-logistic-nohs.svg', annotate_hs=False)
