"""
Pages 8–11: Piecewise cubic series on ISLR Wage data.
All four figures share the same 100-point sample (random_state=10).

Outputs:
  ../piecewise-cubic-discontinuous.svg  (page 8)
  ../piecewise-cubic-continuous.svg     (page 9)
  ../cubic-spline.svg                   (page 10)
  ../linear-spline.svg                  (page 11)
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
sample    = wage_data.sample(100, random_state=10)
age_s     = sample['age'].values
wage_s    = sample['wage'].values
knot      = 50.0

def trunc_pos(x, xi, d):
    return np.where(x >= xi, (x - xi)**d, 0.0)

# ── Design matrices ───────────────────────────────────────────────────────────
def X_pw_disc(x):
    """Page 8: discontinuous piecewise cubic (8 params, no constraint at knot)"""
    left  = (x < knot).astype(float)
    right = (x >= knot).astype(float)
    return np.column_stack([
        left, left*x, left*x**2, left*x**3,
        right, right*x, right*x**2, right*x**3,
    ])

def X_c0(x):
    """Page 9: C0-continuous piecewise cubic (7 params, value-continuous at knot)"""
    return np.column_stack([
        np.ones(len(x)), x, x**2, x**3,
        trunc_pos(x, knot, 1), trunc_pos(x, knot, 2), trunc_pos(x, knot, 3)
    ])

def X_cs(x):
    """Page 10: cubic spline, C2-continuous (5 params)"""
    return np.column_stack([
        np.ones(len(x)), x, x**2, x**3, trunc_pos(x, knot, 3)
    ])

def X_ls(x):
    """Page 11: linear spline (3 params)"""
    return np.column_stack([
        np.ones(len(x)), x, trunc_pos(x, knot, 1)
    ])

age_grid = np.linspace(18, 80, 500)

def make_plot(X_fn, curve_color, out_path, discontinuous=False):
    model = sm.OLS(wage_s, X_fn(age_s)).fit()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(age_s, wage_s, s=22, facecolors='none', edgecolors='#aaaaaa',
               linewidths=0.8, zorder=1)
    if discontinuous:
        g_left  = age_grid[age_grid < knot]
        g_right = age_grid[age_grid >= knot]
        ax.plot(g_left,  model.predict(X_fn(g_left)),  color=curve_color, linewidth=2, zorder=3)
        ax.plot(g_right, model.predict(X_fn(g_right)), color=curve_color, linewidth=2, zorder=3)
    else:
        ax.plot(age_grid, model.predict(X_fn(age_grid)), color=curve_color, linewidth=2, zorder=3)
    ax.axvline(knot, color='#444444', linestyle='--', linewidth=1, zorder=2)
    ax.set_xlabel('Age', fontsize=11)
    ax.set_ylabel('Wage', fontsize=11)
    ax.set_xlim(17, 80)
    ax.set_ylim(20, 310)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close()
    print('Saved:', out_path)

make_plot(X_pw_disc, '#2244aa', '../piecewise-cubic-discontinuous.svg', discontinuous=True)
make_plot(X_c0,      '#2a8c4a', '../piecewise-cubic-continuous.svg')   # green
make_plot(X_cs,      '#e06060', '../cubic-spline.svg')                  # lighter red
make_plot(X_ls,      '#cc3333', '../linear-spline.svg')
