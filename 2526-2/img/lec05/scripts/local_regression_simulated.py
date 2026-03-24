"""
Pages 20-21 & 23: Local regression on simulated data.

Two figures share the same scatter/true/fitted curves; only x0 differs.

Outputs:
  ../local-regression-left.svg   (page 20, x0 = 0.1)
  ../local-regression-mid.svg    (page 21 / 23 sidebar, x0 = 0.45)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(7)
n = 100
x = np.sort(np.random.uniform(0, 1, n))

def f_true(x):
    return np.sin(np.pi * x) - 0.6 * x

y = f_true(x) + np.random.normal(0, 0.25, n)

# ── Tricube kernel ─────────────────────────────────────────────────────────────
def tricube(u):
    u = np.abs(u)
    return np.where(u < 1, (1 - u**3)**3, 0.0)

# ── LOESS fitted curve ─────────────────────────────────────────────────────────
def loess_fit(x_data, y_data, x_grid, span=0.35):
    fitted = []
    h = span
    for x0 in x_grid:
        d = (x_data - x0) / h
        w = tricube(d)
        if w.sum() < 1e-10:
            fitted.append(np.nan)
            continue
        A = np.column_stack([np.ones(len(x_data)), x_data - x0])
        b = np.linalg.solve(A.T @ np.diag(w) @ A, A.T @ (w * y_data))
        fitted.append(b[0])
    return np.array(fitted)

x_grid        = np.linspace(0, 1, 300)
y_true_grid   = f_true(x_grid)
y_fitted_grid = loess_fit(x, y, x_grid, span=0.35)

# ── Local linear fit at x0 ─────────────────────────────────────────────────────
def local_fit_at(x0, span=0.35):
    h = span
    d = (x - x0) / h
    w = tricube(d)
    in_band = w > 0
    A = np.column_stack([np.ones(n), x - x0])
    b = np.linalg.solve(A.T @ np.diag(w) @ A, A.T @ (w * y))
    x_seg = np.linspace(max(0, x0 - h), min(1, x0 + h), 200)
    y_seg = b[0] + b[1] * (x_seg - x0)
    return w, in_band, x_seg, y_seg, b[0]   # b[0] = fitted value at x0

# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(x0, out_path, span=0.35):
    w, in_band, x_seg, y_seg, y_hat = local_fit_at(x0, span)

    # Figure with extra right margin for legend
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.set_title('Local regression on simulated data', fontsize=9)

    # ── Kernel weight fill (from y=0 upward, peak ~0.9 in data coords) ────────
    h = span
    x_ker = np.linspace(max(0, x0 - h), min(1, x0 + h), 300)
    k_ker = tricube((x_ker - x0) / h) * 0.92   # scale so peak ≈ 0.9
    ax.fill_between(x_ker, 0, k_ker, color='#c8d820', alpha=0.75, zorder=1)

    # ── True curve (blue) ──────────────────────────────────────────────────────
    ax.plot(x_grid, y_true_grid,   color='#3377cc', linewidth=2.2, zorder=4)
    # ── Fitted LOESS curve (orange/gold) ──────────────────────────────────────
    ax.plot(x_grid, y_fitted_grid, color='#f5a623', linewidth=2.2, zorder=4)
    # ── Local linear segment (dark orange) ────────────────────────────────────
    ax.plot(x_seg, y_seg,          color='#aa3300', linewidth=2.0, zorder=5)

    # ── Scatter: gray open circles (out of window) ────────────────────────────
    ax.scatter(x[~in_band], y[~in_band],
               s=22, facecolors='none', edgecolors='#999999',
               linewidths=0.8, zorder=2)
    # ── Scatter: red open circles (in window) — SAME SIZE as gray ─────────────
    ax.scatter(x[in_band], y[in_band],
               s=22, facecolors='none', edgecolors='#aa3300',
               linewidths=0.9, zorder=3)

    # ── Vertical line at x0 ───────────────────────────────────────────────────
    ax.axvline(x0, color='#aa3300', linewidth=1.2, zorder=6)
    # ── Filled dot at fitted value ────────────────────────────────────────────
    ax.scatter([x0], [y_hat], s=45, color='#aa3300', zorder=7)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-1.22, 1.65)
    ax.tick_params(labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(True)

    # ── Legend inside axes (upper right) ──────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='#3377cc', linewidth=2,   label=r'true curve $f(x)$'),
        Line2D([0], [0], color='#f5a623', linewidth=2,   label='fitted curve'),
        Line2D([0], [0], color='#aa3300', linewidth=2,   label=r'fitted linear regression at test point $x_0$'),
        Patch(facecolor='#c8d820', alpha=0.75,           label=r'weights of the neighbors of the test point $x_0$'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='none', markeredgecolor='#aa3300',
               markersize=6, linewidth=0,                label='Neighbors whose weights are nonzero'),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, frameon=True,
              framealpha=0.9, edgecolor='#cccccc',
              loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close()
    print('Saved:', out_path)

make_figure(x0=0.10, out_path='../local-regression-left.svg')
make_figure(x0=0.45, out_path='../local-regression-mid.svg')
