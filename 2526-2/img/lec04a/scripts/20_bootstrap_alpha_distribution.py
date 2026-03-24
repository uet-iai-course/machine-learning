"""
Recreate bootstrap-alpha-distribution.png.

Left panel : histogram of 1000 alpha-hat estimates from repeated sampling
             + purple dashed vertical line at true alpha + stats table below.
Right panel: side-by-side boxplots (True vs Bootstrap)
             + purple dotted horizontal line at true alpha.

Parameters (ISLR): sigma_X^2=1, sigma_Y^2=1.25, sigma_XY=0.5 -> alpha=0.6
                   n=100, B=1000

Output: ../bootstrap-alpha-distribution.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Parameters ────────────────────────────────────────────────────────────────
sig2_X  = 1.0
sig2_Y  = 1.25
sig_XY  = 0.5
alpha   = (sig2_Y - sig_XY) / (sig2_X + sig2_Y - 2 * sig_XY)   # = 0.6
n       = 100
B       = 1000
cov_mat = [[sig2_X, sig_XY], [sig_XY, sig2_Y]]
mean    = [0, 0]

def alpha_hat(data):
    s2x  = np.var(data[:, 0], ddof=1)
    s2y  = np.var(data[:, 1], ddof=1)
    sxy  = np.cov(data[:, 0], data[:, 1])[0, 1]
    denom = s2x + s2y - 2 * sxy
    return (s2y - sxy) / denom if denom != 0 else np.nan

# ── True distribution: 1000 independent samples ───────────────────────────────
np.random.seed(0)
true_alphas = np.array([
    alpha_hat(np.random.multivariate_normal(mean, cov_mat, n))
    for _ in range(B)
])

# ── Bootstrap distribution: resample from one fixed dataset ───────────────────
np.random.seed(7)
original = np.random.multivariate_normal(mean, cov_mat, n)
boot_alphas = np.array([
    alpha_hat(original[np.random.choice(n, n, replace=True)])
    for _ in range(B)
])

alpha_hat_val = true_alphas.mean()
se_val        = true_alphas.std(ddof=1)

# ── Figure ────────────────────────────────────────────────────────────────────
PURPLE = '#8855cc'
BLUE   = '#7799bb'
GREEN  = '#66aa66'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2),
                                gridspec_kw={'width_ratios': [1.2, 1]})

# ── Left: histogram ───────────────────────────────────────────────────────────
ax1.hist(true_alphas, bins=30, color=BLUE, alpha=0.75, edgecolor='white', linewidth=0.4)
counts, bin_edges = np.histogram(true_alphas, bins=30)
peak_count = counts.max()
ax1.axvline(alpha, color=PURPLE, linestyle='--', linewidth=1.6)
ax1.annotate('true value',
             xy=(alpha, peak_count * 0.97),
             xytext=(alpha + 0.07, peak_count * 0.88),
             fontsize=8, color=PURPLE,
             arrowprops=dict(arrowstyle='->', color=PURPLE, lw=0.9))

ax1.set_xlabel(r'$\hat{\alpha}$', fontsize=10)
ax1.set_title(r'Thousand estimates of $\alpha$', fontsize=9, color='#333')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=8)
ax1.set_xlim(0.25, 0.95)

# Stats table below the plot
stats_text = (
    r'$\sigma_X^2 = 1$' + '          ' + r'$\hat{\alpha} = %.4f$' % alpha_hat_val + '\n'
    r'$\sigma_Y^2 = 1.25$' + '     ' + r'$SE(\hat{\alpha}) = %.3f$' % se_val + '\n'
    r'$\sigma_{XY} = 0.5$' + '\n'
    r'$\alpha = 0.6$'
)
ax1.text(0.02, -0.28, stats_text,
         transform=ax1.transAxes, fontsize=8.5, color='#333',
         va='top', linespacing=1.7)

# ── Right: boxplots ───────────────────────────────────────────────────────────
bp_true = ax2.boxplot(true_alphas, positions=[1], widths=0.45,
                      patch_artist=True, manage_ticks=False,
                      medianprops=dict(color='black', linewidth=2),
                      boxprops=dict(facecolor=BLUE, alpha=0.7),
                      whiskerprops=dict(linewidth=1),
                      capprops=dict(linewidth=1),
                      flierprops=dict(marker='+', markersize=4, color='#555'))

bp_boot = ax2.boxplot(boot_alphas, positions=[2], widths=0.45,
                      patch_artist=True, manage_ticks=False,
                      medianprops=dict(color='black', linewidth=2),
                      boxprops=dict(facecolor=GREEN, alpha=0.7,
                                    linestyle='--', linewidth=1.2),
                      whiskerprops=dict(linewidth=1, linestyle='--'),
                      capprops=dict(linewidth=1),
                      flierprops=dict(marker='+', markersize=4, color='#555'))

ax2.axhline(alpha, color=PURPLE, linestyle=':', linewidth=1.6)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['True', 'Bootstrap'], fontsize=9)
ax2.set_ylabel(r'$\alpha$', fontsize=11, rotation=0, labelpad=10)
ax2.set_title('Simulated data vs. Bootstrap', fontsize=9, color='#333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=8)
ax2.set_xlim(0.4, 2.6)
ax2.set_ylim(0.25, 0.95)

plt.tight_layout()
plt.savefig('../bootstrap-alpha-distribution.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved: ../bootstrap-alpha-distribution.svg')
