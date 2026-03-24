"""
Page 22: Three kernel functions — Epanechnikov, Tri-cube, Gaussian.
Matches 08_slides.pdf page 22 figure.

Output: ../local-regression-kernel.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = np.linspace(-3.5, 3.5, 1000)

def epanechnikov(t):
    return np.where(np.abs(t) <= 1, 0.75 * (1 - t**2), 0.0)

def tricube(t):
    return np.where(np.abs(t) <= 1, (1 - np.abs(t)**3)**3, 0.0)

def gaussian(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.plot(t, epanechnikov(t), color='#e07820', linewidth=2,   label='Epanechnikov')
ax.plot(t, tricube(t),      color='#2a8c4a', linewidth=2,   label='Tri-cube')
ax.plot(t, gaussian(t),     color='#4488cc', linewidth=2,   label='Gaussian')

ax.set_xlabel(r'$t$',               fontsize=11)
ax.set_ylabel(r'$K_\lambda(x_0,\,x)$', fontsize=10)
ax.set_xlim(-3.3, 3.3)
ax.set_ylim(-0.01, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(labelsize=9)
ax.legend(fontsize=9, frameon=True, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../local-regression-kernel.svg', format='svg', bbox_inches='tight')
print('Saved: ../local-regression-kernel.svg')
