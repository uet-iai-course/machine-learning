"""
Recreate bootstrap-repeated-sampling.png.

4 scatter plots of 100 samples drawn from a bivariate normal distribution
(ISLR investment example: sigma_X^2=1, sigma_Y^2=1.25, sigma_XY=0.5).
Caption: "What are the estimates of alpha when we repeatedly draw 100 samples?"

Output: ../bootstrap-repeated-sampling.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Covariance matrix matching ISLR parameters
# sigma_X^2 = 1, sigma_Y^2 = 1.25, sigma_XY = 0.5
cov = [[1.0, 0.5],
       [0.5, 1.25]]
mean = [0, 0]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))

for i, ax in enumerate(axes):
    np.random.seed(i)
    data = np.random.multivariate_normal(mean, cov, size=100)
    ax.scatter(data[:, 0], data[:, 1],
               color='#22aa66', s=18, alpha=0.75, linewidths=0)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)

plt.tight_layout()
plt.savefig('../bootstrap-repeated-sampling.svg', format='svg',
            bbox_inches='tight')
plt.close()
print('Saved: ../bootstrap-repeated-sampling.svg')
