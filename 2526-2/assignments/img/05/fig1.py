"""Generate Figure 1 for assignment-05.
Function: f(x) = 2^(1-x), passing through (0,2), (1,1), (2,0.5), (3,0.25).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4.2, 3.4))

x = [0, 1, 2, 3]
y = [2, 1, 0.5, 0.25]     # piecewise linear through these points

ax.plot(x, y, color='#5566cc', linewidth=2)

# y-axis: linear scale, ticks at the specific values used in the original
yticks = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
ax.set_yticks(yticks)
ax.set_yticklabels([str(v) for v in yticks])
ax.set_ylim(-0.04, 2.18)

ax.set_xticks([0, 1, 2, 3])
ax.set_xlim(-0.05, 3.15)

ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)

# dashed horizontal gridlines
ax.yaxis.grid(True, linestyle='--', color='#ccc', linewidth=0.8)
ax.xaxis.grid(True, linestyle='--', color='#ccc', linewidth=0.8)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig1.svg', format='svg', bbox_inches='tight')
plt.close()
print('Saved fig1.svg')
