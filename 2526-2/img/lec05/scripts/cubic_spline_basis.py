"""
Page 12: Basis functions of a cubic spline with 3 knots at 0.1, 0.3, 0.6.
- 1, x, x², x³  : solid curves
- (x-ξ)³₊       : dashed curves
- Labels placed on each curve, rotated to match slope, no legend box.
- Full box (all 4 spines visible), no vertical knot lines.
Output: ../cubic-spline-basis.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x     = np.linspace(0, 1, 500)
knots = [0.1, 0.3, 0.6]

def trunc_cubic(x, xi):
    return np.where(x >= xi, (x - xi)**3, 0.0)

fig, ax = plt.subplots(figsize=(6, 4.5))

curves = [
    ('poly', np.ones_like(x),      '#2255dd', r'$1$',            0.15),
    ('poly', x,                    '#e07800', r'$x$',            0.28),
    ('poly', x**2,                 '#2a8c2a', r'$x^2$',          0.38),
    ('poly', x**3,                 '#aa1111', r'$x^3$',          0.50),
    ('tc',   trunc_cubic(x, 0.1),  '#6644cc', r'$(x-0.1)^3_+$', 0.60),
    ('tc',   trunc_cubic(x, 0.3),  '#885500', r'$(x-0.3)^3_+$', 0.73),
    ('tc',   trunc_cubic(x, 0.6),  '#cc44aa', r'$(x-0.6)^3_+$', 0.88),
]

for kind, yv, col, lbl, lx in curves:
    ax.plot(x, yv, color=col, linewidth=2, linestyle='--' if kind == 'tc' else '-')

ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.08)
plt.tight_layout()

def curve_angle(xvals, yvals, x0):
    idx = np.clip(np.searchsorted(xvals, x0), 1, len(xvals)-1)
    dx, dy = xvals[idx]-xvals[idx-1], yvals[idx]-yvals[idx-1]
    ax_bbox = ax.get_position()
    fw, fh  = fig.get_size_inches()
    xr = ax.get_xlim()[1]-ax.get_xlim()[0]
    yr = ax.get_ylim()[1]-ax.get_ylim()[0]
    slope_d = (dy/yr) / (dx/xr) * (ax_bbox.height*fh / (ax_bbox.width*fw))
    return np.clip(np.degrees(np.arctan(slope_d)), -60, 60)

for kind, yv, col, lbl, lx in curves:
    ly  = yv[np.searchsorted(x, lx)]
    ang = curve_angle(x, yv, lx)
    ax.annotate(lbl, xy=(lx, ly), xytext=(4, 4), textcoords='offset points',
                color=col, fontsize=8.5,
                rotation=ang, rotation_mode='anchor', ha='left', va='bottom')

ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('Basis function value', fontsize=10)
ax.tick_params(labelsize=9)

plt.savefig('../cubic-spline-basis.svg', format='svg', bbox_inches='tight')
print('Saved: ../cubic-spline-basis.svg')
