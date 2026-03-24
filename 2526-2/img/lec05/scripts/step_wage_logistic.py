"""
Page 7: Step function logistic regression for Pr(Wage > 250 | Age).
Cutpoints [33.5, 49.0, 64.5]. Arrows all share same origin.
Output: ../step-wage-logistic.svg
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

wage_data = sm.datasets.get_rdataset('Wage', 'ISLR').data
age  = wage_data['age'].values
wage = wage_data['wage'].values
high = (wage > 250).astype(int)

cuts = [33.5, 49.0, 64.5]

def make_step_design(x, cutpoints):
    cp   = [-np.inf] + cutpoints + [np.inf]
    cols = [np.ones(len(x))]
    for k in range(1, len(cp) - 1):
        cols.append(((x >= cp[k]) & (x < cp[k+1])).astype(float))
    return np.column_stack(cols)

X     = make_step_design(age, cuts)
model = sm.Logit(high, X).fit(disp=0)

age_grid = np.linspace(18, 82, 600)
pred     = model.get_prediction(make_step_design(age_grid, cuts)).summary_frame(alpha=0.05)
p_fit    = pred['predicted'].values
p_lo     = pred['ci_lower'].values
p_hi     = pred['ci_upper'].values

fig, ax = plt.subplots(figsize=(6, 4))

ax.fill_between(age_grid, p_lo, p_hi, color='#aaaaaa', alpha=0.4, linewidth=0)
ax.step(age_grid, p_fit, where='post', color='#1a6b2e', linewidth=2)
ax.step(age_grid, p_lo,  where='post', color='#888888', linewidth=1, linestyle='--')
ax.step(age_grid, p_hi,  where='post', color='#888888', linewidth=1, linestyle='--')

# Rug plots
ax.plot(age[high == 1], np.full(high.sum(),    0.198), '|',
        color='#333333', alpha=0.5,  markersize=4, markeredgewidth=0.8)
ax.plot(age[high == 0], np.full((high==0).sum(), 0.002), '|',
        color='#333333', alpha=0.15, markersize=4, markeredgewidth=0.8)

for c in cuts:
    ax.axvline(c, color='#aaaaaa', linestyle=':', linewidth=1)

# All arrows from same origin
origin = (67.0, 0.155)
ax.text(origin[0] + 0.5, origin[1], 'cutpoints',
        color='#2266cc', fontsize=9, va='center', ha='left')
for c in cuts:
    ax.annotate('', xy=(c, 0.005), xytext=origin,
                arrowprops=dict(arrowstyle='->', color='#2266cc', lw=1.2))

ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Pr(Wage > 250 | Age)', fontsize=11)
ax.set_xlim(18, 82)
ax.set_ylim(-0.01, 0.22)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../step-wage-logistic.svg', format='svg', bbox_inches='tight')
print('Saved: ../step-wage-logistic.svg')
