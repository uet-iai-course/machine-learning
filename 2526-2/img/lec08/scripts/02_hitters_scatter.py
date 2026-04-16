"""Hitters-like dataset: scatter Years vs Hits with 3 regions.
Simulate data matching ISLR Hitters pattern since OpenML doesn't have it."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
n = 260

# Simulate Hitters-like data
years = np.random.exponential(5, n).clip(1, 24).astype(int)
hits = np.random.normal(100, 50, n).clip(1, 238).astype(int)
# Salary correlated with years and hits
salary = np.exp(4.5 + 0.08 * years + 0.005 * hits + np.random.normal(0, 0.5, n))
salary = salary.clip(60, 2500)

# Thresholds from ISLR
t_years = 4.5
t_hits = 117.5

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(years, hits, s=12, color="#d4a017", alpha=0.7, edgecolors="none")
ax.set_xlim(0.5, 25)
ax.set_ylim(0, 240)

# Partition lines
ax.plot([t_years, t_years], [0, 240], color="#2c3e50", linewidth=1.5)
ax.plot([t_years, 25], [t_hits, t_hits], color="#2c3e50", linewidth=1.5)

# Region means
r1 = salary[years < t_years]
r2 = salary[(years >= t_years) & (hits < t_hits)]
r3 = salary[(years >= t_years) & (hits >= t_hits)]

ax.text(2.5, 120, f"$R_1$\n\\${np.mean(r1):,.0f}", fontsize=9, ha="center", va="center",
        color="#c0392b", fontweight="bold")
ax.text(15, 55, f"$R_2$\n\\${np.mean(r2):,.0f}", fontsize=9, ha="center", va="center",
        color="#c0392b", fontweight="bold")
ax.text(18, 180, f"$R_3$\n\\${np.mean(r3):,.0f}", fontsize=9, ha="center", va="center",
        color="#c0392b", fontweight="bold")

ax.set_xlabel("Years", fontsize=11)
ax.set_ylabel("Hits", fontsize=11)
ax.text(t_years, -8, "4.5", fontsize=8, ha="center", va="top", color="#2c3e50")
ax.text(25.5, t_hits, "117.5", fontsize=8, ha="left", va="center", color="#2c3e50")

fig.tight_layout(pad=0.5)
fig.savefig("../hitters-scatter.svg", format="svg", bbox_inches="tight")
print("Saved hitters-scatter.svg")
