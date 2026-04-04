"""Euclidean vs correlation-based distance illustration."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)
t = np.linspace(0, 4 * np.pi, 50)

# Obs 1: high-amplitude sine
obs1 = 10 + 8 * np.sin(t)
# Obs 2: same shape, different offset (high corr, far Euclidean)
obs2 = 2 + 8 * np.sin(t)
# Obs 3: near obs1 in magnitude but different shape (close Euclidean, low corr)
obs3 = 10 + 8 * np.cos(t)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, (o1, o2, o3, title) in zip(axes, [
    (obs1, obs2, obs3, "Euclid: Obs 1 ≈ Obs 3  |  Obs 1 xa Obs 2"),
    (obs1/obs1.max(), obs2/obs2.max(), obs3/obs3.max(), "Tương quan: Obs 1 ≈ Obs 2  |  Obs 1 xa Obs 3"),
]):
    ax.plot(o1, color="#4a90d9", label="Obs 1", linewidth=2.0)
    ax.plot(o2, color="#e8732a", label="Obs 2", linewidth=2.0, linestyle="--")
    ax.plot(o3, color="#5aaa44", label="Obs 3", linewidth=2.0, linestyle=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Biến (feature index)", fontsize=9)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)

axes[0].set_ylabel("Giá trị")
fig.tight_layout(pad=1.0)
fig.savefig("../dissimilarity-comparison.svg", format="svg", bbox_inches="tight")
print("Saved dissimilarity-comparison.svg")
