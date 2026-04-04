"""Generate Figure 1 and Figure 2 for Homework 07 (Problem 2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

# ── Figure 1: two interleaved moons → single linkage better
X1, _ = make_moons(n_samples=80, noise=0.08, random_state=3)
fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
ax1.scatter(X1[:, 0], X1[:, 1], s=55, color="#2222cc", edgecolors="none")
ax1.set_xlabel("X-axis"); ax1.set_ylabel("Y-axis")
ax1.set_xlim(-1.5, 2.5); ax1.set_ylim(-0.8, 1.4)
ax1.tick_params(labelsize=9)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
fig1.tight_layout()
fig1.savefig("fig1.svg", format="svg", bbox_inches="tight")
print("Saved fig1.svg")

# ── Figure 2: two compact blobs of different sizes → complete linkage better
rng = np.random.RandomState(7)
blob1 = rng.randn(60, 2) * 0.55 + np.array([0.0, 0.0])
blob2 = rng.randn(40, 2) * 0.80 + np.array([2.5, 3.2])
X2 = np.vstack([blob1, blob2])
fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
ax2.scatter(X2[:, 0], X2[:, 1], s=55, color="#2222cc", edgecolors="none")
ax2.set_xlabel("X-axis"); ax2.set_ylabel("Y-axis")
ax2.set_xlim(-2.5, 5.0); ax2.set_ylim(-2.5, 5.5)
ax2.tick_params(labelsize=9)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
fig2.tight_layout()
fig2.savefig("fig2.svg", format="svg", bbox_inches="tight")
print("Saved fig2.svg")
