"""Shoppers buying profiles: socks vs computers for 8 customers."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
n = 8
socks = rng.randint(8, 60, n).astype(float)
computers = rng.randint(0, 4, n).astype(float)
customers = [f"C{i+1}" for i in range(n)]

x = np.arange(n)
width = 0.35
colors_s = plt.cm.tab10(np.linspace(0, 0.8, n))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

# Left: raw counts
ax = axes[0]
bars1 = ax.bar(x - width/2, socks, width, label="Tất (Socks)", color="#4a90d9", alpha=0.85)
bars2 = ax.bar(x + width/2, computers, width, label="Máy tính (Computers)", color="#e8732a", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(customers, fontsize=8)
ax.set_ylabel("Số lượng mua")
ax.set_title("")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Right: standardized (z-score)
socks_std = (socks - socks.mean()) / socks.std()
comp_std  = (computers - computers.mean()) / (computers.std() + 1e-9)
ax2 = axes[1]
ax2.bar(x - width/2, socks_std, width, label="Tất (chuẩn hóa)", color="#4a90d9", alpha=0.85)
ax2.bar(x + width/2, comp_std,  width, label="Máy tính (chuẩn hóa)", color="#e8732a", alpha=0.85)
ax2.axhline(0, color="#999", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(customers, fontsize=8)
ax2.set_ylabel("Giá trị chuẩn hóa (z-score)")
ax2.set_title("")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

fig.tight_layout(pad=0.8)
fig.savefig("../shoppers-profiles.svg", format="svg", bbox_inches="tight")
print("Saved shoppers-profiles.svg")
